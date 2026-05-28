"""
Runtime MCP tool access for mining workflows.

The web UI owns MCP server/tool configuration. This service reads the enabled
registry from the database and calls tools with JSON-RPC over streamable HTTP.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.database import AsyncSessionLocal
from backend.models import MCPServer, MCPTool


@dataclass
class EnabledMCPTool:
    """Enabled MCP tool with its parent server connection details."""

    id: int
    name: str
    server_id: int
    server_name: str
    url: str
    headers: Dict[str, Any]
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


def candidate_mcp_urls(raw_url: str) -> List[str]:
    """Return URLs reachable from Docker for a user-configured MCP endpoint."""
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return [raw_url]

    urls: List[str] = []
    if parsed.hostname in {"localhost", "127.0.0.1", "172.0.0.1"}:
        hosts = ("172.17.0.1", "host.docker.internal") if parsed.hostname == "172.0.0.1" else (
            "host.docker.internal",
            "172.17.0.1",
        )
        for host in hosts:
            netloc = host
            if parsed.port:
                netloc = f"{host}:{parsed.port}"
            candidate = urlunparse(parsed._replace(netloc=netloc))
            if candidate not in urls:
                urls.append(candidate)
    if raw_url not in urls:
        urls.append(raw_url)
    return urls


def json_rpc_payload(method: str, params: Optional[Dict[str, Any]] = None, request_id: int = 1) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return payload


def extract_json_response(response: httpx.Response) -> Dict[str, Any]:
    """Parse regular JSON or SSE-style MCP JSON-RPC responses."""
    content_type = response.headers.get("content-type", "")
    text = response.text.strip()
    if "text/event-stream" in content_type or text.startswith("event:") or text.startswith("data:"):
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    return json.loads(data)
        raise ValueError("SSE response did not include JSON data")
    return response.json()


def decode_mcp_tool_result(result: Dict[str, Any]) -> Any:
    """
    Decode a tools/call JSON-RPC response into native Python data.

    MCP tool responses commonly put JSON text inside result.content[].text.
    """
    if result.get("error"):
        raise RuntimeError(str(result["error"]))

    payload = result.get("result", result)
    if isinstance(payload, dict) and payload.get("isError"):
        raise RuntimeError(str(payload))

    content = payload.get("content") if isinstance(payload, dict) else None
    if isinstance(content, list) and content:
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text") is not None
        ]
        if texts:
            text = "\n".join(str(item) for item in texts).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

    return payload


class MCPToolClient:
    """Dynamic MCP tool client backed by the web-managed registry."""

    def __init__(self, timeout_seconds: float = 960.0):
        self.timeout = httpx.Timeout(timeout_seconds, connect=5.0, write=30.0, pool=12.0)

    async def list_enabled_tools(self) -> List[EnabledMCPTool]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MCPTool)
                .join(MCPServer)
                .options(selectinload(MCPTool.server))
                .where(MCPServer.is_enabled == True, MCPTool.is_enabled == True)
                .order_by(MCPServer.name, MCPTool.name)
            )
            tools: List[EnabledMCPTool] = []
            for tool in result.scalars().all():
                server = tool.server
                if not server:
                    continue
                tools.append(
                    EnabledMCPTool(
                        id=tool.id,
                        name=tool.name,
                        server_id=server.id,
                        server_name=server.name,
                        url=server.url,
                        headers=server.headers or {},
                        description=tool.description,
                        input_schema=tool.input_schema or {},
                    )
                )
            return tools

    async def get_enabled_tool(self, name: str) -> Optional[EnabledMCPTool]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MCPTool)
                .join(MCPServer)
                .options(selectinload(MCPTool.server))
                .where(
                    MCPServer.is_enabled == True,
                    MCPTool.is_enabled == True,
                    MCPTool.name == name,
                )
                .order_by(MCPServer.name, MCPTool.id)
                .limit(1)
            )
            tool = result.scalar_one_or_none()
            if not tool or not tool.server:
                return None
            server = tool.server
            return EnabledMCPTool(
                id=tool.id,
                name=tool.name,
                server_id=server.id,
                server_name=server.name,
                url=server.url,
                headers=server.headers or {},
                description=tool.description,
                input_schema=tool.input_schema or {},
            )

    async def is_tool_enabled(self, name: str) -> bool:
        return await self.get_enabled_tool(name) is not None

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        tool = await self.get_enabled_tool(name)
        if not tool:
            raise LookupError(f"MCP tool is disabled or unavailable: {name}")

        last_error: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for url in candidate_mcp_urls(tool.url):
                session_id = None
                try:
                    initialize_result, session_id = await self._post(
                        client,
                        url,
                        json_rpc_payload(
                            "initialize",
                            {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "AIAC Mining Runtime", "version": "1.0.0"},
                            },
                            request_id=1,
                        ),
                        tool.headers,
                    )
                    if initialize_result.get("error"):
                        raise RuntimeError(str(initialize_result["error"]))

                    call_result, _ = await self._post(
                        client,
                        url,
                        json_rpc_payload(
                            "tools/call",
                            {"name": name, "arguments": arguments or {}},
                            request_id=2,
                        ),
                        tool.headers,
                        session_id=session_id,
                    )

                    await self._mark_server_status(tool.server_id, "CONNECTED", None)
                    return decode_mcp_tool_result(call_result)
                except Exception as exc:
                    last_error = exc
                    logger.debug(f"MCP tool call failed | tool={name} url={url} error={exc}")

        message = str(last_error) if last_error else "Unknown MCP tool call failure"
        await self._mark_server_status(tool.server_id, "FAILED", message[:1000])
        raise RuntimeError(f"MCP tool call failed for {name}: {message}") from last_error

    async def _post(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        request_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **{str(k): str(v) for k, v in (headers or {}).items() if v is not None},
        }
        if session_id:
            request_headers["Mcp-Session-Id"] = session_id

        response = await client.post(url, json=payload, headers=request_headers)
        response.raise_for_status()
        parsed = extract_json_response(response)
        next_session_id = response.headers.get("mcp-session-id") or response.headers.get("Mcp-Session-Id") or session_id
        return parsed, next_session_id

    async def _mark_server_status(self, server_id: int, status: str, error: Optional[str]) -> None:
        try:
            async with AsyncSessionLocal() as db:
                server = await db.get(MCPServer, server_id)
                if not server:
                    return
                server.last_status = status
                server.last_error = error
                server.last_checked_at = datetime.utcnow()
                await db.commit()
        except Exception as exc:
            logger.debug(f"Failed to update MCP server status: {exc}")
