"""
MCP Router - Model Context Protocol server and tool management.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models import MCPServer, MCPTool


router = APIRouter(
    prefix="/mcp",
    tags=["mcp"],
    responses={404: {"description": "Not found"}},
)


class MCPToolResponse(BaseModel):
    id: int
    server_id: int
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool
    last_seen_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MCPServerResponse(BaseModel):
    id: int
    name: str
    url: str
    transport: str
    description: Optional[str] = None
    headers: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool
    last_status: str
    last_error: Optional[str] = None
    last_checked_at: Optional[datetime] = None
    tools: List[MCPToolResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class MCPServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    url: str = Field(..., min_length=1, max_length=1000)
    transport: str = "streamable_http"
    description: Optional[str] = None
    headers: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True


class MCPServerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    url: Optional[str] = Field(None, min_length=1, max_length=1000)
    transport: Optional[str] = None
    description: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    is_enabled: Optional[bool] = None


class MCPToolUpdate(BaseModel):
    is_enabled: bool


class MCPTestResponse(BaseModel):
    ok: bool
    status: str
    message: str
    server_id: Optional[int] = None
    tried_urls: List[str] = Field(default_factory=list)
    tools: List[Dict[str, Any]] = Field(default_factory=list)


def _serialize_server(server: MCPServer) -> MCPServerResponse:
    return MCPServerResponse(
        id=server.id,
        name=server.name,
        url=server.url,
        transport=server.transport or "streamable_http",
        description=server.description,
        headers=server.headers or {},
        is_enabled=bool(server.is_enabled),
        last_status=server.last_status or "UNKNOWN",
        last_error=server.last_error,
        last_checked_at=server.last_checked_at,
        tools=[
            MCPToolResponse(
                id=tool.id,
                server_id=tool.server_id,
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema or {},
                is_enabled=bool(tool.is_enabled),
                last_seen_at=tool.last_seen_at,
            )
            for tool in sorted(server.tools or [], key=lambda item: item.name)
        ],
    )


def _candidate_urls(raw_url: str) -> List[str]:
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return [raw_url]

    urls = []
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


def _json_rpc(method: str, params: Optional[Dict[str, Any]] = None, request_id: int = 1) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return payload


def _extract_json_response(response: httpx.Response) -> Dict[str, Any]:
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


async def _post_mcp(
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
    parsed = _extract_json_response(response)
    next_session_id = response.headers.get("mcp-session-id") or response.headers.get("Mcp-Session-Id") or session_id
    return parsed, next_session_id


async def _probe_mcp_server(url: str, headers: Dict[str, Any]) -> MCPTestResponse:
    errors = []
    tried_urls = []
    initialize_payload = _json_rpc(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "AIAC Web MCP Manager", "version": "1.0.0"},
        },
        request_id=1,
    )

    async with httpx.AsyncClient(timeout=12.0) as client:
        for candidate in _candidate_urls(url):
            tried_urls.append(candidate)
            session_id = None
            try:
                initialize_result, session_id = await _post_mcp(client, candidate, initialize_payload, headers)
                if initialize_result.get("error"):
                    raise ValueError(initialize_result["error"])

                tools_result, _ = await _post_mcp(
                    client,
                    candidate,
                    _json_rpc("tools/list", request_id=2),
                    headers,
                    session_id=session_id,
                )
                if tools_result.get("error"):
                    raise ValueError(tools_result["error"])

                tools = tools_result.get("result", {}).get("tools", [])
                return MCPTestResponse(
                    ok=True,
                    status="CONNECTED",
                    message=f"Connected. Discovered {len(tools)} tools.",
                    tried_urls=tried_urls,
                    tools=tools,
                )
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

    return MCPTestResponse(
        ok=False,
        status="FAILED",
        message="; ".join(errors)[:1000] or "Connection failed",
        tried_urls=tried_urls,
        tools=[],
    )


async def _get_server_or_404(db: AsyncSession, server_id: int) -> MCPServer:
    result = await db.execute(
        select(MCPServer).options(selectinload(MCPServer.tools)).where(MCPServer.id == server_id)
    )
    server = result.scalar_one_or_none()
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return server


async def _upsert_tools(db: AsyncSession, server: MCPServer, tools: List[Dict[str, Any]]) -> None:
    existing_by_name = {tool.name: tool for tool in server.tools or []}
    now = datetime.utcnow()
    for tool_data in tools:
        name = tool_data.get("name")
        if not name:
            continue
        tool = existing_by_name.get(name)
        if tool is None:
            tool = MCPTool(server_id=server.id, name=name, is_enabled=True)
            db.add(tool)
        tool.description = tool_data.get("description")
        tool.input_schema = tool_data.get("inputSchema") or tool_data.get("input_schema") or {}
        tool.last_seen_at = now


@router.get("/servers", response_model=List[MCPServerResponse])
async def list_servers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MCPServer).options(selectinload(MCPServer.tools)).order_by(MCPServer.name))
    return [_serialize_server(server) for server in result.scalars().all()]


@router.post("/servers", response_model=MCPServerResponse)
async def create_server(request: MCPServerCreate, db: AsyncSession = Depends(get_db)):
    server = MCPServer(
        name=request.name,
        url=request.url,
        transport=request.transport,
        description=request.description,
        headers=request.headers,
        is_enabled=request.is_enabled,
    )
    db.add(server)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create MCP server: {exc}") from exc
    return _serialize_server(await _get_server_or_404(db, server.id))


@router.put("/servers/{server_id}", response_model=MCPServerResponse)
async def update_server(server_id: int, request: MCPServerUpdate, db: AsyncSession = Depends(get_db)):
    server = await _get_server_or_404(db, server_id)
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(server, key, value)
    await db.commit()
    return _serialize_server(await _get_server_or_404(db, server_id))


@router.delete("/servers/{server_id}")
async def delete_server(server_id: int, db: AsyncSession = Depends(get_db)):
    server = await _get_server_or_404(db, server_id)
    await db.delete(server)
    await db.commit()
    return {"success": True}


@router.post("/servers/{server_id}/test", response_model=MCPTestResponse)
async def test_server(server_id: int, db: AsyncSession = Depends(get_db)):
    server = await _get_server_or_404(db, server_id)
    result = await _probe_mcp_server(server.url, server.headers or {})
    server.last_status = result.status
    server.last_error = None if result.ok else result.message
    server.last_checked_at = datetime.utcnow()
    if result.ok:
        await _upsert_tools(db, server, result.tools)
    await db.commit()
    result.server_id = server.id
    return result


@router.post("/test", response_model=MCPTestResponse)
async def test_raw_server(request: MCPServerCreate):
    return await _probe_mcp_server(request.url, request.headers or {})


@router.post("/servers/{server_id}/refresh-tools", response_model=MCPTestResponse)
async def refresh_tools(server_id: int, db: AsyncSession = Depends(get_db)):
    return await test_server(server_id, db)


@router.put("/tools/{tool_id}", response_model=MCPToolResponse)
async def update_tool(tool_id: int, request: MCPToolUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MCPTool).where(MCPTool.id == tool_id))
    tool = result.scalar_one_or_none()
    if not tool:
        raise HTTPException(status_code=404, detail="MCP tool not found")
    tool.is_enabled = request.is_enabled
    await db.commit()
    await db.refresh(tool)
    return MCPToolResponse(
        id=tool.id,
        server_id=tool.server_id,
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema or {},
        is_enabled=bool(tool.is_enabled),
        last_seen_at=tool.last_seen_at,
    )
