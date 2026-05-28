"""
MCP runtime nodes for LangGraph mining.
"""

import time
from typing import Dict

from langchain_core.runnables import RunnableConfig
from loguru import logger

from backend.agents.graph.state import MiningState
from backend.agents.graph.nodes.base import record_trace
from backend.mcp_tool_service import MCPToolClient


async def node_load_mcp_tools(
    state: MiningState,
    config: RunnableConfig = None,
) -> Dict:
    """
    Load enabled MCP tools for this graph run.

    Actual tool calls still check the database live through MCPBrainAdapter, so
    web UI toggles can affect long-running mining without restarting workers.
    """
    start_time = time.time()
    node_name = "MCP_TOOLS"
    trace_service = config.get("configurable", {}).get("trace_service") if config else None

    try:
        client = MCPToolClient()
        enabled_tools = await client.list_enabled_tools()
        tool_summaries = [
            {
                "id": tool.id,
                "name": tool.name,
                "server_id": tool.server_id,
                "server_name": tool.server_name,
            }
            for tool in enabled_tools
        ]
        tool_names = [tool["name"] for tool in tool_summaries]
        logger.info(f"[{node_name}] Loaded enabled MCP tools | count={len(tool_names)}")
        status = "SUCCESS"
        error_message = None
    except Exception as exc:
        logger.warning(f"[{node_name}] Failed to load MCP tools: {exc}")
        tool_summaries = []
        tool_names = []
        status = "FAILED"
        error_message = str(exc)

    duration_ms = int((time.time() - start_time) * 1000)
    trace_update = await record_trace(
        state,
        trace_service,
        node_name,
        {"task_id": state.task_id, "dataset_id": state.dataset_id},
        {
            "enabled": bool(tool_names),
            "tool_count": len(tool_names),
            "tools": tool_names,
        },
        duration_ms,
        status,
        error_message,
    )

    return {
        "mcp_execution_enabled": bool(tool_names),
        "mcp_tool_names": tool_names,
        "mcp_tools": tool_summaries,
        **trace_update,
    }
