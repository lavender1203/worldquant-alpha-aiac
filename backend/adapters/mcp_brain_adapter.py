"""
MCP-aware BRAIN adapter.

Mining code continues to depend on the BrainAdapter protocol, while this
adapter dynamically routes supported operations through enabled MCP tools.
If a tool is disabled from the web page or fails, read-only operations fall
back to the direct BRAIN adapter; simulation falls back to direct BRAIN only
when MCP create_multi_simulation is unavailable or fails.
"""

from typing import Any, Dict, List, Optional
import asyncio

from loguru import logger

from backend.adapters.brain_adapter import BrainAdapter
from backend.mcp_tool_service import MCPToolClient


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("results"), list):
            return value["results"]
        return [value]
    return []


def _extract_result_items(payload: Any, expected_count: int = 0) -> List[Dict[str, Any]]:
    """Extract likely per-alpha result objects from a MCP simulation response."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in (
        "results",
        "individual_results",
        "alpha_results",
        "alphas",
        "simulations",
        "children",
        "details",
    ):
        value = payload.get(key)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return value
        if isinstance(value, dict):
            nested = _extract_result_items(value, expected_count)
            if nested:
                return nested

    if payload.get("alpha_id") or payload.get("alpha") or payload.get("id") or payload.get("is") or payload.get("metrics"):
        return [payload]

    for value in payload.values():
        nested = _extract_result_items(value, expected_count)
        if nested and (not expected_count or len(nested) >= min(expected_count, 1)):
            return nested

    return []


def _normalize_alpha_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MCP alpha detail shapes to the local BrainAdapter shape."""
    if item.get("success") is False:
        return {
            "success": False,
            "error": item.get("error") or item.get("message") or f"MCP simulation failed: {item!r}",
            "source": "mcp",
            "raw": item,
        }

    alpha = item.get("alpha") if isinstance(item.get("alpha"), dict) else item
    alpha_id = item.get("alpha_id") or item.get("alphaId")
    if not alpha_id and isinstance(item.get("alpha"), str):
        alpha_id = item.get("alpha")
    if not alpha_id:
        alpha_id = alpha.get("alpha_id") or alpha.get("id") or alpha.get("alphaId")

    is_stats = alpha.get("is") or item.get("is") or {}
    train_stats = alpha.get("train") or item.get("train") or {}
    test_stats = alpha.get("test") or item.get("test") or {}
    os_stats = alpha.get("os") or item.get("os") or {}
    metrics = dict(alpha.get("metrics") or item.get("metrics") or {})

    if is_stats:
        metrics.update(
            {
                "sharpe": metrics.get("sharpe", is_stats.get("sharpe")),
                "returns": metrics.get("returns", is_stats.get("returns")),
                "turnover": metrics.get("turnover", is_stats.get("turnover")),
                "fitness": metrics.get("fitness", is_stats.get("fitness")),
                "drawdown": metrics.get("drawdown", is_stats.get("drawdown")),
                "pnl": metrics.get("pnl", is_stats.get("pnl")),
                "margin": metrics.get("margin", is_stats.get("margin")),
                "bookSize": metrics.get("bookSize", is_stats.get("bookSize")),
                "longCount": metrics.get("longCount", is_stats.get("longCount")),
                "shortCount": metrics.get("shortCount", is_stats.get("shortCount")),
                "investabilityConstrained": metrics.get(
                    "investabilityConstrained", is_stats.get("investabilityConstrained") or {}
                ),
                "riskNeutralized": metrics.get("riskNeutralized", is_stats.get("riskNeutralized") or {}),
            }
        )

    if train_stats:
        metrics.update(
            {
                "train_sharpe": metrics.get("train_sharpe", train_stats.get("sharpe")),
                "train_fitness": metrics.get("train_fitness", train_stats.get("fitness")),
                "train_turnover": metrics.get("train_turnover", train_stats.get("turnover")),
                "train_returns": metrics.get("train_returns", train_stats.get("returns")),
                "train_drawdown": metrics.get("train_drawdown", train_stats.get("drawdown")),
                "train_investabilityConstrained": metrics.get(
                    "train_investabilityConstrained", train_stats.get("investabilityConstrained") or {}
                ),
                "train_riskNeutralized": metrics.get(
                    "train_riskNeutralized", train_stats.get("riskNeutralized") or {}
                ),
            }
        )

    if test_stats:
        metrics.update(
            {
                "test_sharpe": metrics.get("test_sharpe", test_stats.get("sharpe")),
                "test_fitness": metrics.get("test_fitness", test_stats.get("fitness")),
                "test_turnover": metrics.get("test_turnover", test_stats.get("turnover")),
                "test_returns": metrics.get("test_returns", test_stats.get("returns")),
                "test_drawdown": metrics.get("test_drawdown", test_stats.get("drawdown")),
                "test_investabilityConstrained": metrics.get(
                    "test_investabilityConstrained", test_stats.get("investabilityConstrained") or {}
                ),
                "test_riskNeutralized": metrics.get("test_riskNeutralized", test_stats.get("riskNeutralized") or {}),
            }
        )

    if os_stats:
        metrics.update(
            {
                "os_sharpe": metrics.get("os_sharpe", os_stats.get("sharpe")),
                "os_fitness": metrics.get("os_fitness", os_stats.get("fitness")),
            }
        )

    rn_sharpe = metrics.get("risk_neutralized_sharpe", metrics.get("rn_sharpe"))
    rn_fitness = metrics.get("risk_neutralized_fitness", metrics.get("rn_fitness"))
    if rn_sharpe is None:
        rn_sharpe = item.get("risk_neutralized_sharpe", item.get("rn_sharpe"))
    if rn_fitness is None:
        rn_fitness = item.get("risk_neutralized_fitness", item.get("rn_fitness"))
    if rn_sharpe is not None or rn_fitness is not None:
        rn_metrics = dict(metrics.get("riskNeutralized") or {})
        if rn_sharpe is not None:
            rn_metrics["sharpe"] = rn_sharpe
        if rn_fitness is not None:
            rn_metrics["fitness"] = rn_fitness
        metrics["riskNeutralized"] = rn_metrics

    checks = item.get("checks") or alpha.get("checks") or is_stats.get("checks") or metrics.get("checks") or []
    failed_checks = item.get("failed_checks") or [c.get("name") for c in checks if isinstance(c, dict) and c.get("result") == "FAIL"]
    pending_checks = item.get("pending_checks") or [
        c.get("name") for c in checks if isinstance(c, dict) and c.get("result") == "PENDING"
    ]
    passed_checks = item.get("passed_checks") or [c.get("name") for c in checks if isinstance(c, dict) and c.get("result") == "PASS"]

    regular = alpha.get("regular") or item.get("regular") or {}
    expression = item.get("expression") or regular.get("code")

    return {
        "success": bool(alpha_id or metrics) and not item.get("error"),
        "alpha_id": alpha_id,
        "expression": expression,
        "settings": alpha.get("settings") or item.get("settings") or {},
        "stage": alpha.get("stage") or item.get("stage"),
        "status": alpha.get("status") or item.get("status"),
        "type": alpha.get("type") or item.get("type"),
        "dateCreated": alpha.get("dateCreated") or item.get("dateCreated"),
        "dateSubmitted": alpha.get("dateSubmitted") or item.get("dateSubmitted"),
        "metrics": metrics,
        "checks": checks,
        "failed_checks": failed_checks,
        "pending_checks": pending_checks,
        "passed_checks": passed_checks,
        "can_submit": bool(item.get("can_submit", len(failed_checks) == 0 and len(pending_checks) == 0)),
        "is": is_stats,
        "os": os_stats,
        "train": train_stats,
        "test": test_stats,
        "regular": regular,
        "raw": alpha,
        "source": "mcp",
    }


def _max_correlation(payload: Any) -> Optional[float]:
    """Normalize diverse MCP correlation responses to {'max': value}."""
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, str):
        try:
            return float(payload)
        except ValueError:
            return None
    if isinstance(payload, list):
        values = [_max_correlation(item) for item in payload]
        numeric = [value for value in values if value is not None]
        return max(numeric, default=None)
    if not isinstance(payload, dict):
        return None

    for key in ("max", "max_correlation", "maxCorrelation", "max_self_correlation", "prod_corr", "production_correlation"):
        value = payload.get(key)
        if value is not None:
            try:
                return abs(float(value))
            except (TypeError, ValueError):
                pass

    for key in ("correlations", "records", "results", "top_correlations"):
        values = payload.get(key)
        if isinstance(values, list):
            nested = [_max_correlation(item) for item in values]
            numeric = [value for value in nested if value is not None]
            return max(numeric, default=None)

    for key in ("correlation", "value"):
        value = payload.get(key)
        if value is not None:
            try:
                return abs(float(value))
            except (TypeError, ValueError):
                pass

    return None


class MCPBrainAdapter:
    """BrainAdapter-compatible runtime that prefers enabled MCP tools."""

    def __init__(self, fallback: Optional[BrainAdapter] = None, mcp_client: Optional[MCPToolClient] = None):
        self.fallback = fallback or BrainAdapter()
        self.mcp = mcp_client or MCPToolClient()
        self._fallback_entered = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self._fallback_entered:
            await self.fallback.__aexit__(*args)
            self._fallback_entered = False

    async def _ensure_fallback(self) -> BrainAdapter:
        if not self._fallback_entered:
            await self.fallback.__aenter__()
            self._fallback_entered = True
        return self.fallback

    async def enabled_tool_names(self) -> List[str]:
        tools = await self.mcp.list_enabled_tools()
        return [tool.name for tool in tools]

    async def simulate_alpha(self, expression: str, **kwargs) -> Dict[str, Any]:
        results = await self.simulate_batch([expression], **kwargs)
        return results[0] if results else {"success": False, "error": "No simulation result returned"}

    async def simulate_batch(
        self,
        expressions: List[str],
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 4,
        neutralization: str = "SUBINDUSTRY",
        truncation: float = 0.08,
        test_period: str = "P2Y0M",
        max_wait: int = 1200,
        timeout_grace_seconds: int = 180,
        no_child_timeout_seconds: int = 0,
    ) -> List[Dict[str, Any]]:
        if await self.mcp.is_tool_enabled("create_multi_simulation"):
            try:
                timeout_seconds = max(1, int(max_wait))
                payload = await asyncio.wait_for(
                    self.mcp.call_tool(
                        "create_multi_simulation",
                        {
                            "alpha_expressions": expressions,
                            "instrument_type": "EQUITY",
                            "region": region,
                            "universe": universe,
                            "delay": delay,
                            "decay": decay,
                            "neutralization": neutralization,
                            "truncation": truncation,
                            "test_period": test_period,
                            "unit_handling": "VERIFY",
                            "nan_handling": "OFF",
                            "language": "FASTEXPR",
                            "visualization": False,
                            "pasteurization": "ON",
                            "max_trade": "ON",
                        },
                    ),
                    timeout=timeout_seconds,
                )
                items = _extract_result_items(payload, expected_count=len(expressions))
                results = [_normalize_alpha_result(item) for item in items]
                if len(results) < len(expressions):
                    results.extend(
                        {
                            "success": False,
                            "error": "MCP create_multi_simulation returned fewer results than requested",
                            "source": "mcp",
                        }
                        for _ in range(len(expressions) - len(results))
                    )
                logger.info(f"[MCPBrainAdapter] simulate_batch via MCP | count={len(expressions)}")
                return results[: len(expressions)]
            except asyncio.TimeoutError:
                logger.warning(
                    "[MCPBrainAdapter] MCP create_multi_simulation timed out after "
                    f"{timeout_seconds}s"
                )
                return [
                    {
                        "success": False,
                        "error": f"MCP create_multi_simulation timed out after {timeout_seconds}s",
                        "source": "mcp",
                    }
                    for _ in expressions
                ]
            except Exception as exc:
                logger.warning(
                    "[MCPBrainAdapter] MCP create_multi_simulation failed, using fallback: "
                    f"{type(exc).__name__}: {exc!r}"
                )

        fallback = await self._ensure_fallback()
        return await fallback.simulate_batch(
            expressions=expressions,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            test_period=test_period,
            max_wait=max_wait,
            timeout_grace_seconds=timeout_grace_seconds,
            no_child_timeout_seconds=no_child_timeout_seconds,
        )

    async def check_correlation(self, alpha_id: str, check_type: str = "PROD") -> Dict[str, Any]:
        tool_name = "check_self_correlation" if check_type.upper() == "SELF" else "check_correlation"
        max_attempts = 3 if check_type.upper() == "PROD" else 1
        if await self.mcp.is_tool_enabled(tool_name):
            for attempt in range(max_attempts):
                try:
                    args: Dict[str, Any] = {"alpha_id": alpha_id}
                    if tool_name == "check_self_correlation":
                        args["threshold"] = 0.5
                    payload = await self.mcp.call_tool(tool_name, args)
                    max_corr = _max_correlation(payload)
                    logger.info(
                        "[MCPBrainAdapter] check_correlation via MCP | "
                        f"alpha={alpha_id} type={check_type} attempt={attempt + 1} max={max_corr}"
                    )
                    if max_corr is not None:
                        return {"max": max_corr, "raw": payload, "source": "mcp"}
                    if attempt == max_attempts - 1:
                        break
                    await asyncio.sleep(10 * (attempt + 1))
                except Exception as exc:
                    logger.warning(f"[MCPBrainAdapter] MCP {tool_name} failed, using fallback: {exc}")
                    break

        fallback = await self._ensure_fallback()
        payload = await fallback.check_correlation(alpha_id, check_type=check_type)
        return {"max": _max_correlation(payload), "raw": payload, "source": "brain_api"}

    async def get_datasets(self, region: str = "USA", delay: int = 1, universe: str = "TOP3000") -> List[Dict[str, Any]]:
        if await self.mcp.is_tool_enabled("get_datasets"):
            try:
                payload = await self.mcp.call_tool(
                    "get_datasets",
                    {"region": region, "delay": delay, "universe": universe},
                )
                return _as_list(payload)
            except Exception as exc:
                logger.warning(f"[MCPBrainAdapter] MCP get_datasets failed, using fallback: {exc}")
        fallback = await self._ensure_fallback()
        return await fallback.get_datasets(region=region, delay=delay, universe=universe)

    async def get_datafields(
        self,
        dataset_id: str,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
    ) -> List[Dict[str, Any]]:
        if await self.mcp.is_tool_enabled("get_datafields"):
            try:
                payload = await self.mcp.call_tool(
                    "get_datafields",
                    {
                        "dataset_id": dataset_id,
                        "region": region,
                        "delay": delay,
                        "universe": universe,
                    },
                )
                return _as_list(payload)
            except Exception as exc:
                logger.warning(f"[MCPBrainAdapter] MCP get_datafields failed, using fallback: {exc}")
        fallback = await self._ensure_fallback()
        return await fallback.get_datafields(dataset_id=dataset_id, region=region, delay=delay, universe=universe)

    async def get_operators(self, detailed: bool = False) -> List[Any]:
        if await self.mcp.is_tool_enabled("get_operators"):
            try:
                payload = await self.mcp.call_tool("get_operators", {})
                operators = _as_list(payload)
                if detailed:
                    return operators
                return [op.get("name") if isinstance(op, dict) else op for op in operators]
            except Exception as exc:
                logger.warning(f"[MCPBrainAdapter] MCP get_operators failed, using fallback: {exc}")
        fallback = await self._ensure_fallback()
        return await fallback.get_operators(detailed=detailed)

    async def get_alpha_pnl(self, alpha_id: str) -> Dict[str, Any]:
        if await self.mcp.is_tool_enabled("get_alpha_pnl"):
            try:
                return await self.mcp.call_tool("get_alpha_pnl", {"alpha_id": alpha_id})
            except Exception as exc:
                logger.warning(f"[MCPBrainAdapter] MCP get_alpha_pnl failed, using fallback: {exc}")
        fallback = await self._ensure_fallback()
        return await fallback.get_alpha_pnl(alpha_id)

    async def get_user_alphas(
        self,
        limit: int = 100,
        offset: int = 0,
        stage: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        if await self.mcp.is_tool_enabled("get_user_alphas"):
            try:
                args: Dict[str, Any] = {"limit": limit, "offset": offset}
                if stage:
                    args["stage"] = stage
                if search:
                    args["search"] = search
                if start_date:
                    args["start_date"] = start_date
                payload = await self.mcp.call_tool("get_user_alphas", args)
                return payload if isinstance(payload, dict) else {"results": _as_list(payload)}
            except Exception as exc:
                logger.warning(f"[MCPBrainAdapter] MCP get_user_alphas failed, using fallback: {exc}")
        fallback = await self._ensure_fallback()
        return await fallback.get_user_alphas(
            limit=limit,
            offset=offset,
            stage=stage,
            search=search,
            start_date=start_date,
        )
