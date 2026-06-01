"""Helpers for deriving consistent per-round metrics from trace steps."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def normalize_round_summary_metrics(steps: Iterable[Any]) -> None:
    """Patch ROUND_SUMMARY output_data with attempt counts from sibling steps.

    Simulation failures without alpha IDs are stored as failure records rather
    than Alpha rows. Older workers therefore emitted ROUND_SUMMARY metrics with
    total_generated=0 even when CODE_GEN and SIMULATE clearly attempted alphas.
    This display-time normalization keeps API consumers consistent with trace
    evidence without rewriting historical trace records.
    """
    by_iteration: Dict[int, Dict[str, int]] = {}

    for step in steps:
        iteration = int(getattr(step, "iteration", 0) or 0)
        if not iteration:
            continue
        stats = by_iteration.setdefault(
            iteration,
            {"generated": 0, "simulated_attempts": 0, "simulation_errors": 0},
        )
        output = getattr(step, "output_data", None) or {}
        step_type = getattr(step, "step_type", "")

        if step_type == "CODE_GEN":
            generated = output.get("alphas_generated")
            if generated is None:
                expressions = output.get("expressions")
                generated = len(expressions) if isinstance(expressions, list) else 0
            stats["generated"] = max(stats["generated"], _as_int(generated))
        elif step_type == "SIMULATE":
            attempted = _as_int(output.get("simulated_count") or output.get("batch_size"))
            success = _as_int(output.get("success_count"))
            result_errors = sum(
                1
                for item in output.get("results") or []
                if isinstance(item, dict) and item.get("err")
            )
            stats["simulated_attempts"] = max(stats["simulated_attempts"], attempted)
            stats["simulation_errors"] = max(
                stats["simulation_errors"],
                result_errors or max(0, attempted - success),
            )

    for step in steps:
        if getattr(step, "step_type", "") != "ROUND_SUMMARY":
            continue
        stats = by_iteration.get(int(getattr(step, "iteration", 0) or 0), {})
        if not stats:
            continue

        output = dict(getattr(step, "output_data", None) or {})
        metrics = dict(output.get("round_metrics") or {})
        if stats.get("generated"):
            metrics["total_generated"] = max(
                _as_int(metrics.get("total_generated")),
                stats["generated"],
            )
        if stats.get("simulation_errors"):
            metrics["simulation_errors"] = max(
                _as_int(metrics.get("simulation_errors")),
                stats["simulation_errors"],
            )
        if stats.get("simulated_attempts"):
            metrics["total_attempted"] = max(
                _as_int(metrics.get("total_attempted")),
                stats["simulated_attempts"],
            )
        output["round_metrics"] = metrics
        step.output_data = output


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
