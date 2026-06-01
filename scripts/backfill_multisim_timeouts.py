#!/usr/bin/env python3
"""Backfill alpha rows from timed-out multi-simulation parents.

The mining workflow records parent simulation locations when BRAIN parent
multi-sims time out. Children can appear after the local timeout boundary, so
this project entrypoint re-reads those parent locations through BrainAdapter
and persists recovered child alpha details using the same Alpha schema.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select, desc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.adapters.brain_adapter import BrainAdapter
from backend.alpha_semantic_validator import compute_expression_hash, load_operators_from_db
from backend.agents.graph.nodes.evaluation import _strict_gate_failures
from backend.database import AsyncSessionLocal, init_db
from backend.models import Alpha, AlphaFailure, MiningTask
from backend.tasks.mining_tasks import _get_dataset_fields


def _default_thresholds() -> Dict[str, Any]:
    return {
        "sharpe_min": 1.58,
        "two_year_sharpe_min": 1.6,
        "fitness_min": 1.0,
        "rn_sharpe_min": 1.58,
        "rn_fitness_min": 1.0,
        "margin_min": 0.001,
        "turnover_min": 0.05,
        "turnover_max": 0.30,
        "prod_corr_max": 0.7,
        "self_corr_max": 0.5,
        "ra_fails_max": 0,
        "max_operator_count": 5,
    }


def _raw_details(failure: AlphaFailure) -> Dict[str, Any]:
    try:
        return json.loads(failure.raw_response or "{}")
    except Exception:
        return {}


def _failure_location(failure: AlphaFailure) -> Optional[str]:
    raw = _raw_details(failure)
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    return (
        metrics.get("_simulation_location")
        or raw.get("_simulation_location")
        or raw.get("simulation_location")
        or raw.get("location")
    )


def _candidate_metadata(failure: AlphaFailure) -> Dict[str, Any]:
    raw = _raw_details(failure)
    meta = raw.get("candidate_metadata") if isinstance(raw.get("candidate_metadata"), dict) else {}
    return dict(meta)


def _result_metrics(
    result: Dict[str, Any],
    expression: str,
    metadata: Dict[str, Any],
    thresholds: Dict[str, Any],
    fields: List[Dict[str, Any]],
    failure_id: int,
    location: str,
) -> Dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    checks = result.get("checks", []) or []
    brain_failed_checks = result.get("failed_checks", []) or []
    two_year = None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "LOW_2Y_SHARPE":
            two_year = check.get("value")
            break

    prod_corr = None
    self_corr = None
    metrics.update({
        "checks": checks,
        "can_submit": result.get("can_submit", False),
        "failed_checks": brain_failed_checks,
        "pending_checks": result.get("pending_checks", []) or [],
        "passed_checks": result.get("passed_checks", []) or [],
        "stage": result.get("stage"),
        "status": result.get("status"),
        "_settings": result.get("settings", {}) or {},
        "_raw_response": result.get("raw_response", result.get("raw")),
        "_simulation_location": result.get("location") or location,
        "_candidate_metadata": {
            **metadata,
            "_backfilled_from_failure_id": failure_id,
            "_backfilled_from_location": location,
        },
        "_prod_corr": prod_corr,
        "_self_corr": self_corr,
        "_corr_checked": False,
    })
    if two_year is not None:
        metrics["two_year_sharpe"] = two_year

    strict_failures = _strict_gate_failures(
        metrics,
        brain_failed_checks,
        prod_corr,
        self_corr,
        thresholds,
        expression=expression,
        fields=fields,
    )
    metrics["_strict_gate_failures"] = strict_failures
    metrics["_hard_pass"] = not strict_failures
    return metrics


def _quality_status(metrics: Dict[str, Any]) -> str:
    if metrics.get("_hard_pass"):
        return "PASS"
    sharpe = metrics.get("sharpe")
    fitness = metrics.get("fitness")
    try:
        sharpe_value = float(sharpe)
    except Exception:
        sharpe_value = 0.0
    try:
        fitness_value = float(fitness)
    except Exception:
        fitness_value = 0.0
    if abs(sharpe_value) >= 0.5 or fitness_value >= 0.2:
        return "OPTIMIZE"
    return "FAIL"


def _should_include_failure(args: argparse.Namespace, failure: AlphaFailure) -> bool:
    """Return whether a timed-out failure belongs in this backfill pass."""
    location = _failure_location(failure)
    if not location:
        return False
    meta = _candidate_metadata(failure)
    if getattr(args, "first_order_only", False):
        return meta.get("source") == "first_order_operator_probe"
    return True


async def _candidate_failures(args: argparse.Namespace) -> List[AlphaFailure]:
    async with AsyncSessionLocal() as db:
        query = (
            select(AlphaFailure)
            .where(AlphaFailure.error_type == "SIMULATION_ERROR")
            .order_by(desc(AlphaFailure.id))
        )
        if args.run_id:
            query = query.where(AlphaFailure.run_id.in_(args.run_id))
        if args.task_id:
            query = query.where(AlphaFailure.task_id.in_(args.task_id))
        rows = (await db.execute(query.limit(args.failure_limit))).scalars().all()

    candidates: List[AlphaFailure] = []
    for failure in rows:
        if not _should_include_failure(args, failure):
            continue
        candidates.append(failure)
    return candidates


async def _persist_backfilled_alpha(
    failure: AlphaFailure,
    result: Dict[str, Any],
    expression: str,
    metadata: Dict[str, Any],
    location: str,
    fields: List[Dict[str, Any]],
) -> Optional[str]:
    task_config: Dict[str, Any] = {}
    async with AsyncSessionLocal() as db:
        task = await db.get(MiningTask, failure.task_id) if failure.task_id else None
        if task and task.config:
            task_config = task.config
        thresholds = {**_default_thresholds(), **{k: task_config[k] for k in _default_thresholds() if k in task_config}}

        alpha_id = result.get("alpha_id")
        if not alpha_id:
            return None
        existing = (
            await db.execute(select(Alpha).where(Alpha.alpha_id == alpha_id).limit(1))
        ).scalar_one_or_none()

        metrics = _result_metrics(
            result=result,
            expression=expression,
            metadata=metadata,
            thresholds=thresholds,
            fields=fields,
            failure_id=failure.id,
            location=location,
        )
        settings = metrics.get("_settings") or {}
        values = {
            "task_id": failure.task_id,
            "run_id": failure.run_id,
            "alpha_id": alpha_id,
            "expression": expression,
            "expression_hash": compute_expression_hash(expression),
            "hypothesis": f"Backfilled timed-out multi-sim child for {metadata.get('probe_operator') or 'candidate'}",
            "logic_explanation": "Recovered from a parent multi-simulation timeout through the project backfill workflow.",
            "region": settings.get("region", task.region if task else "IND"),
            "universe": settings.get("universe", task.universe if task else "TOP500"),
            "dataset_id": metadata.get("dataset_id") or (task.target_datasets[0] if task and task.target_datasets else None),
            "quality_status": _quality_status(metrics),
            "delay": settings.get("delay", 1),
            "decay": settings.get("decay", 4),
            "neutralization": settings.get("neutralization", "CROWDING"),
            "truncation": settings.get("truncation", 0.08),
            "instrument_type": settings.get("instrumentType", settings.get("instrument_type", "EQUITY")),
            "settings": settings,
            "stage": metrics.get("stage") or "IS",
            "status": metrics.get("status") or "simulated",
            "is_sharpe": metrics.get("sharpe"),
            "is_turnover": metrics.get("turnover"),
            "is_fitness": metrics.get("fitness"),
            "is_returns": metrics.get("returns"),
            "is_drawdown": metrics.get("drawdown"),
            "is_margin": metrics.get("margin"),
            "is_long_count": metrics.get("longCount"),
            "is_short_count": metrics.get("shortCount"),
            "checks": metrics.get("checks"),
            "is_metrics": metrics,
            "metrics": metrics,
        }
        if existing:
            for key, value in values.items():
                if key in {"task_id", "run_id", "hypothesis", "logic_explanation"}:
                    continue
                setattr(existing, key, value)
        else:
            db.add(Alpha(**values))
        await db.commit()
        return alpha_id


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    await init_db()
    async with AsyncSessionLocal() as db:
        await load_operators_from_db(db)
    failures = await _candidate_failures(args)
    grouped: Dict[str, List[AlphaFailure]] = defaultdict(list)
    for failure in failures:
        grouped[_failure_location(failure)].append(failure)

    locations = list(grouped)
    if args.location:
        requested = set(args.location)
        locations = [location for location in locations if location in requested]
    locations = locations[: args.location_limit]

    print({"candidate_failures": len(failures), "candidate_locations": len(grouped), "selected_locations": len(locations)})
    if args.dry_run:
        for location in locations:
            print({"location": location, "failures": [(f.id, f.expression, _candidate_metadata(f).get("probe_operator")) for f in grouped[location]]})
        return {
            "candidate_failures": len(failures),
            "candidate_locations": len(grouped),
            "selected_locations": len(locations),
            "recovered": 0,
            "unrecovered": 0,
        }

    recovered = []
    unrecovered = []
    fields_cache: Dict[tuple, List[Dict[str, Any]]] = {}
    async with BrainAdapter() as brain:
        for location in locations:
            failures_for_location = sorted(grouped[location], key=lambda item: item.id)
            result = await brain._collect_multisim_results_if_available(
                poll_url=location,
                location=location,
                grace_seconds=args.grace_seconds,
            )
            results = result.get("results") if isinstance(result.get("results"), list) else []
            if not result.get("success") or not results:
                unrecovered.append({"location": location, "error": result.get("error"), "failures": [f.id for f in failures_for_location]})
                continue

            for failure, child_result in zip(failures_for_location, results):
                if not child_result.get("success") or not child_result.get("alpha_id"):
                    unrecovered.append({
                        "location": location,
                        "failure_id": failure.id,
                        "expression": failure.expression,
                        "error": child_result.get("error"),
                    })
                    continue
                meta = _candidate_metadata(failure)
                async with AsyncSessionLocal() as db:
                    task = await db.get(MiningTask, failure.task_id) if failure.task_id else None
                    region = task.region if task else "IND"
                    universe = task.universe if task else "TOP500"
                    dataset = (task.target_datasets[0] if task and task.target_datasets else "ml_factor_proj")
                cache_key = (dataset, region, universe)
                if cache_key not in fields_cache:
                    async with AsyncSessionLocal() as db:
                        fields_cache[cache_key] = await _get_dataset_fields(db, dataset, region, universe)
                alpha_id = await _persist_backfilled_alpha(
                    failure=failure,
                    result=child_result,
                    expression=failure.expression,
                    metadata={**meta, "dataset_id": dataset},
                    location=location,
                    fields=fields_cache[cache_key],
                )
                recovered.append({
                    "failure_id": failure.id,
                    "alpha_id": alpha_id,
                    "operator": meta.get("probe_operator"),
                    "expression": failure.expression,
                    "sharpe": (child_result.get("metrics") or {}).get("sharpe"),
                    "fitness": (child_result.get("metrics") or {}).get("fitness"),
                    "turnover": (child_result.get("metrics") or {}).get("turnover"),
                    "margin": (child_result.get("metrics") or {}).get("margin"),
                })

    print({"recovered": len(recovered), "unrecovered": len(unrecovered)})
    for item in recovered:
        print("RECOVERED", item)
    for item in unrecovered[:50]:
        print("UNRECOVERED", item)
    return {
        "candidate_failures": len(failures),
        "candidate_locations": len(grouped),
        "selected_locations": len(locations),
        "recovered": len(recovered),
        "unrecovered": len(unrecovered),
        "recovered_items": recovered,
        "unrecovered_items": unrecovered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill alphas from timed-out project multi-sim parents.")
    parser.add_argument("--run-id", nargs="*", type=int, default=None)
    parser.add_argument("--task-id", nargs="*", type=int, default=None)
    parser.add_argument("--location", nargs="*", default=None)
    parser.add_argument("--failure-limit", type=int, default=200)
    parser.add_argument("--location-limit", type=int, default=20)
    parser.add_argument("--grace-seconds", type=int, default=30)
    parser.add_argument("--include-non-first-order", action="store_true", help="Deprecated: all project candidate sources are included by default.")
    parser.add_argument("--first-order-only", action="store_true", help="Only backfill first-order operator probe failures.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
