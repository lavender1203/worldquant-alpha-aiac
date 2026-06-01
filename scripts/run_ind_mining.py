#!/usr/bin/env python3
"""Run an IND/TOP500 mining pass through the project pipeline.

This script is intentionally thin: it creates a normal MiningTask/ExperimentRun
and calls MiningAgent.run_evolution_loop so generation, validation, simulation,
evaluation, observability, and optimization all use the repository code.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import select, desc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.adapters.mcp_brain_adapter import MCPBrainAdapter
from backend.agents import MiningAgent
from backend.database import AsyncSessionLocal, init_db
from backend.models import Alpha, DatasetMetadata, ExperimentRun, KnowledgeEntry, MiningTask
from backend.tasks.mining_tasks import _get_dataset_fields, _get_datasets_to_mine, _get_operators
from scripts.backfill_multisim_timeouts import run as run_timeout_backfill


DEFAULT_DATASETS = [
    "analyst_revision_horizons",
    "model39",
    "model38",
    "model32",
    "risk68",
    "model30",
    "analyst_consensus",
    "model28",
    "risk70",
    "sentiment23",
    "news36",
    "other384",
    "intraday_pv_feats",
]


DEFAULT_MODEL39_SWEEP_EXPRESSIONS = [
    {
        "expression": "group_rank(add(ts_delta(sector_value_momentum_rank_float, 21), ts_zscore(industry_value_momentum_rank_float, 44)), industry)",
        "description": "Low production-correlation sector/industry value-momentum blend.",
    },
    {
        "expression": "group_rank(add(ts_delta(sector_value_momentum_rank_float, 21), ts_zscore(industry_value_momentum_rank_float, 44)), subindustry)",
        "description": "Subindustry-ranked variant of the low production-correlation value-momentum blend.",
    },
    {
        "expression": "group_rank(add(sector_value_momentum_rank_float, ts_delta(short_term_price_momentum_score_2, 10)), industry)",
        "description": "Sector value momentum blended with short-term price momentum change.",
    },
    {
        "expression": "group_rank(add(sector_value_momentum_rank_float, ts_delta(short_term_price_momentum_score_2, 10)), subindustry)",
        "description": "Subindustry-ranked sector value and short-term price momentum blend.",
    },
    {
        "expression": "signed_power(group_rank(ts_rank(industry_value_momentum_rank_float, 500), industry), 3.0)",
        "description": "High-Sharpe industry value momentum rank template for settings sensitivity.",
    },
    {
        "expression": "signed_power(group_rank(ts_rank(global_value_momentum_rank_float, 500), industry), 3.0)",
        "description": "High-Sharpe global value momentum rank template for settings sensitivity.",
    },
]


def default_config() -> Dict[str, Any]:
    return {
        "delay": 1,
        "decay": 4,
        "neutralization": "CROWDING",
        "truncation": 0.08,
        "test_period": "P2Y0M",
        "max_trade": "ON",
        "type": "REGULAR",
        "max_fields": 2,
        "max_operator_count": 5,
        "no_trade_when": True,
        "avoid_operators": [],
        "sharpe_min": 1.58,
        "two_year_sharpe_min": 1.6,
        "fitness_min": 1.0,
        "rn_sharpe_min": 1.58,
        "rn_fitness_min": 1.0,
        "margin_min": 0.001,
        "turnover_min": 0.05,
        "turnover_max": 0.30,
        "self_corr_max": 0.5,
        "prod_corr_max": 0.7,
        "ra_fails_max": 0,
        "optimization_min_sharpe": 0.6,
        "optimization_min_fitness": 0.25,
        "optimization_reversal_abs_sharpe_min": 0.8,
        "optimization_max_candidates": 3,
        "optimization_max_rewrites": 10,
        "optimization_settings_limit": 4,
        "optimization_max_batches_per_candidate": 1,
        "optimization_simulation_max_wait": 300,
        "generation_candidate_pool": 40,
        "deterministic_templates_first": True,
        "template_candidate_quota": 40,
        "first_order_operator_probe": True,
        "first_order_operator_probe_skip_covered": True,
        "first_order_operator_probe_batch_size": 24,
        "first_order_probe_deferred_retry_slots": 1,
        "first_order_auto_strengthen_after_active_coverage": True,
        "first_order_auto_strengthen_max_seeds": 6,
        "first_order_auto_strengthen_variants_per_seed": 4,
        "avoid_attempted_expressions": True,
        "attempted_expression_limit": 500,
        "operator_coverage_interval": 10,
        "simulation_batch_size": 4,
        "simulation_max_wait": 900,
        "simulation_timeout_grace_seconds": 180,
        "simulation_no_child_timeout_seconds": 150,
        "settings_sweep_batch_size": 4,
        "settings_sweep_max_batches": 6,
        "genetic_optimization_enabled": False,
        "dataset_error_round_limit": 2,
        "dataset_weak_signal_round_limit": 3,
        "dataset_weak_signal_sharpe_floor": 0.50,
        "dataset_weak_signal_fitness_floor": 0.20,
        "dataset_weak_rn_stop_ratio": 0.60,
        "focus_hypotheses": [
            "Build comparable-scale ratios or spreads before ranking.",
            "Residualize the primary field against risk, returns, volatility, liquidity, or crowding.",
            "Prefer compact industry/subindustry grouped ranks and neutralized deviations.",
            "Use state frequency, event counts, second-order changes, and extreme deviations for diversity.",
        ],
        "preferred_operators": [
            "rank",
            "group_rank",
            "group_neutralize",
            "ts_zscore",
            "ts_delta",
            "ts_mean",
            "subtract",
            "divide",
            "winsorize",
        ],
    }


def build_settings_sweep_variants(args: argparse.Namespace, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = []
    for neutralization in args.settings_sweep_neutralizations:
        for decay in args.settings_sweep_decays:
            for truncation in args.settings_sweep_truncations:
                variants.append({
                    "delay": config["delay"],
                    "decay": decay,
                    "neutralization": neutralization,
                    "truncation": truncation,
                    "test_period": config["test_period"],
                    "description": f"Settings sweep d{config['delay']} decay{decay} {neutralization} trunc{truncation:g}",
                })
    return variants


def known_sweep_expressions(dataset_id: str) -> List[Dict[str, Any]]:
    if dataset_id == "model39":
        return DEFAULT_MODEL39_SWEEP_EXPRESSIONS
    return []


async def seed_forum_knowledge() -> None:
    async with AsyncSessionLocal() as db:
        existing = await db.execute(
            select(KnowledgeEntry).where(
                KnowledgeEntry.entry_type == "SUCCESS_PATTERN",
                KnowledgeEntry.pattern == "forum_durable_field_template_construction",
            )
        )
        if existing.scalar_one_or_none():
            return
        db.add(
            KnowledgeEntry(
                entry_type="SUCCESS_PATTERN",
                pattern="forum_durable_field_template_construction",
                description=(
                    "Construct durable alphas by comparing fields on a common scale, "
                    "forming spreads or ratios, residualizing against controls, "
                    "measuring center or extreme deviation, using second-order changes, "
                    "and counting persistent states instead of copying shallow formulas."
                ),
                meta_data={
                    "source": "forum_legend_durable_template",
                    "regions": ["IND"],
                    "dataset_categories": ["analyst", "model", "risk", "sentiment", "news", "other", "pv"],
                    "dataset_category": "other",
                },
                created_by="SYSTEM",
            )
        )
        await db.commit()


async def choose_auto_datasets(limit: int, include_lit_pyramids: bool = False) -> List[str]:
    async with AsyncSessionLocal() as db:
        query = (
            select(DatasetMetadata)
            .where(
                DatasetMetadata.region == "IND",
                DatasetMetadata.universe == "TOP500",
                DatasetMetadata.delay == 1,
                DatasetMetadata.is_active == True,
                DatasetMetadata.field_count > 0,
            )
        )
        if not include_lit_pyramids:
            query = query.where((DatasetMetadata.alpha_count == None) | (DatasetMetadata.alpha_count == 0))
        query = query.order_by(
            desc(DatasetMetadata.mining_weight),
            desc(DatasetMetadata.pyramid_multiplier),
            desc(DatasetMetadata.value_score),
            desc(DatasetMetadata.coverage),
            DatasetMetadata.alpha_count.asc(),
        ).limit(limit)
        rows = (await db.execute(query)).scalars().all()
        return [row.dataset_id for row in rows]


async def summarize_run(run_id: int) -> None:
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(Alpha)
                .where(Alpha.run_id == run_id)
                .order_by(desc(Alpha.is_sharpe), desc(Alpha.is_fitness))
                .limit(20)
            )
        ).scalars().all()
        print(f"\nRun {run_id} top alphas:")
        for alpha in rows:
            metrics = alpha.metrics or {}
            rn = metrics.get("riskNeutralized") or {}
            print(
                "  "
                f"alpha={alpha.alpha_id} ds={alpha.dataset_id} q={alpha.quality_status} "
                f"stage={alpha.stage} status={alpha.status} "
                f"sharpe={alpha.is_sharpe} fitness={alpha.is_fitness} "
                f"turnover={alpha.is_turnover} margin={alpha.is_margin} "
                f"rn=({rn.get('sharpe')},{rn.get('fitness')}) "
                f"pc={metrics.get('_prod_corr')} sc={metrics.get('_self_corr')} "
                f"hard={metrics.get('_hard_pass')} expr={alpha.expression[:120]}"
            )


async def run(args: argparse.Namespace) -> None:
    await init_db()
    if args.seed_forum_knowledge:
        await seed_forum_knowledge()

    datasets = args.datasets
    if args.auto_datasets:
        datasets = await choose_auto_datasets(
            args.auto_datasets,
            include_lit_pyramids=args.include_lit_pyramids,
        )
    if not datasets:
        datasets = await choose_auto_datasets(
            5,
            include_lit_pyramids=args.include_lit_pyramids,
        )
    if not datasets:
        datasets = DEFAULT_DATASETS

    config = default_config()
    if args.neutralization:
        config["neutralization"] = args.neutralization
    if args.decay is not None:
        config["decay"] = args.decay
    if args.truncation is not None:
        config["truncation"] = args.truncation
    if args.optimization_batches is not None:
        config["optimization_max_batches_per_candidate"] = args.optimization_batches
    if args.optimization_simulation_max_wait is not None:
        config["optimization_simulation_max_wait"] = args.optimization_simulation_max_wait
    if args.generation_pool is not None:
        config["generation_candidate_pool"] = args.generation_pool
    if args.template_quota is not None:
        config["template_candidate_quota"] = args.template_quota
    if args.simulation_max_wait is not None:
        config["simulation_max_wait"] = args.simulation_max_wait
    if args.simulation_timeout_grace_seconds is not None:
        config["simulation_timeout_grace_seconds"] = args.simulation_timeout_grace_seconds
    if args.simulation_no_child_timeout_seconds is not None:
        config["simulation_no_child_timeout_seconds"] = args.simulation_no_child_timeout_seconds
    if args.simulation_batch_size is not None:
        config["simulation_batch_size"] = args.simulation_batch_size
    if args.simulation_cancel_retry_passes is not None:
        config["simulation_cancel_retry_passes"] = args.simulation_cancel_retry_passes
    if args.first_order_probe_start_index is not None:
        config["first_order_operator_probe_start_index"] = args.first_order_probe_start_index
    if args.first_order_deferred_retry_slots is not None:
        config["first_order_probe_deferred_retry_slots"] = args.first_order_deferred_retry_slots
    if args.first_order_skip_covered is not None:
        config["first_order_operator_probe_skip_covered"] = args.first_order_skip_covered
    if args.auto_strengthen_after_active_coverage is not None:
        config["first_order_auto_strengthen_after_active_coverage"] = args.auto_strengthen_after_active_coverage
    if args.first_order_auto_strengthen_max_seeds is not None:
        config["first_order_auto_strengthen_max_seeds"] = args.first_order_auto_strengthen_max_seeds
    if args.first_order_auto_strengthen_variants_per_seed is not None:
        config["first_order_auto_strengthen_variants_per_seed"] = args.first_order_auto_strengthen_variants_per_seed
    if args.avoid_attempted_expressions is not None:
        config["avoid_attempted_expressions"] = args.avoid_attempted_expressions
    if args.attempted_expression_limit is not None:
        config["attempted_expression_limit"] = args.attempted_expression_limit
    if args.preferred_fields:
        config["preferred_fields"] = args.preferred_fields
    if args.settings_sweep_max_batches is not None:
        config["settings_sweep_max_batches"] = args.settings_sweep_max_batches
    if args.no_first_order_probe:
        config["first_order_operator_probe"] = False
    if args.strengthen_seed_json:
        seeds = []
        for raw_seed in args.strengthen_seed_json:
            try:
                seed = json.loads(raw_seed)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid --strengthen-seed-json value: {exc}: {raw_seed}") from exc
            if not isinstance(seed, dict) or not seed.get("expression"):
                raise SystemExit("--strengthen-seed-json must be a JSON object with an expression field")
            seeds.append(seed)
        config["first_order_strengthening_seeds"] = seeds
        if args.no_first_order_probe:
            targeted_pool = max(1, int(args.alphas_per_round or 1))
            if args.generation_pool is None:
                config["generation_candidate_pool"] = targeted_pool
            if args.template_quota is None:
                config["template_candidate_quota"] = targeted_pool

    async with AsyncSessionLocal() as db:
        task = MiningTask(
            task_name=f"IND regular mining {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            region="IND",
            universe="TOP500",
            dataset_strategy="SPECIFIC" if datasets else "AUTO",
            target_datasets=datasets,
            agent_mode="AUTONOMOUS",
            status="RUNNING",
            daily_goal=args.goal,
            progress_current=0,
            max_iterations=args.max_iterations,
            config=config,
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)

        run_row = ExperimentRun(
            task_id=task.id,
            status="RUNNING",
            trigger_source="SCRIPT",
            config_snapshot={"task": {"region": task.region, "universe": task.universe, "config": task.config}},
            strategy_snapshot={},
        )
        db.add(run_row)
        await db.commit()
        await db.refresh(run_row)

        selected_datasets = datasets or await _get_datasets_to_mine(db, task)
        operators = await _get_operators(db)
        try:
            from backend.alpha_semantic_validator import load_operators_from_db

            loaded = await load_operators_from_db(db)
            print(f"Loaded operator registry: {len(loaded)} operators")
        except Exception as exc:
            print(f"Operator registry load failed: {type(exc).__name__}: {exc}")
        agent = MiningAgent(db, MCPBrainAdapter())

        total_success = 0
        for dataset_id in selected_datasets:
            fields = await _get_dataset_fields(db, dataset_id, task.region, task.universe)
            if not fields:
                print(f"Skipping {dataset_id}: no fields in DB")
                continue

            if args.settings_sweep_known or args.settings_sweep_seed_json:
                sweep_expressions = known_sweep_expressions(dataset_id)
                if args.settings_sweep_seed_json:
                    for raw_seed in args.settings_sweep_seed_json:
                        try:
                            seed = json.loads(raw_seed)
                        except json.JSONDecodeError as exc:
                            raise SystemExit(f"Invalid --settings-sweep-seed-json value: {exc}: {raw_seed}") from exc
                        if not isinstance(seed, dict) or not seed.get("expression"):
                            raise SystemExit("--settings-sweep-seed-json must be a JSON object with an expression field")
                        sweep_expressions.append(seed)
                if sweep_expressions:
                    sweep_variants = build_settings_sweep_variants(args, config)
                    print(
                        f"\nSettings sweep dataset={dataset_id} "
                        f"expressions={len(sweep_expressions)} settings={len(sweep_variants)}"
                    )
                    sweep_result = await agent.run_expression_settings_sweep(
                        task=task,
                        dataset_id=dataset_id,
                        seed_expressions=sweep_expressions,
                        settings_variants=sweep_variants,
                        fields=fields,
                        run_id=run_row.id,
                    )
                    total_success += int(sweep_result.get("strict_pass", 0) or 0)
                    task.progress_current = total_success
                    await db.commit()
                    print(f"Settings sweep done: {sweep_result}")
                    if total_success >= args.goal or args.settings_sweep_only:
                        break

            print(
                f"\nMining dataset={dataset_id} fields={len(fields)} "
                f"operators={len(operators)} remaining_goal={args.goal - total_success}"
            )
            result = await agent.run_evolution_loop(
                task=task,
                dataset_id=dataset_id,
                fields=fields,
                operators=operators,
                max_iterations=args.max_iterations,
                target_alphas=max(1, args.goal - total_success),
                num_alphas_per_round=args.alphas_per_round,
                run_id=run_row.id,
            )
            total_success += int(result.get("total_success", 0) or 0)
            task.progress_current = total_success
            await db.commit()

            print(
                f"Dataset {dataset_id} done: success={result.get('total_success')} "
                f"target_reached={result.get('target_reached')} "
                f"stop={result.get('dataset_stop_reason')}"
            )
            if total_success >= args.goal:
                break

        task.status = "COMPLETED"
        run_row.status = "COMPLETED"
        run_row.finished_at = datetime.utcnow()
        await db.commit()
        run_id = run_row.id

    await summarize_run(run_id)
    if args.auto_backfill_timeouts:
        backfill_args = argparse.Namespace(
            run_id=[run_id],
            task_id=None,
            location=None,
            failure_limit=args.auto_backfill_failure_limit,
            location_limit=args.auto_backfill_location_limit,
            grace_seconds=args.auto_backfill_grace_seconds,
            include_non_first_order=True,
            first_order_only=False,
            dry_run=False,
        )
        for pass_index in range(max(1, args.auto_backfill_passes)):
            print(f"\nAuto backfill timed-out multi-sims pass {pass_index + 1}/{args.auto_backfill_passes}:")
            summary = await run_timeout_backfill(backfill_args)
            if not summary.get("recovered"):
                break
        await summarize_run(run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IND regular alpha mining through project pipeline.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Specific dataset IDs to mine.")
    parser.add_argument("--auto-datasets", type=int, default=5, help="Use top N auto-selected unlit datasets.")
    parser.add_argument("--include-lit-pyramids", action="store_true", help="Allow datasets with nonzero platform alpha count.")
    parser.add_argument("--goal", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--alphas-per-round", type=int, default=4)
    parser.add_argument("--neutralization", default=None)
    parser.add_argument("--decay", type=int, default=None)
    parser.add_argument("--truncation", type=float, default=None)
    parser.add_argument("--optimization-batches", type=int, default=None)
    parser.add_argument("--optimization-simulation-max-wait", type=int, default=None)
    parser.add_argument("--generation-pool", type=int, default=None)
    parser.add_argument("--template-quota", type=int, default=None)
    parser.add_argument("--simulation-max-wait", type=int, default=None)
    parser.add_argument("--simulation-timeout-grace-seconds", type=int, default=None)
    parser.add_argument("--simulation-no-child-timeout-seconds", type=int, default=None)
    parser.add_argument("--simulation-batch-size", type=int, default=None)
    parser.add_argument("--simulation-cancel-retry-passes", type=int, default=None)
    parser.add_argument("--first-order-probe-start-index", type=int, default=None, help="Start offset into the REGULAR operator probe list.")
    parser.add_argument("--first-order-deferred-retry-slots", type=int, default=None, help="Number of first-order timeout-deferred operators to retry in each probe batch.")
    parser.add_argument("--first-order-skip-covered", action=argparse.BooleanOptionalAction, default=None, help="Skip operators already covered by previous first-order probes.")
    parser.add_argument("--auto-strengthen-after-active-coverage", action=argparse.BooleanOptionalAction, default=None, help="Switch to second-order strengthening once only deferred first-order probe operators remain.")
    parser.add_argument("--first-order-auto-strengthen-max-seeds", type=int, default=None, help="Maximum auto-selected first-order signal seeds for second-order strengthening.")
    parser.add_argument("--first-order-auto-strengthen-variants-per-seed", type=int, default=None, help="Maximum second-order rewrite variants per auto-selected seed.")
    parser.add_argument("--avoid-attempted-expressions", action=argparse.BooleanOptionalAction, default=None, help="Skip exact expressions already attempted for this dataset before DB simulation dedup.")
    parser.add_argument("--attempted-expression-limit", type=int, default=None)
    parser.add_argument("--preferred-fields", nargs="*", default=None, help="Field IDs to prioritize in prompts and deterministic first-order/template generation.")
    parser.add_argument("--auto-backfill-timeouts", action=argparse.BooleanOptionalAction, default=True, help="After mining, re-read timed-out multi-simulation parents through the project backfill workflow.")
    parser.add_argument("--auto-backfill-passes", type=int, default=1)
    parser.add_argument("--auto-backfill-grace-seconds", type=int, default=45)
    parser.add_argument("--auto-backfill-failure-limit", type=int, default=200)
    parser.add_argument("--auto-backfill-location-limit", type=int, default=20)
    parser.add_argument("--settings-sweep-known", action="store_true")
    parser.add_argument("--settings-sweep-only", action="store_true")
    parser.add_argument("--settings-sweep-max-batches", type=int, default=None)
    parser.add_argument(
        "--settings-sweep-seed-json",
        action="append",
        default=None,
        help="JSON seed expression for settings sweep, e.g. '{\"expression\":\"rank(x)\",\"description\":\"seed\"}'.",
    )
    parser.add_argument(
        "--strengthen-seed-json",
        action="append",
        default=None,
        help="JSON seed from a first-order signal, e.g. '{\"expression\":\"rank(x)\",\"probe_operator\":\"rank\",\"sharpe\":0.8}'.",
    )
    parser.add_argument("--settings-sweep-neutralizations", nargs="*", default=["CROWDING", "INDUSTRY", "SUBINDUSTRY"])
    parser.add_argument("--settings-sweep-decays", nargs="*", type=int, default=[2, 4, 6])
    parser.add_argument("--settings-sweep-truncations", nargs="*", type=float, default=[0.04, 0.08])
    parser.add_argument("--no-first-order-probe", action="store_true", help="Disable one-main-operator REGULAR probe candidates.")
    parser.add_argument("--seed-forum-knowledge", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
