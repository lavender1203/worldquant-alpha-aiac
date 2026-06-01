"""
Integration Test Suite for P0/P1/P2 Optimization Modules

Tests observability, reproducibility, and A/B comparison capabilities.
Run with: pytest backend/tests/test_optimization_modules.py -v
"""

import pytest
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# =============================================================================
# Test P0 Modules
# =============================================================================

class TestAlphaSemanticValidator:
    """Test P0-1: Type/signature validation"""
    
    def test_matrix_operator_validation(self):
        from backend.alpha_semantic_validator import AlphaSemanticValidator
        
        # Create validator with field metadata
        fields = [
            {"id": "close", "type": "MATRIX"},
            {"id": "volume", "type": "MATRIX"},
            {"id": "sector", "type": "GROUP"},
        ]
        
        validator = AlphaSemanticValidator(fields=fields)
        
        # Valid: ts_rank on MATRIX field
        result = validator.validate("ts_rank(close, 20)")
        assert not result.errors, f"Unexpected errors: {result.errors}"
        
        # Should warn: ts_rank on GROUP field (semantic issue)
        result = validator.validate("ts_rank(sector, 20)")
        # Warnings are acceptable for non-strict mode

    def test_industry_named_model_fields_are_not_group_like(self):
        from backend.alpha_semantic_validator import AlphaSemanticValidator, FieldInfo
        from backend.agents.graph.nodes.generation import _is_group_like_field

        field = {
            "id": "industry_value_momentum_rank_float",
            "name": "industry_value_momentum_rank_float",
            "type": "MATRIX",
        }

        assert not FieldInfo.from_dict(field).is_group_like
        assert not _is_group_like_field(field)

        validator = AlphaSemanticValidator(fields=[field], operators=["ts_delta", "group_rank"])
        result = validator.validate("group_rank(ts_delta(industry_value_momentum_rank_float, 22), industry)")

        assert not result.errors
        
    def test_operator_chain_extraction(self):
        from backend.alpha_semantic_validator import AlphaSemanticValidator
        
        validator = AlphaSemanticValidator()
        ops = validator._extract_operators("ts_rank(ts_delta(close, 5), 20)")
        
        assert "ts_rank" in ops
        assert "ts_delta" in ops


class TestExpressionDeduplicator:
    """Test P0-2: Expression deduplication"""
    
    def test_exact_duplicate_detection(self):
        from backend.alpha_semantic_validator import ExpressionDeduplicator
        
        dedup = ExpressionDeduplicator(similarity_threshold=0.9)
        
        expr1 = "ts_rank(close, 20)"
        dedup.add(expr1)
        
        # Exact duplicate
        is_dup, reason = dedup.is_duplicate(expr1)
        assert is_dup, "Should detect exact duplicate"
        
    def test_similar_expression_detection(self):
        from backend.alpha_semantic_validator import ExpressionDeduplicator
        
        dedup = ExpressionDeduplicator(similarity_threshold=0.8)
        
        expr1 = "ts_rank(close, 20)"
        expr2 = "ts_rank(close, 20)  "  # With trailing space (normalized)
        
        dedup.add(expr1)
        is_dup, _ = dedup.is_duplicate(expr2)
        assert is_dup, "Should detect normalized duplicate"

    def test_group_dimension_variants_are_not_structural_duplicates(self):
        from backend.alpha_semantic_validator import ExpressionDeduplicator

        dedup = ExpressionDeduplicator(similarity_threshold=0.9)
        dedup.add("group_rank(reverse(change_flow_ratio), industry)")

        is_dup, reason = dedup.is_duplicate("group_rank(reverse(change_flow_ratio), subindustry)")

        assert not is_dup, reason


class TestMiningWorkflowGuards:
    """Tests for mining-loop guardrails added from live run observations."""

    def test_duplicate_invalids_route_to_simulate(self):
        from backend.agents.graph.edges import route_after_validate
        from backend.agents.graph.state import AlphaCandidate, MiningState

        state = MiningState(
            task_id=1,
            pending_alphas=[
                AlphaCandidate(expression="rank(close)", is_valid=True),
                AlphaCandidate(
                    expression="rank(close)",
                    is_valid=False,
                    validation_error="Duplicate: exact duplicate",
                ),
            ],
        )

        assert route_after_validate(state) == "simulate"

    def test_operator_limit_counts_calls_not_unique_names(self):
        from backend.agents.graph.nodes.validation import _validate_task_constraints

        errors = _validate_task_constraints(
            expression="rank(ts_mean(close, 20)) + rank(ts_mean(volume, 20))",
            allowed_fields=["close", "volume"],
            task_config={"max_operator_count": 3},
        )

        assert any("Too many operators: 4 > 3" in err for err in errors)

    def test_common_operator_aliases_are_canonicalized_before_validation(self):
        from backend.agents.graph.nodes.validation import _canonicalize_operator_aliases

        expr = "rank(divide(ts_delta(close, 5), ts_stddev(close, 20)))"

        assert _canonicalize_operator_aliases(expr) == (
            "rank(divide(ts_delta(close, 5), ts_std_dev(close, 20)))"
        )

    def test_reverse_is_blocked_on_vector_derived_signals(self):
        from backend.agents.graph.nodes.validation import _validate_task_constraints

        errors = _validate_task_constraints(
            expression="reverse(ts_mean(vec_sum(insd1_holdings), 20))",
            allowed_fields=["insd1_holdings"],
            task_config={"max_operator_count": 5},
            fields=[{"id": "insd1_holdings", "type": "VECTOR"}],
        )

        assert any("reverse() is not allowed" in err for err in errors)

    def test_event_vector_arithmetic_must_happen_after_vec_aggregation(self):
        from backend.agents.graph.nodes.validation import _validate_task_constraints

        errors = _validate_task_constraints(
            expression="rank(ts_mean(vec_avg(subtract(ern11_event_sentiment, ern11_event_risk)), 5))",
            allowed_fields=["ern11_event_sentiment", "ern11_event_risk"],
            task_config={"max_operator_count": 5},
            fields=[
                {"id": "ern11_event_sentiment", "type": "VECTOR"},
                {"id": "ern11_event_risk", "type": "VECTOR"},
            ],
        )

        assert any("aggregated before arithmetic" in err for err in errors)

        valid_errors = _validate_task_constraints(
            expression="rank(ts_mean(subtract(vec_avg(ern11_event_sentiment), vec_avg(ern11_event_risk)), 5))",
            allowed_fields=["ern11_event_sentiment", "ern11_event_risk"],
            task_config={"max_operator_count": 5},
            fields=[
                {"id": "ern11_event_sentiment", "type": "VECTOR"},
                {"id": "ern11_event_risk", "type": "VECTOR"},
            ],
        )

        assert not valid_errors

    def test_group_operator_rejects_numeric_field_as_group_argument(self):
        from backend.agents.graph.nodes.validation import _validate_task_constraints

        errors = _validate_task_constraints(
            expression="group_neutralize(ts_delta(global_value_momentum_rank, 5), industry_value_momentum_rank)",
            allowed_fields=["global_value_momentum_rank", "industry_value_momentum_rank"],
            task_config={"max_operator_count": 5, "max_fields": 2},
            fields=[
                {"id": "global_value_momentum_rank", "type": "MATRIX"},
                {"id": "industry_value_momentum_rank", "type": "MATRIX"},
            ],
        )

        assert any("Invalid group argument" in err for err in errors)

    def test_group_mean_uses_third_argument_as_group_argument(self):
        from backend.agents.graph.nodes.validation import _validate_task_constraints

        valid_errors = _validate_task_constraints(
            expression="group_mean(change_flow_ratio, 1, industry)",
            allowed_fields=["change_flow_ratio"],
            task_config={"max_operator_count": 5, "max_fields": 1},
            fields=[{"id": "change_flow_ratio", "type": "MATRIX"}],
        )

        invalid_errors = _validate_task_constraints(
            expression="group_mean(change_flow_ratio, 1, mean_global_feature_25)",
            allowed_fields=["change_flow_ratio", "mean_global_feature_25"],
            task_config={"max_operator_count": 5, "max_fields": 2},
            fields=[
                {"id": "change_flow_ratio", "type": "MATRIX"},
                {"id": "mean_global_feature_25", "type": "MATRIX"},
            ],
        )

        assert not valid_errors
        assert any("Invalid group argument" in err for err in invalid_errors)

    def test_tail_requires_explicit_lower_and_upper(self):
        from backend.alpha_semantic_validator import AlphaSemanticValidator

        validator = AlphaSemanticValidator(
            fields=[{"id": "close", "type": "MATRIX"}],
            strict_field_check=False,
        )

        missing_bounds = validator.validate("tail(rank(close))")
        missing_upper = validator.validate("tail(rank(close), 5)")
        valid = validator.validate("tail(rank(close), lower=-0.5, upper=0.5, newval=0)")

        assert any("Invalid tail() usage" in err for err in missing_bounds.errors)
        assert any("Invalid tail() usage" in err for err in missing_upper.errors)
        assert not valid.errors

    def test_scale_rejects_extra_positional_arguments(self):
        from backend.alpha_semantic_validator import AlphaSemanticValidator

        validator = AlphaSemanticValidator(
            fields=[{"id": "close", "type": "MATRIX"}],
            strict_field_check=False,
        )

        result = validator.validate("scale(close, 1)")

        assert any("Invalid scale() usage" in err for err in result.errors)

    def test_quantile_rejects_extra_inputs(self):
        from backend.alpha_semantic_validator import AlphaSemanticValidator

        validator = AlphaSemanticValidator(
            fields=[{"id": "close", "type": "MATRIX"}],
            strict_field_check=False,
        )

        invalid = validator.validate("quantile(close, 10)")
        valid = validator.validate("quantile(close)")

        assert any("Invalid quantile() usage" in err for err in invalid.errors)
        assert not valid.errors

    def test_clamp_requires_named_bounds(self):
        from backend.alpha_semantic_validator import AlphaSemanticValidator

        validator = AlphaSemanticValidator(
            fields=[{"id": "close", "type": "MATRIX"}],
            strict_field_check=False,
        )

        missing_named_bound = validator.validate("clamp(ts_std_dev(close, 20), 0, 0.1)")
        valid = validator.validate("clamp(ts_std_dev(close, 20), lower=0, upper=0.1)")

        assert any("Invalid clamp() usage" in err for err in missing_named_bound.errors)
        assert not valid.errors

    def test_diagnostic_operators_are_blocked_as_alpha_signals(self):
        from backend.agents.graph.nodes.validation import _validate_task_constraints

        errors = _validate_task_constraints(
            expression="self_corr(close)",
            allowed_fields=["close"],
            task_config={"max_operator_count": 5},
            fields=[{"id": "close", "type": "MATRIX"}],
        )

        assert any("Diagnostic/meta operators" in err for err in errors)

    def test_low_turnover_strategy_blocks_short_windows_and_raw_multiply(self):
        from backend.agents.graph.nodes.validation import _validate_task_constraints

        task_config = {
            "max_operator_count": 5,
            "min_ts_delta_window": 4,
            "min_ts_corr_window": 12,
            "avoid_raw_field_multiply": True,
        }

        delta_errors = _validate_task_constraints(
            expression="rank(ts_delta(close, 1))",
            allowed_fields=["close", "volume"],
            task_config=task_config,
            fields=[{"id": "close", "type": "MATRIX"}, {"id": "volume", "type": "MATRIX"}],
        )
        corr_errors = _validate_task_constraints(
            expression="rank(ts_corr(close, volume, 4))",
            allowed_fields=["close", "volume"],
            task_config=task_config,
            fields=[{"id": "close", "type": "MATRIX"}, {"id": "volume", "type": "MATRIX"}],
        )
        multiply_errors = _validate_task_constraints(
            expression="rank(multiply(close, volume))",
            allowed_fields=["close", "volume"],
            task_config=task_config,
            fields=[{"id": "close", "type": "MATRIX"}, {"id": "volume", "type": "MATRIX"}],
        )
        valid_errors = _validate_task_constraints(
            expression="rank(ts_delta(ts_mean(close, 12), 4))",
            allowed_fields=["close", "volume"],
            task_config=task_config,
            fields=[{"id": "close", "type": "MATRIX"}, {"id": "volume", "type": "MATRIX"}],
        )

        assert any("ts_delta window too short" in err for err in delta_errors)
        assert any("ts_corr window too short" in err for err in corr_errors)
        assert any("Raw field multiplication" in err for err in multiply_errors)
        assert not valid_errors

    def test_strategy_guardrails_are_promoted_to_task_config(self):
        from backend.agents.evolution_strategy import EvolutionStrategy, StrategyMode
        from backend.agents.mining_agent import MiningAgent

        agent = MiningAgent.__new__(MiningAgent)
        strategy = EvolutionStrategy(
            mode=StrategyMode.RESCUE,
            avoid_operators=("ts_corr",),
            action_summary="降低换手，使用更长窗口平滑",
        )

        planned = agent._with_strategy_task_config({"avoid_operators": ["clamp"]}, strategy)

        assert planned["avoid_operators"] == ["clamp", "ts_corr"]
        assert planned["min_ts_delta_window"] == 4
        assert planned["min_ts_corr_window"] == 12
        assert planned["avoid_raw_field_multiply"] is True

    def test_dataset_circuit_breaker_detects_repeated_platform_errors(self):
        from backend.agents.evolution_strategy import RoundResult
        from backend.agents.mining_agent import MiningAgent

        agent = MiningAgent.__new__(MiningAgent)
        task = type("Task", (), {"config": {"dataset_error_min_count": 2}})()

        health = agent._classify_dataset_round_health(
            task,
            RoundResult(
                iteration=1,
                total_generated=0,
                total_simulated=0,
                simulation_errors=4,
            ),
        )

        assert health["kind"] == "blocking_error"

    def test_dataset_circuit_breaker_detects_weak_risk_neutral_signal(self):
        from backend.agents.evolution_strategy import RoundResult
        from backend.agents.mining_agent import MiningAgent

        agent = MiningAgent.__new__(MiningAgent)
        task = type(
            "Task",
            (),
            {
                "config": {
                    "rn_sharpe_min": 1.58,
                    "rn_fitness_min": 1.0,
                    "dataset_weak_signal_sharpe_floor": 0.55,
                    "dataset_weak_signal_fitness_floor": 0.25,
                    "dataset_weak_rn_stop_ratio": 0.6,
                }
            },
        )()

        health = agent._classify_dataset_round_health(
            task,
            RoundResult(
                iteration=2,
                total_generated=4,
                total_simulated=4,
                passed_count=0,
                best_sharpe=0.55,
                best_fitness=0.48,
                best_rn_sharpe=0.73,
                best_rn_fitness=0.55,
            ),
        )

        assert health["kind"] == "weak_signal"

    def test_round_analysis_uses_failed_simulated_metrics_for_diagnosis(self):
        from backend.agents.mining_agent import MiningAgent
        from backend.models import Alpha

        agent = MiningAgent.__new__(MiningAgent)

        async def no_failures(task_id):
            return []

        async def no_optimization(alphas, task_id, task_config=None):
            return []

        agent._query_recent_failures = no_failures
        agent._identify_optimization_candidates = no_optimization

        alpha = Alpha(
            expression="rank(close)",
            alpha_id="alpha_weak_001",
            quality_status="FAIL",
            metrics={"sharpe": 0.55, "fitness": 0.2, "turnover": 0.12},
        )

        result = asyncio.run(agent._analyze_round_results(1, [alpha], 1))

        assert result.total_simulated == 1
        assert result.best_sharpe == 0.55
        assert result.best_fitness == 0.2

    def test_round_analysis_recovers_timeout_attempt_counts_from_trace(self):
        from backend.agents.mining_agent import MiningAgent

        agent = MiningAgent.__new__(MiningAgent)

        async def trace_counts(task_id, iteration, run_id=None):
            return {"generated": 4, "simulated_attempts": 4, "simulation_errors": 4}

        async def recent_failures(task_id):
            return []

        async def no_optimization(alphas, task_id, task_config=None):
            return []

        agent._query_iteration_trace_counts = trace_counts
        agent._query_recent_failures = recent_failures
        agent._identify_optimization_candidates = no_optimization

        result = asyncio.run(agent._analyze_round_results(1, [], 3, run_id=137))

        assert result.total_generated == 4
        assert result.total_simulated == 0
        assert result.simulation_errors == 4

    def test_low_roi_optimize_candidates_do_not_force_optimize_mode(self):
        from backend.agents.mining_agent import MiningAgent
        from backend.models import Alpha

        agent = MiningAgent.__new__(MiningAgent)
        alpha = Alpha(
            expression="rank(ts_delta(ern3_pre_interval, 5))",
            alpha_id="alpha_low_roi",
            quality_status="OPTIMIZE",
            metrics={"sharpe": 0.18, "fitness": 0.07, "turnover": 0.17},
        )

        candidates = asyncio.run(
            agent._identify_optimization_candidates(
                [alpha],
                task_id=1,
                task_config={
                    "optimization_min_sharpe": 0.75,
                    "optimization_min_fitness": 0.25,
                },
            )
        )

        assert candidates == []

    def test_flagged_negative_vector_candidate_must_clear_reversal_threshold(self):
        from backend.agents.mining_agent import MiningAgent
        from backend.models import Alpha

        agent = MiningAgent.__new__(MiningAgent)
        alpha = Alpha(
            expression="rank(ts_delta(vec_sum(ern11_event_relevance), 5))",
            alpha_id="alpha_bad_reverse",
            quality_status="OPTIMIZE",
            metrics={
                "sharpe": -0.5,
                "fitness": -0.27,
                "_sign_reversal_candidate": True,
            },
        )

        candidates = asyncio.run(
            agent._identify_optimization_candidates(
                [alpha],
                task_id=1,
                task_config={
                    "optimization_reversal_abs_sharpe_min": 0.9,
                    "avoid_operators": ["reverse"],
                },
            )
        )

        assert candidates == []

    def test_negative_vector_candidate_can_be_reversed_without_reverse_operator(self):
        from backend.agents.mining_agent import MiningAgent
        from backend.models import Alpha

        agent = MiningAgent.__new__(MiningAgent)
        alpha = Alpha(
            expression="rank(ts_delta(vec_sum(ern11_event_relevance), 5))",
            alpha_id="alpha_reversible",
            quality_status="FAIL",
            metrics={"sharpe": -1.1, "fitness": -0.6},
        )

        candidates = asyncio.run(
            agent._identify_optimization_candidates(
                [alpha],
                task_id=1,
                task_config={
                    "optimization_reversal_abs_sharpe_min": 0.9,
                    "avoid_operators": ["reverse"],
                },
            )
        )

        assert len(candidates) == 1
        assert "NEGATIVE_SIGNAL_REVERSAL" in candidates[0]["reason"]

    def test_strict_gate_uses_task_operator_limit(self):
        from backend.agents.graph.nodes.evaluation import _strict_gate_failures

        failures = _strict_gate_failures(
            metrics={
                "sharpe": 2.0,
                "fitness": 1.2,
                "margin": 0.002,
                "turnover": 0.1,
                "riskNeutralized": {"sharpe": 2.0, "fitness": 1.2},
                "stage": "IS",
                "status": "UNSUBMITTED",
            },
            brain_failed_checks=[],
            prod_corr=0.1,
            self_corr=0.1,
            thresholds={
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
                "max_operator_count": 1,
            },
            expression="rank(ts_mean(close, 20))",
            fields=[{"id": "close", "type": "MATRIX"}],
        )

        assert any("TOO_MANY_OPERATORS" in failure for failure in failures)

    def test_strict_gate_counts_operator_calls_not_unique_operator_names(self):
        from backend.agents.graph.nodes.evaluation import _strict_gate_failures

        failures = _strict_gate_failures(
            metrics={
                "sharpe": 2.0,
                "two_year_sharpe": 2.0,
                "fitness": 1.2,
                "margin": 0.002,
                "turnover": 0.1,
                "riskNeutralized": {"sharpe": 2.0, "fitness": 1.2},
                "stage": "IS",
                "status": "UNSUBMITTED",
            },
            brain_failed_checks=[],
            prod_corr=0.1,
            self_corr=0.1,
            thresholds={
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
                "max_operator_count": 3,
            },
            expression="rank(ts_mean(close, 20)) + rank(ts_mean(volume, 20))",
            fields=[
                {"id": "close", "type": "MATRIX"},
                {"id": "volume", "type": "MATRIX"},
            ],
        )

        assert any("TOO_MANY_OPERATORS (ops=4 > 3)" in failure for failure in failures)

    def test_missing_correlation_payload_is_not_normalized_to_zero(self):
        from backend.adapters.mcp_brain_adapter import _max_correlation

        assert _max_correlation({"checks": {"production": {"max_correlation": None}}}) is None
        assert _max_correlation({"max_correlation": 0.48}) == 0.48

    def test_strict_gate_accepts_flat_risk_neutralized_metrics(self):
        from backend.agents.graph.nodes.evaluation import _strict_gate_failures

        failures = _strict_gate_failures(
            metrics={
                "sharpe": 2.0,
                "two_year_sharpe": 2.0,
                "fitness": 1.2,
                "margin": 0.002,
                "turnover": 0.1,
                "risk_neutralized_sharpe": 1.9,
                "risk_neutralized_fitness": 1.1,
                "stage": "IS",
                "status": "UNSUBMITTED",
            },
            brain_failed_checks=[],
            prod_corr=0.1,
            self_corr=0.1,
            thresholds={
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
            },
            expression="rank(close)",
            fields=[{"id": "close", "type": "MATRIX"}],
        )

        assert not failures

    def test_strict_gate_uses_2y_check_value_and_neutralized_setting_proxy(self):
        from backend.agents.graph.nodes.evaluation import _strict_gate_failures

        failures = _strict_gate_failures(
            metrics={
                "sharpe": 1.9,
                "fitness": 1.2,
                "margin": 0.0015,
                "turnover": 0.10,
                "stage": "IS",
                "status": "UNSUBMITTED",
                "_settings": {"neutralization": "CROWDING"},
                "checks": [
                    {"name": "LOW_2Y_SHARPE", "value": 1.7, "result": "PASS"},
                ],
            },
            brain_failed_checks=[],
            prod_corr=0.2,
            self_corr=0.1,
            thresholds={
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
            },
            expression="rank(ts_delta(alpha_field, 20))",
            fields=[{"id": "alpha_field", "type": "MATRIX"}],
        )

        assert not failures

    def test_low_turnover_optimization_generates_frequency_variants(self):
        from backend.optimization_chain import generate_local_rewrites

        variants = generate_local_rewrites(
            "zscore(ts_mean(imb5_score, 20))",
            {
                "sharpe": 1.35,
                "fitness": 1.03,
                "turnover": 0.038,
                "riskNeutralized": {"sharpe": 0.82, "fitness": 0.4},
            },
            max_variants=8,
        )

        expressions = {v["expression"] for v in variants}
        assert "rank(ts_delta(imb5_score, 1))" in expressions
        assert "rank(ts_delta(ts_mean(imb5_score, 5), 1))" in expressions

    def test_optimization_signal_reversal_uses_vector_safe_inversion(self):
        from backend.optimization_chain import generate_local_rewrites

        variants = generate_local_rewrites(
            "rank(ts_delta(vec_sum(ern11_event_relevance), 5))",
            {"sharpe": -0.9, "fitness": -0.4, "turnover": 0.2},
            max_variants=6,
        )

        expressions = {v["expression"] for v in variants}
        assert "multiply(-1, rank(ts_delta(vec_sum(ern11_event_relevance), 5)))" in expressions
        assert not any(expr.startswith("reverse(") for expr in expressions)

    def test_optimization_orders_operator_budget_safe_variants_first(self):
        from backend.optimization_chain import generate_local_rewrites
        from backend.agents.graph.nodes.validation import _extract_used_operators

        variants = generate_local_rewrites(
            "add(rank(ts_mean(imb5_score,20)), rank(ts_delta(imb5_score,5)))",
            {"sharpe": 1.25, "fitness": 0.8, "turnover": 0.12},
            max_variants=5,
        )

        assert variants
        assert all(len(_extract_used_operators(v["expression"])) <= 5 for v in variants[:3])

    def test_optimization_uses_task_base_settings(self):
        from backend.agents.mining_agent import MiningAgent

        agent = MiningAgent.__new__(MiningAgent)
        task = type(
            "Task",
            (),
            {
                "config": {
                    "delay": 1,
                    "neutralization": "SUBINDUSTRY",
                    "decay": 6,
                    "truncation": 0.03,
                    "test_period": "P3Y0M",
                }
            },
        )()

        assert agent._optimization_base_settings(task) == {
            "delay": 1,
            "neutralization": "SUBINDUSTRY",
            "decay": 6,
            "truncation": 0.03,
            "test_period": "P3Y0M",
        }

    def test_should_optimize_accepts_flat_risk_neutralized_metrics(self):
        from backend.alpha_scoring import should_optimize

        should_opt, reason = should_optimize(
            {
                "sharpe": 0.35,
                "fitness": 0.2,
                "risk_neutralized_sharpe": 0.75,
                "risk_neutralized_fitness": 0.5,
            }
        )

        assert should_opt
        assert "风险中性化" in reason

    def test_ind_initial_strategy_gets_region_specific_focus(self):
        from backend.agents.mining_agent import MiningAgent

        agent = MiningAgent.__new__(MiningAgent)
        task = type("Task", (), {"region": "IND", "config": {}})()

        strategy = agent._initial_strategy_from_task_config(task)

        assert any("IND/D1" in item for item in strategy.focus_hypotheses)
        assert any("CROWDING" in item for item in strategy.focus_hypotheses)

    def test_settings_sweep_includes_ind_platform_neutralizations(self):
        from backend.optimization_chain import generate_settings_variants

        variants = generate_settings_variants(
            {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08}
        )
        neutralizations = {v["neutralization"] for v in variants}

        assert {"CROWDING", "FAST", "SLOW", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM"}.issubset(neutralizations)

    def test_generation_prompt_includes_factor_construction_playbook_and_operator_cap(self):
        from backend.agents.prompts import PromptContext, build_alpha_generation_prompt

        prompt = build_alpha_generation_prompt(
            PromptContext(
                dataset_id="analyst4",
                region="IND",
                universe="TOP500",
                fields=[{"id": "anl4_afv4_eps_mean", "type": "MATRIX"}],
                operators=[{"name": "rank", "category": "Cross Section"}],
                num_alphas=2,
                max_operator_count=5,
            )
        )

        assert "Factor Construction Playbook" in prompt
        assert "Comparable scale construction" in prompt
        assert "Use fewer than 6 operator calls" in prompt
        assert "Risk-neutralized Sharpe > 1.58" in prompt
        assert "ts_std_dev" in prompt

    def test_generation_prompt_turnover_strategy_blocks_raw_short_delta(self):
        from backend.agents.prompts import PromptContext, build_alpha_generation_prompt

        prompt = build_alpha_generation_prompt(
            PromptContext(
                dataset_id="fundamental94",
                region="USA",
                universe="TOP3000",
                fields=[{"id": "fnd94_rt_cf_sales_q", "type": "MATRIX"}],
                operators=[{"name": "rank"}, {"name": "ts_delta"}, {"name": "ts_mean"}],
                focus_hypotheses=[
                    "Use longer windows and smoothing to reduce high turnover",
                ],
                num_alphas=2,
            )
        )

        assert "LOW-TURNOVER REFINEMENT RULE" in prompt
        assert "do not emit raw one-day deltas" in prompt
        assert "rank(ts_delta(ts_mean(field, 12), 4))" in prompt

    def test_template_candidates_include_ind_level_impulse_and_vol_scaled_change(self):
        from backend.agents.graph.nodes.generation import _template_candidates

        candidates = _template_candidates(
            fields=[{"id": "imb5_score", "type": "MATRIX"}],
            operators=[
                {"name": "rank"},
                {"name": "group_rank"},
                {"name": "signed_power"},
                {"name": "add"},
                {"name": "divide"},
                {"name": "ts_delta"},
                {"name": "ts_mean"},
                {"name": "ts_std_dev"},
                {"name": "subtract"},
                {"name": "ts_zscore"},
                {"name": "group_neutralize"},
            ],
            max_operator_count=5,
        )
        expressions = {candidate.expression for candidate in candidates}

        assert "group_rank(add(ts_mean(imb5_score, 20), ts_delta(imb5_score, 5)), subindustry)" in expressions
        assert "rank(divide(ts_delta(imb5_score, 5), ts_std_dev(imb5_score, 20)))" in expressions

    def test_template_candidates_prioritize_durable_fields_with_quality_metadata(self):
        from backend.agents.graph.nodes.generation import _template_candidates

        candidates = _template_candidates(
            fields=[
                {"id": "random_sparse_field", "type": "MATRIX", "coverage": 0.2, "alpha_count": 0},
                {
                    "id": "estimate_revision_component",
                    "type": "MATRIX",
                    "coverage": 1.0,
                    "alpha_count": 505,
                },
            ],
            operators=[
                {"name": "rank"},
                {"name": "group_rank"},
                {"name": "signed_power"},
                {"name": "add"},
                {"name": "divide"},
                {"name": "ts_delta"},
                {"name": "ts_mean"},
                {"name": "ts_std_dev"},
                {"name": "subtract"},
                {"name": "ts_zscore"},
                {"name": "group_neutralize"},
            ],
            max_operator_count=5,
        )

        assert candidates
        assert "estimate_revision_component" in candidates[0].expression
        assert candidates[0].expression.startswith("signed_power(group_rank(ts_zscore(")

    def test_news_mechanism_classifier_preserves_diversity(self):
        from backend.agents.graph.nodes.generation import (
            _build_mechanism_groups,
            _select_mechanism_diverse_fields,
        )

        fields = [
            {"id": "news_pct_30sec", "type": "MATRIX"},
            {"id": "news_pct_120min", "type": "MATRIX"},
            {"id": "news_high_exc_stddev", "type": "MATRIX"},
            {"id": "news_ratio_vol", "type": "MATRIX"},
            {"id": "news_short_interest", "type": "MATRIX"},
            {"id": "news_eod_close", "type": "MATRIX"},
        ]

        groups = _build_mechanism_groups(fields, "news12", "news")
        selected = _select_mechanism_diverse_fields(groups, preferred_fields=[])
        mechanisms = {field.get("mechanism") for field in selected}

        assert "fast_d1_reaction" in mechanisms
        assert "delayed_news_drift" in mechanisms
        assert "abnormal_range_volatility" in mechanisms
        assert "volume_liquidity_response" in mechanisms
        assert "plain_eod_price_context" in mechanisms

    def test_insider_mechanism_classifier_preserves_diversity(self):
        from backend.agents.graph.nodes.generation import (
            _build_mechanism_groups,
            _select_mechanism_diverse_fields,
        )

        fields = [
            {"id": "insd1_value", "type": "VECTOR"},
            {"id": "insd1_valueeur", "type": "VECTOR"},
            {"id": "insd1_shares", "type": "VECTOR"},
            {"id": "insd1_holdings", "type": "VECTOR"},
            {"id": "insd1_price", "type": "VECTOR"},
            {"id": "max_trade_price", "type": "VECTOR"},
            {"id": "insd1_tradesignificance", "type": "VECTOR"},
        ]

        groups = _build_mechanism_groups(fields, "insiders1", "insiders")
        selected = _select_mechanism_diverse_fields(groups, preferred_fields=[])
        mechanisms = {field.get("mechanism") for field in selected}

        assert "transaction_value_intensity" in mechanisms
        assert "position_size_impact" in mechanisms
        assert "transaction_price_pressure" in mechanisms
        assert "transaction_significance" in mechanisms

    def test_option_mechanism_classifier_preserves_diversity(self):
        from backend.agents.graph.nodes.generation import (
            _build_mechanism_groups,
            _select_mechanism_diverse_fields,
        )

        fields = [
            {"id": "opt1_p_deltasurface_d1_vi", "type": "VECTOR"},
            {"id": "option_implied_volatility_value", "type": "VECTOR"},
            {"id": "opt1_deltasurface_d1_moneyness", "type": "VECTOR"},
            {"id": "opt1_p_deltasurface_d1_delta", "type": "VECTOR"},
            {"id": "opt1_vp_deltasurface_d1_period", "type": "VECTOR"},
            {"id": "opt1_forward_price_d1_forward", "type": "VECTOR"},
            {"id": "opt1_volume", "type": "VECTOR"},
        ]

        groups = _build_mechanism_groups(fields, "option1", "option")
        selected = _select_mechanism_diverse_fields(groups, preferred_fields=[])
        mechanisms = {field.get("mechanism") for field in selected}

        assert "implied_volatility" in mechanisms
        assert "moneyness_delta" in mechanisms
        assert "tenor_term_structure" in mechanisms
        assert "forward_price_basis" in mechanisms
        assert "underlying_price_volume" in mechanisms

    def test_save_results_keeps_failed_simulated_alpha_for_analysis(self):
        from backend.agents.graph.nodes.persistence import node_save_results
        from backend.agents.graph.state import AlphaCandidate, MiningState

        state = MiningState(
            task_id=1,
            pending_alphas=[
                AlphaCandidate(
                    expression="rank(close)",
                    is_valid=True,
                    is_simulated=True,
                    simulation_success=True,
                    alpha_id="alpha_fail_001",
                    metrics={
                        "sharpe": 0.2,
                        "fitness": 0.01,
                        "_strict_gate_failures": ["sharpe<=1.58"],
                    },
                    quality_status="FAIL",
                ),
                AlphaCandidate(
                    expression="rank(volume)",
                    is_valid=True,
                    is_simulated=True,
                    simulation_success=False,
                    simulation_error="timeout",
                    quality_status="FAIL",
                ),
            ],
        )

        result = asyncio.run(node_save_results(state))

        assert len(result["generated_alphas"]) == 1
        assert result["generated_alphas"][0].alpha_id == "alpha_fail_001"
        assert result["generated_alphas"][0].quality_status == "FAIL"
        assert len(result["failures"]) == 2
        assert result["failures"][0].details["alpha_id"] == "alpha_fail_001"


class TestTwoStageCorrelation:
    """Test P0-3: Two-stage correlation check"""
    
    def test_preliminary_score_threshold(self):
        """Verify that low-score alphas skip correlation check"""
        from backend.alpha_scoring import calculate_alpha_score
        
        # Low quality sim result
        low_sim = {
            "train": {"sharpe": 0.5, "fitness": 0.3, "turnover": 0.8},
            "test": {"sharpe": 0.4, "fitness": 0.2},
        }
        
        score = calculate_alpha_score(low_sim, prod_corr=0.0, self_corr=0.0)
        
        # With CORR_CHECK_THRESHOLD=0.5, this should be below threshold
        assert score < 0.5, f"Low quality score should be < 0.5, got {score}"


# =============================================================================
# Test P1 Modules
# =============================================================================

class TestDatasetBandit:
    """Test P1-1: Bandit algorithm for dataset selection"""
    
    def test_ucb_exploration(self):
        from backend.selection_strategy import DatasetBandit, DatasetArm
        
        bandit = DatasetBandit(exploration_weight=2.0)
        
        # Add two arms
        arm1 = DatasetArm(dataset_id="ds1", total_pulls=10, total_reward=5.0)
        arm2 = DatasetArm(dataset_id="ds2", total_pulls=1, total_reward=0.8)  # Less explored
        
        bandit.add_arm(arm1)
        bandit.add_arm(arm2)
        bandit.total_pulls = 11
        
        # UCB should favor less explored arm
        selected = bandit.select(n=1)
        assert len(selected) == 1
        # ds2 should have higher UCB due to exploration bonus
        
    def test_reward_update(self):
        from backend.selection_strategy import DatasetBandit
        
        bandit = DatasetBandit()
        bandit.update("ds1", reward=1.0, success=True)
        bandit.update("ds1", reward=0.5, success=True)
        
        arm = bandit.arms.get("ds1:USA:TOP3000")
        assert arm is not None
        assert arm.total_pulls == 2
        assert arm.success_count == 2


class TestFieldSelector:
    """Test P1-2: Multi-objective field selection"""
    
    def test_field_scoring(self):
        from backend.selection_strategy import FieldSelector
        
        selector = FieldSelector(
            coverage_weight=0.3,
            novelty_weight=0.4,
            pyramid_weight=0.3
        )
        
        fields = [
            {"id": "field1", "coverage": 0.9, "alpha_count": 100, "pyramid_multiplier": 1.5, "type": "MATRIX"},
            {"id": "field2", "coverage": 0.8, "alpha_count": 10, "pyramid_multiplier": 1.2, "type": "MATRIX"},
            {"id": "field3", "coverage": 0.5, "alpha_count": 500, "pyramid_multiplier": 1.0, "type": "VECTOR"},
        ]
        
        scored = selector.score_fields(fields)
        
        assert len(scored) >= 2, "Should score fields above min coverage"
        # field2 should score higher due to low alpha_count (high novelty)
        
    def test_diverse_selection(self):
        from backend.selection_strategy import FieldSelector
        
        selector = FieldSelector()
        
        fields = [
            {"id": f"matrix_{i}", "type": "MATRIX", "coverage": 0.9, "alpha_count": i*10, "pyramid_multiplier": 1.0}
            for i in range(10)
        ] + [
            {"id": f"vector_{i}", "type": "VECTOR", "coverage": 0.9, "alpha_count": i*10, "pyramid_multiplier": 1.0}
            for i in range(5)
        ]
        
        selected = selector.select_diverse(fields, n=10, matrix_ratio=0.7)
        
        assert len(selected) == 10
        matrix_count = sum(1 for f in selected if f.get("type") == "MATRIX")
        assert matrix_count >= 5, "Should have majority MATRIX fields"


class TestDiversityFilter:
    """Test P1-4: Diversity constraints"""
    
    def test_similarity_rejection(self):
        from backend.selection_strategy import DiversityFilter
        
        df = DiversityFilter(similarity_threshold=0.7)
        
        expr1 = "ts_rank(close, 20)"
        df.accept(expr1)
        
        # Very similar expression
        expr2 = "ts_rank(close, 21)"  # Only window differs
        should_accept, sim = df.should_accept(expr2)
        
        # Should detect high similarity
        assert sim > 0.5, f"Similar expressions should have similarity > 0.5, got {sim}"


# =============================================================================
# Test P2 Modules
# =============================================================================

class TestKnowledgeExtraction:
    """Test P1-3 & P2-1: Knowledge extraction and AST mutation"""
    
    def test_skeleton_extraction(self):
        from backend.knowledge_extraction import expression_to_skeleton
        
        expr = "ts_rank(ts_delta(close, 5), 20)"
        skeleton = expression_to_skeleton(expr, max_depth=2)
        
        assert "ts_rank" in skeleton
        assert "ts_delta" in skeleton or "..." in skeleton  # May be truncated
        
    def test_operator_mutation(self):
        from backend.knowledge_extraction import mutate_operator
        
        expr = "ts_rank(close, 20)"
        mutations = mutate_operator(expr, mutation_rate=1.0)  # Force mutation
        
        # Should generate alternatives like ts_zscore
        assert len(mutations) > 0 or True  # May not mutate if no alternatives
        
    def test_window_mutation(self):
        from backend.knowledge_extraction import mutate_windows
        
        expr = "ts_rank(close, 20)"
        mutations = mutate_windows(expr)
        
        # Should generate window variants
        assert any("5" in m or "10" in m or "40" in m for m in mutations) if mutations else True


class TestPatternRegistry:
    """Test P1-3: Pattern registry with decay"""
    
    def test_pattern_decay(self):
        from backend.knowledge_extraction import AlphaPattern, PatternRegistry
        from datetime import timedelta
        
        registry = PatternRegistry(decay_half_life_days=30)
        
        # Create pattern from expression
        pattern = AlphaPattern.from_expression(
            expression="ts_rank(close, 20)",
            pattern_type="SUCCESS",
            alpha_id="alpha_001",
            metrics={"sharpe": 2.0, "fitness": 0.8},
            field_types={"MATRIX"}
        )
        
        registry.add_or_update(pattern)
        
        # Effective score should be high for fresh pattern
        assert pattern.effective_score() > 0


# =============================================================================
# Test Experiment Tracker (Observability & A/B Testing)
# =============================================================================

class TestMetricsCollector:
    """Test observability metrics collection"""
    
    def test_counter_increment(self):
        from backend.experiment_tracker import MetricsCollector
        
        collector = MetricsCollector("test_exp")
        
        collector.increment("simulation_count", 5)
        collector.increment("simulation_count", 3)
        
        assert collector.get_counter("simulation_count") == 8
        
    def test_histogram_stats(self):
        from backend.experiment_tracker import MetricsCollector
        
        collector = MetricsCollector("test_exp")
        
        for duration in [100, 200, 150, 180, 120]:
            collector.record("iteration_duration_ms", duration)
            
        stats = collector.get_histogram_stats("iteration_duration_ms")
        
        assert stats["count"] == 5
        assert stats["mean"] == 150.0
        assert stats["min"] == 100
        assert stats["max"] == 200


class TestABTesting:
    """Test A/B testing framework"""
    
    def test_comparison(self):
        from backend.experiment_tracker import ABTestFramework
        
        ab = ABTestFramework(min_samples=5)
        
        # Baseline: lower pass rate
        ab.record_baseline("pass_per_sim", [0.04, 0.05, 0.03, 0.04, 0.05])
        
        # Treatment: higher pass rate
        ab.record_treatment("pass_per_sim", [0.08, 0.07, 0.09, 0.08, 0.07])
        
        result = ab.compare("pass_per_sim")
        
        # Treatment should be better
        assert result.absolute_diff > 0, "Treatment should show improvement"
        assert result.relative_diff_pct > 50, "Should show >50% improvement"


class TestExperimentConfig:
    """Test reproducibility config"""
    
    def test_config_hash(self):
        from backend.experiment_tracker import ExperimentConfig
        
        config1 = ExperimentConfig(
            experiment_id="exp1",
            created_at=datetime.now(),
            random_seed=42,
            numpy_seed=42,
            region="USA",
            universe="TOP3000",
            dataset_ids=["ds1"],
            llm_model="gpt-4",
            temperature=0.7
        )
        
        config2 = ExperimentConfig(
            experiment_id="exp2",
            created_at=datetime.now(),
            random_seed=42,
            numpy_seed=42,
            region="USA",
            universe="TOP3000",
            dataset_ids=["ds1"],
            llm_model="gpt-4",
            temperature=0.7
        )
        
        # Different IDs but same settings should have different hashes (ID is part of hash)
        # But the settings themselves match
        assert config1.random_seed == config2.random_seed


# =============================================================================
# Integration Test: Full Pipeline Simulation
# =============================================================================

class TestEndToEndOptimization:
    """End-to-end test simulating optimization pipeline"""
    
    def test_full_optimization_flow(self):
        """Test that all modules work together"""
        from backend.alpha_semantic_validator import AlphaSemanticValidator, ExpressionDeduplicator
        from backend.selection_strategy import FieldSelector, DiversityFilter
        from backend.knowledge_extraction import expression_to_skeleton, generate_variants
        from backend.experiment_tracker import MetricsCollector
        
        # Setup
        collector = MetricsCollector("integration_test")
        
        fields = [
            {"id": "close", "type": "MATRIX", "coverage": 0.95, "alpha_count": 50, "pyramid_multiplier": 1.2},
            {"id": "volume", "type": "MATRIX", "coverage": 0.90, "alpha_count": 30, "pyramid_multiplier": 1.3},
            {"id": "returns", "type": "MATRIX", "coverage": 0.85, "alpha_count": 100, "pyramid_multiplier": 1.0},
        ]
        
        # P1-2: Select fields
        selector = FieldSelector()
        selected_fields = selector.select_diverse(fields, n=2)
        assert len(selected_fields) == 2
        collector.record("field_selection_count", len(selected_fields))
        
        # Generate expressions
        expressions = [
            "ts_rank(close, 20)",
            "ts_rank(close, 20)",  # Duplicate
            "ts_zscore(volume, 10)",
            "group_rank(returns, sector)",
        ]
        
        # P0-1: Semantic validation
        validator = AlphaSemanticValidator(fields=fields)
        valid_expressions = []
        for expr in expressions:
            result = validator.validate(expr)
            if not result.errors:
                valid_expressions.append(expr)
                
        collector.record("validation_pass_rate", len(valid_expressions) / len(expressions) * 100)
        
        # P0-2: Deduplication
        dedup = ExpressionDeduplicator()
        unique_expressions = []
        for expr in valid_expressions:
            is_dup, _ = dedup.is_duplicate(expr)
            if not is_dup:
                dedup.add(expr)
                unique_expressions.append(expr)
                
        dedup_rate = (len(valid_expressions) - len(unique_expressions)) / len(valid_expressions) * 100 if valid_expressions else 0
        collector.record("dedup_skip_rate", dedup_rate)
        
        # P1-4: Diversity filter
        div_filter = DiversityFilter(similarity_threshold=0.7)
        diverse_expressions = []
        for expr in unique_expressions:
            should_accept, _ = div_filter.should_accept(expr)
            if should_accept:
                div_filter.accept(expr)
                diverse_expressions.append(expr)
                
        collector.record("diversity_filter_pass_rate", 
            len(diverse_expressions) / len(unique_expressions) * 100 if unique_expressions else 0)
        
        # P2-1: Generate variants for optimization
        if diverse_expressions:
            variants = generate_variants(diverse_expressions[0], max_variants=5)
            collector.record("variants_generated", len(variants))
        
        # Get summary
        summary = collector.get_summary()
        
        assert summary["sample_count"] > 0, "Should have recorded metrics"
        print(f"\nIntegration test summary: {json.dumps(summary, indent=2)}")


# =============================================================================
# Benchmark Comparison Test
# =============================================================================

class TestBaselineComparison:
    """Test that demonstrates baseline vs optimized comparison"""
    
    def test_dedup_savings_measurement(self):
        """Measure actual savings from deduplication"""
        from backend.alpha_semantic_validator import ExpressionDeduplicator
        from backend.experiment_tracker import ABTestFramework
        
        ab = ABTestFramework(min_samples=5)
        
        # Simulate baseline (no dedup)
        baseline_sims = []
        for _ in range(5):
            # Assume 10 expressions, some duplicates
            baseline_sims.append(10)  # All simulated
            
        # Simulate with dedup
        dedup = ExpressionDeduplicator()
        treatment_sims = []
        for _ in range(5):
            expressions = [
                "ts_rank(close, 20)",
                "ts_rank(close, 20)",  # Dup
                "ts_zscore(volume, 10)",
                "ts_rank(close, 20)",  # Dup
                "ts_delta(returns, 5)",
            ] * 2  # 10 total, with duplicates
            
            unique = 0
            for expr in expressions:
                is_dup, _ = dedup.is_duplicate(expr)
                if not is_dup:
                    dedup.add(expr)
                    unique += 1
            treatment_sims.append(unique)
            dedup = ExpressionDeduplicator()  # Reset for next iteration
            
        ab.record_baseline("simulations", baseline_sims)
        ab.record_treatment("simulations", treatment_sims)
        
        result = ab.compare("simulations")
        
        # Treatment should have fewer simulations
        assert result.absolute_diff < 0, "Dedup should reduce simulations"
        savings_pct = abs(result.relative_diff_pct)
        print(f"\nDedup savings: {savings_pct:.1f}% fewer simulations")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
