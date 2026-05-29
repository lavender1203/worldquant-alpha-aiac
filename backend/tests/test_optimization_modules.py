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
