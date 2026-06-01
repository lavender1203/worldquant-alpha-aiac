from backend.agents.graph.nodes.generation import (
    _augment_with_template_candidates,
    _first_order_expression_for_operator,
    _first_order_operator_probe_candidates,
    _op_count,
    _second_order_strengthening_candidates,
)


def test_first_order_probe_covers_regular_operators_and_excludes_combo_only():
    fields = [
        {"id": "alpha_signal_a", "type": "MATRIX", "coverage": 0.9},
        {"id": "alpha_signal_b", "type": "MATRIX", "coverage": 0.8},
        {"id": "alpha_signal_c", "type": "MATRIX", "coverage": 0.7},
        {"id": "vector_signal", "type": "VECTOR", "coverage": 0.9},
    ]
    operators = [
        {"name": "rank", "scope": ["REGULAR"], "category": "Cross Sectional"},
        {"name": "ts_corr", "scope": ["REGULAR"], "category": "Time Series"},
        {"name": "group_rank", "scope": ["REGULAR"], "category": "Group"},
        {"name": "vec_sum", "scope": ["REGULAR"], "category": "Vector"},
        {"name": "generate_stats", "scope": ["COMBO"], "category": "Transformational"},
    ]

    candidates = _first_order_operator_probe_candidates(fields, operators, max_operator_count=5)
    by_operator = {
        candidate.metadata["probe_operator"]: candidate
        for candidate in candidates
    }

    assert set(by_operator) == {"rank", "ts_corr", "group_rank", "vec_sum"}
    assert by_operator["rank"].expression == "rank(alpha_signal_a)"
    assert by_operator["ts_corr"].expression == "ts_corr(alpha_signal_a, alpha_signal_b, 60)"
    assert by_operator["group_rank"].expression == "group_rank(alpha_signal_a, industry)"
    assert by_operator["vec_sum"].expression == "vec_sum(vector_signal)"
    assert all(_op_count(candidate.expression) <= 5 for candidate in candidates)


def test_first_order_probe_skips_vector_operator_without_vector_field():
    fields = [
        {"id": "alpha_signal_a", "type": "MATRIX", "coverage": 0.9},
        {"id": "alpha_signal_b", "type": "MATRIX", "coverage": 0.8},
    ]
    operators = [
        {"name": "rank", "scope": ["REGULAR"]},
        {"name": "vec_sum", "scope": ["REGULAR"]},
    ]

    candidates = _first_order_operator_probe_candidates(fields, operators, max_operator_count=5)
    probed = {candidate.metadata["probe_operator"] for candidate in candidates}

    assert probed == {"rank"}


def test_first_order_probeable_names_exclude_vector_ops_without_vector_field():
    from backend.agents.mining_agent import MiningAgent

    fields = [{"id": "alpha_signal_a", "type": "MATRIX", "coverage": 0.9}]
    operators = [
        {"name": "rank", "scope": ["REGULAR"]},
        {"name": "vec_sum", "scope": ["REGULAR"]},
    ]

    probeable = MiningAgent._first_order_probeable_operator_names(
        fields=fields,
        operators=operators,
        max_operator_count=5,
        regular_order=["rank", "vec_sum"],
    )

    assert probeable == ["rank"]


def test_first_order_probe_start_index_offsets_short_task_batches():
    fields = [{"id": "alpha_signal_a", "type": "MATRIX", "coverage": 0.9}]
    operators = [
        {"name": "rank", "scope": ["REGULAR"]},
        {"name": "group_rank", "scope": ["REGULAR"]},
        {"name": "ts_delta", "scope": ["REGULAR"]},
        {"name": "ts_zscore", "scope": ["REGULAR"]},
    ]

    candidates = _augment_with_template_candidates(
        pending_alphas=[],
        fields=fields,
        operators=operators,
        task_config={
            "first_order_operator_probe": True,
            "first_order_operator_probe_batch_size": 2,
            "first_order_operator_probe_start_index": 2,
            "deterministic_templates_first": True,
            "generation_candidate_pool": 2,
            "template_candidate_quota": 2,
            "max_operator_count": 5,
        },
        target_count=2,
    )

    assert [candidate.metadata["probe_operator"] for candidate in candidates] == [
        "ts_delta",
        "ts_zscore",
    ]


def test_first_order_probe_target_operators_override_offset_and_preserve_order():
    fields = [{"id": "alpha_signal_a", "type": "MATRIX", "coverage": 0.9}]
    operators = [
        {"name": "rank", "scope": ["REGULAR"]},
        {"name": "group_rank", "scope": ["REGULAR"]},
        {"name": "ts_delta", "scope": ["REGULAR"]},
        {"name": "ts_zscore", "scope": ["REGULAR"]},
    ]

    candidates = _augment_with_template_candidates(
        pending_alphas=[],
        fields=fields,
        operators=operators,
        task_config={
            "first_order_operator_probe": True,
            "first_order_operator_probe_batch_size": 2,
            "first_order_operator_probe_start_index": 2,
            "first_order_operator_probe_target_operators": ["ts_zscore", "rank"],
            "deterministic_templates_first": True,
            "generation_candidate_pool": 2,
            "template_candidate_quota": 2,
            "max_operator_count": 5,
        },
        target_count=2,
    )

    assert [candidate.metadata["probe_operator"] for candidate in candidates] == [
        "ts_zscore",
        "rank",
    ]


def test_first_order_probe_excludes_completed_operators_before_offset_slice():
    fields = [{"id": "alpha_signal_a", "type": "MATRIX", "coverage": 0.9}]
    operators = [
        {"name": "rank", "scope": ["REGULAR"]},
        {"name": "group_rank", "scope": ["REGULAR"]},
        {"name": "ts_delta", "scope": ["REGULAR"]},
        {"name": "ts_zscore", "scope": ["REGULAR"]},
    ]

    candidates = _augment_with_template_candidates(
        pending_alphas=[],
        fields=fields,
        operators=operators,
        task_config={
            "first_order_operator_probe": True,
            "first_order_operator_probe_batch_size": 3,
            "first_order_operator_probe_exclude_operators": ["rank", "ts_delta"],
            "deterministic_templates_first": True,
            "generation_candidate_pool": 3,
            "template_candidate_quota": 3,
            "max_operator_count": 5,
        },
        target_count=3,
    )

    probe_candidates = [
        candidate for candidate in candidates
        if candidate.metadata.get("source") == "first_order_operator_probe"
    ]
    assert [candidate.metadata["probe_operator"] for candidate in probe_candidates] == [
        "group_rank",
        "ts_zscore",
    ]


def test_generation_skips_exact_attempted_expressions_before_db_dedup():
    fields = [{"id": "alpha_signal_a", "type": "MATRIX", "coverage": 0.9}]
    operators = [
        {"name": "rank", "scope": ["REGULAR"]},
        {"name": "group_rank", "scope": ["REGULAR"]},
    ]

    candidates = _augment_with_template_candidates(
        pending_alphas=[],
        fields=fields,
        operators=operators,
        task_config={
            "first_order_operator_probe": True,
            "first_order_operator_probe_target_operators": ["rank", "group_rank"],
            "attempted_expressions": ["rank(alpha_signal_a)"],
            "deterministic_templates_first": True,
            "generation_candidate_pool": 2,
            "template_candidate_quota": 2,
            "max_operator_count": 5,
        },
        target_count=2,
    )

    expressions = [candidate.expression for candidate in candidates]

    assert "rank(alpha_signal_a)" not in expressions
    assert "group_rank(alpha_signal_a, industry)" in expressions


def test_first_order_generation_prioritizes_configured_preferred_fields():
    fields = [
        {"id": "change_flow_ratio", "type": "MATRIX", "coverage": 0.9},
        {"id": "change_operating_cashflow_margin_2", "type": "MATRIX", "coverage": 0.8},
    ]
    operators = [{"name": "rank", "scope": ["REGULAR"]}]

    candidates = _augment_with_template_candidates(
        pending_alphas=[],
        fields=fields,
        operators=operators,
        task_config={
            "first_order_operator_probe": True,
            "first_order_operator_probe_target_operators": ["rank"],
            "preferred_fields": ["change_operating_cashflow_margin_2"],
            "deterministic_templates_first": True,
            "generation_candidate_pool": 1,
            "template_candidate_quota": 1,
            "max_operator_count": 5,
        },
        target_count=1,
    )

    assert candidates[0].expression == "rank(change_operating_cashflow_margin_2)"


def test_first_order_probe_plan_interleaves_timeout_deferred_retries():
    from backend.agents.mining_agent import MiningAgent

    targets = MiningAgent._compose_first_order_probe_targets(
        ordered_ops=["add", "tail", "to_nan", "ts_corr", "ts_covariance"],
        active_remaining=["ts_corr", "ts_covariance"],
        deferred_remaining=["add", "tail", "to_nan"],
        deferred_counts={"add": 3, "tail": 1, "to_nan": 2},
        target_size=3,
        deferred_retry_slots=1,
    )

    assert targets == ["ts_corr", "ts_covariance", "tail"]


def test_first_order_probe_plan_retries_deferred_when_active_queue_empty():
    from backend.agents.mining_agent import MiningAgent

    targets = MiningAgent._compose_first_order_probe_targets(
        ordered_ops=["add", "tail", "to_nan"],
        active_remaining=[],
        deferred_remaining=["add", "tail", "to_nan"],
        deferred_counts={"add": 3, "tail": 1, "to_nan": 2},
        target_size=2,
        deferred_retry_slots=0,
    )

    assert targets == ["tail", "to_nan"]


def test_first_order_signal_details_merge_into_strengthening_seeds():
    from backend.agents.mining_agent import MiningAgent

    seeds = MiningAgent._merge_first_order_strengthening_seeds(
        existing_seeds=[
            {"expression": "reverse(change_flow_ratio)", "probe_operator": "reverse"},
        ],
        signal_details=[
            {
                "operator": "reverse",
                "alpha_id": "old",
                "expression": "reverse(change_flow_ratio)",
                "sharpe": 0.99,
                "fitness": 0.63,
                "turnover": 0.036,
                "margin": 0.0027,
            },
            {
                "operator": "bucket",
                "alpha_id": "new",
                "expression": 'group_count(change_flow_ratio, bucket(rank(change_flow_ratio), range="0,1,0.1"))',
                "sharpe": 0.99,
                "fitness": 0.64,
                "turnover": 0.158,
                "margin": 0.0008,
            },
        ],
        max_auto_seeds=1,
        max_variants_per_seed=3,
    )

    assert len(seeds) == 2
    assert seeds[0]["expression"] == "reverse(change_flow_ratio)"
    assert seeds[1]["probe_operator"] == "bucket"
    assert seeds[1]["max_variants"] == 3


def test_timeout_backfill_includes_all_project_candidate_sources_by_default():
    import json
    from types import SimpleNamespace

    from scripts.backfill_multisim_timeouts import _should_include_failure

    failure = SimpleNamespace(
        raw_response=json.dumps({
            "metrics": {"_simulation_location": "https://api.worldquantbrain.com/simulations/abc"},
            "candidate_metadata": {"source": "second_order_strengthening"},
        })
    )

    assert _should_include_failure(SimpleNamespace(first_order_only=False), failure)
    assert not _should_include_failure(SimpleNamespace(first_order_only=True), failure)


def test_first_order_special_cases_return_numeric_alpha_shapes():
    assert (
        _first_order_expression_for_operator("densify", "x", "y", "z", None, None)
        == "group_count(x, densify(industry))"
    )
    assert (
        _first_order_expression_for_operator("group_cartesian_product", "x", "y", "z", None, None)
        == "group_count(x, group_cartesian_product(industry, subindustry))"
    )
    assert (
        _first_order_expression_for_operator("group_mean", "x", "y", "z", None, None)
        == "group_mean(x, 1, industry)"
    )
    assert (
        _first_order_expression_for_operator("scale_down", "x", "y", "z", None, None)
        == "scale_down(x)"
    )
    assert (
        _first_order_expression_for_operator("round_down", "x", "y", "z", None, None)
        == "round_down(x)"
    )
    assert (
        _first_order_expression_for_operator("bucket", "x", "y", "z", None, None)
        == 'group_count(x, bucket(rank(x), range="0,1,0.1"))'
    )
    assert (
        _first_order_expression_for_operator("if_else", "x", "y", "z", None, None)
        == "if_else(x > 0, y, 0)"
    )
    assert (
        _first_order_expression_for_operator("jump_decay", "x", "y", "z", None, None)
        == "jump_decay(x, 20, sensitivity=0.5, force=0.1)"
    )
    assert (
        _first_order_expression_for_operator("kth_element", "x", "y", "z", None, None)
        == "kth_element(x, 20, k=1)"
    )
    assert (
        _first_order_expression_for_operator("tail", "x", "y", "z", None, None)
        == "tail(x, lower=-0.5, upper=0.5, newval=0)"
    )
    assert (
        _first_order_expression_for_operator("to_nan", "x", "y", "z", None, None)
        == "to_nan(x)"
    )
    assert (
        _first_order_expression_for_operator("ts_rank_gmean_amean_diff", "x", "y", "z", None, None)
        == "ts_rank_gmean_amean_diff(x, y, x, y, 60)"
    )
    assert (
        _first_order_expression_for_operator("ts_target_tvr_decay", "x", "y", "z", None, None)
        == "ts_target_tvr_decay(x, lambda_min=0, lambda_max=1, target_tvr=0.1)"
    )
    assert (
        _first_order_expression_for_operator("ts_target_tvr_delta_limit", "x", "y", "z", None, None)
        == "ts_target_tvr_delta_limit(x, y, lambda_min=0, lambda_max=1, target_tvr=0.1)"
    )
    assert (
        _first_order_expression_for_operator("ts_target_tvr_hump", "x", "y", "z", None, None)
        == "ts_target_tvr_hump(x, lambda_min=0, lambda_max=1, target_tvr=0.1)"
    )
    assert (
        _first_order_expression_for_operator("ts_triple_corr", "x", "y", "z", None, None)
        == "ts_triple_corr(x, y, x, 60)"
    )


def test_trade_when_probe_bypasses_default_trade_when_ban_only_for_probe():
    from backend.agents.graph.nodes.validation import _validate_task_constraints

    expression = "trade_when(x, y, -1)"
    fields = [{"id": "x", "type": "MATRIX"}, {"id": "y", "type": "MATRIX"}]
    task_config = {"no_trade_when": True, "max_operator_count": 5}

    regular_errors = _validate_task_constraints(
        expression=expression,
        allowed_fields=["x", "y"],
        fields=fields,
        task_config=task_config,
    )
    probe_errors = _validate_task_constraints(
        expression=expression,
        allowed_fields=["x", "y"],
        fields=fields,
        task_config=task_config,
        candidate_metadata={
            "source": "first_order_operator_probe",
            "probe_operator": "trade_when",
        },
    )

    assert "Forbidden operator: trade_when" in regular_errors
    assert "Forbidden operator: trade_when" not in probe_errors


def test_local_validators_accept_brain_named_parameters():
    from backend.alpha_semantic_validator import AlphaSemanticValidator
    from validator import ExpressionValidator

    fields = [{"id": "x", "type": "MATRIX"}, {"id": "y", "type": "MATRIX"}]
    expressions = [
        "to_nan(x)",
        "tail(x, lower=-0.5, upper=0.5, newval=0)",
        "kth_element(x, 20, k=1)",
        "ts_target_tvr_decay(x, lambda_min=0, lambda_max=1, target_tvr=0.1)",
        "ts_target_tvr_delta_limit(x, y, lambda_min=0, lambda_max=1, target_tvr=0.1)",
        "ts_target_tvr_hump(x, lambda_min=0, lambda_max=1, target_tvr=0.1)",
    ]

    for expression in expressions:
        syntax = ExpressionValidator().check_expression(expression, allowed_fields=["x", "y"])
        semantic = AlphaSemanticValidator(
            fields=fields,
            strict_field_check=True,
            strict_type_check=True,
        ).validate(expression)

        assert syntax["valid"], syntax["errors"]
        assert semantic.valid, semantic.errors


def test_optimization_field_extraction_skips_function_tokens():
    from backend.optimization_chain import _first_field_like_token

    assert _first_field_like_token("arc_tan(change_residual_return_variance)") == "change_residual_return_variance"
    assert (
        _first_field_like_token(
            'group_count(change_residual_return_variance, bucket(rank(change_residual_return_variance), range="0,1,0.1"))'
        )
        == "change_residual_return_variance"
    )


def test_first_order_weak_signal_gets_second_order_strengthening_variants():
    from backend.optimization_chain import generate_local_rewrites

    expression = 'group_count(change_flow_ratio, bucket(rank(change_flow_ratio), range="0,1,0.1"))'
    variants = generate_local_rewrites(
        expression,
        {
            "sharpe": 0.99,
            "fitness": 0.64,
            "turnover": 0.158,
            "_candidate_metadata": {
                "source": "first_order_operator_probe",
                "probe_operator": "bucket",
            },
        },
        max_variants=8,
    )

    expressions = [variant["expression"] for variant in variants]

    assert expressions[0] == f"group_rank({expression}, industry)"
    assert f"group_rank({expression}, subindustry)" in expressions
    assert f"rank({expression})" in expressions
    assert f"ts_delta({expression}, 5)" in expressions
    assert all(_op_count(candidate) <= 5 for candidate in expressions)


def test_low_turnover_first_order_signal_prioritizes_turnover_lift():
    from backend.optimization_chain import generate_local_rewrites

    expression = "reverse(change_flow_ratio)"
    variants = generate_local_rewrites(
        expression,
        {
            "sharpe": 0.99,
            "fitness": 0.63,
            "turnover": 0.0368,
            "_candidate_metadata": {
                "source": "first_order_operator_probe",
                "probe_operator": "reverse",
            },
        },
        max_variants=4,
    )

    assert variants[0]["expression"] == f"ts_delta({expression}, 5)"


def test_generation_can_strengthen_first_order_seed_across_runs():
    candidates = _second_order_strengthening_candidates(
        [
            {
                "expression": "winsorize(change_flow_ratio)",
                "probe_operator": "winsorize",
                "sharpe": -0.99,
                "fitness": -0.63,
                "turnover": 0.0368,
            }
        ],
        max_operator_count=5,
    )

    expressions = [candidate.expression for candidate in candidates]

    assert "multiply(-1, winsorize(change_flow_ratio))" in expressions
    assert "group_rank(winsorize(change_flow_ratio), industry)" in expressions
    assert all(candidate.metadata["operator_order"] == 2 for candidate in candidates)
    assert all(_op_count(expression) <= 5 for expression in expressions)


def test_second_order_strengthening_interleaves_multiple_signal_seeds():
    candidates = _second_order_strengthening_candidates(
        [
            {
                "expression": "reverse(change_flow_ratio)",
                "probe_operator": "reverse",
                "sharpe": 0.99,
                "fitness": 0.63,
                "turnover": 0.0368,
                "max_variants": 2,
            },
            {
                "expression": "multiply(change_flow_ratio, change_3m_revision_fy2_eps)",
                "probe_operator": "multiply",
                "sharpe": 0.65,
                "fitness": 0.35,
                "turnover": 0.0504,
                "max_variants": 2,
            },
        ],
        max_operator_count=5,
    )

    assert candidates[0].metadata["parent_probe_operator"] == "reverse"
    assert candidates[1].metadata["parent_probe_operator"] == "multiply"


def test_second_order_probe_variants_only_apply_to_first_order_probe_metadata():
    from backend.optimization_chain import generate_local_rewrites

    variants = generate_local_rewrites(
        "rank(change_flow_ratio)",
        {"sharpe": 0.7, "fitness": 0.3, "turnover": 0.1},
        max_variants=5,
    )

    assert all(
        not variant["description"].startswith("Second-order")
        for variant in variants
    )


def test_negative_first_order_signal_does_not_trigger_weak_dataset_stop():
    from types import SimpleNamespace

    from backend.agents.evolution_strategy import RoundResult
    from backend.agents.mining_agent import MiningAgent

    agent = MiningAgent.__new__(MiningAgent)
    result = RoundResult(
        iteration=1,
        total_generated=2,
        total_simulated=2,
        passed_count=0,
        best_sharpe=-0.99,
        best_abs_sharpe=0.99,
        best_fitness=-0.63,
        best_abs_fitness=0.63,
    )

    health = agent._classify_dataset_round_health(
        SimpleNamespace(config={"optimization_reversal_abs_sharpe_min": 0.8}),
        result,
    )

    assert health["kind"] == "healthy"
