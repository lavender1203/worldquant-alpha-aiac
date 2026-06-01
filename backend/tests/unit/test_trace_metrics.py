from dataclasses import dataclass

from backend.services.trace_metrics import normalize_round_summary_metrics


@dataclass
class Step:
    iteration: int
    step_type: str
    output_data: dict


def test_normalize_round_summary_metrics_recovers_timeout_attempts():
    steps = [
        Step(
            iteration=3,
            step_type="CODE_GEN",
            output_data={"expressions": ["a", "b", "c", "d"]},
        ),
        Step(
            iteration=3,
            step_type="SIMULATE",
            output_data={
                "simulated_count": 4,
                "success_count": 0,
                "results": [{"err": "timeout"} for _ in range(4)],
            },
        ),
        Step(
            iteration=3,
            step_type="ROUND_SUMMARY",
            output_data={
                "round_metrics": {
                    "total_generated": 0,
                    "total_simulated": 0,
                    "simulation_errors": 0,
                }
            },
        ),
    ]

    normalize_round_summary_metrics(steps)

    metrics = steps[-1].output_data["round_metrics"]
    assert metrics["total_generated"] == 4
    assert metrics["simulation_errors"] == 4
    assert metrics["total_attempted"] == 4
