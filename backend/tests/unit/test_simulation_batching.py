from backend.agents.graph.nodes.evaluation import _padding_expression


def _batch_like_node_simulate(expressions, batch_size):
    jobs = list(enumerate(expressions))
    batches = [
        jobs[offset: offset + batch_size]
        for offset in range(0, len(jobs), batch_size)
    ]
    submitted = []
    for job_batch in batches:
        expression_batch = [expr for _, expr in job_batch]
        if len(expression_batch) == 1:
            expression_batch = [expression_batch[0], _padding_expression()]
        submitted.append(expression_batch)
    return submitted


def test_tail_singleton_is_padded_instead_of_merged_into_three_expression_batch():
    batches = _batch_like_node_simulate(["a", "b", "c"], batch_size=2)

    assert batches == [["a", "b"], ["c", _padding_expression()]]
    assert all(len(batch) == 2 for batch in batches)
