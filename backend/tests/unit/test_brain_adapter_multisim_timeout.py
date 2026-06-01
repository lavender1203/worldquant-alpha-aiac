import asyncio

from backend.adapters.brain_adapter import BrainAdapter


class _Response:
    def __init__(self, data, status_code=200, headers=None, text=""):
        self._data = data
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._data


class _TimeoutBoundaryBrainAdapter(BrainAdapter):
    def __init__(self):
        super().__init__(email="unused", password="unused")
        self.calls = []

    async def _safe_api_call(self, method: str, endpoint: str, **kwargs):
        self.calls.append((method, endpoint))
        if endpoint == "https://api.worldquantbrain.com/simulations/parent":
            return _Response({
                "status": "COMPLETE",
                "children": ["child_a", "child_b"],
            })
        if endpoint == "/simulations/child_a":
            return _Response({"status": "COMPLETE", "alpha": "alpha_a"})
        if endpoint == "/simulations/child_b":
            return _Response({"status": "COMPLETE", "alpha": "alpha_b"})
        raise AssertionError(f"unexpected endpoint {endpoint}")

    async def _get_completed_alpha_details(self, alpha_id: str):
        return {
            "success": True,
            "alpha_id": alpha_id,
            "metrics": {"sharpe": 1.0},
        }


class _PendingAlphaDetailsBrainAdapter(BrainAdapter):
    def __init__(self):
        super().__init__(email="unused", password="unused")
        self.calls = 0

    async def _safe_api_call(self, method: str, endpoint: str, **kwargs):
        self.calls += 1
        return _Response({"status": "PENDING"}, headers={"Retry-After": "0.01"})


class _GraceCaptureBrainAdapter(BrainAdapter):
    def __init__(self):
        super().__init__(email="unused", password="unused")
        self.wait_kwargs = None

    async def _safe_api_call(self, method: str, endpoint: str, **kwargs):
        return _Response(
            {"id": "parent"},
            status_code=201,
            headers={"Location": "/simulations/parent"},
        )

    async def _wait_for_multisim(
        self,
        location: str,
        max_wait: int = 1200,
        timeout_grace_seconds: int = 180,
        no_child_timeout_seconds: int = 0,
    ):
        self.wait_kwargs = {
            "location": location,
            "max_wait": max_wait,
            "timeout_grace_seconds": timeout_grace_seconds,
            "no_child_timeout_seconds": no_child_timeout_seconds,
        }
        return {
            "success": True,
            "results": [{"success": True, "alpha_id": "alpha_a"}],
        }


class _NoChildStaleBrainAdapter(BrainAdapter):
    def __init__(self):
        super().__init__(email="unused", password="unused")
        self.calls = 0

    async def _safe_api_call(self, method: str, endpoint: str, **kwargs):
        self.calls += 1
        return _Response(
            {"status": "PENDING", "progress": 0.35, "children": []},
            headers={"Retry-After": "0.01"},
        )


def test_multisim_timeout_boundary_fetches_completed_children():
    adapter = _TimeoutBoundaryBrainAdapter()

    result = asyncio.run(adapter._wait_for_multisim(
        "https://api.worldquantbrain.com/simulations/parent",
        max_wait=0,
    ))

    assert result["success"] is True
    assert [item["alpha_id"] for item in result["results"]] == ["alpha_a", "alpha_b"]
    assert all(item["location"].endswith("/simulations/parent") for item in result["results"])


def test_alpha_details_polling_is_bounded():
    adapter = _PendingAlphaDetailsBrainAdapter()

    result = asyncio.run(adapter._get_completed_alpha_details("alpha_pending", max_wait=0.02))

    assert result["success"] is False
    assert result["alpha_id"] == "alpha_pending"
    assert "timed out" in result["error"]
    assert adapter.calls >= 1


def test_simulate_batch_passes_timeout_grace_to_parent_waiter():
    adapter = _GraceCaptureBrainAdapter()

    result = asyncio.run(adapter.simulate_batch(["rank(close)", "rank(open)"], max_wait=12, timeout_grace_seconds=7))

    assert result[0]["alpha_id"] == "alpha_a"
    assert adapter.wait_kwargs == {
        "location": "/simulations/parent",
        "max_wait": 12,
        "timeout_grace_seconds": 7,
        "no_child_timeout_seconds": 0,
    }


def test_multisim_no_child_stale_timeout_returns_fast_trackable_failure():
    adapter = _NoChildStaleBrainAdapter()

    result = asyncio.run(adapter._wait_for_multisim(
        "https://api.worldquantbrain.com/simulations/stuck-parent",
        max_wait=10,
        no_child_timeout_seconds=0.02,
    ))

    assert result["success"] is False
    assert result["location"].endswith("/simulations/stuck-parent")
    assert "no-child timed out" in result["error"]
    assert "children=0" in result["error"]
    assert adapter.calls >= 2
