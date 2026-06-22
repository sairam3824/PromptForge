from promptforge import Example, Field, Module, Signature
from promptforge.metrics import exact_match
from promptforge.optimizers import BootstrapFewShot
from tests.conftest import MockBackend


def _make_module(responses: dict[str, str] | None = None) -> Module:
    sig = Signature(
        inputs=[Field("q")],
        outputs=[Field("a")],
        task_description="Answer briefly.",
    )
    return Module(sig, MockBackend(responses or {}))


def _examples(n: int = 10) -> list[Example]:
    return [Example(inputs={"q": f"Q{i}"}, outputs={"a": f"A{i}"}) for i in range(n)]


def test_bootstrap_produces_module_with_demos():
    responses = {f"Q{i}": f"A{i}" for i in range(10)}
    mod = _make_module(responses)
    opt = BootstrapFewShot(k=3, threshold=0.5)
    result = opt.optimize(mod, _examples(), exact_match, "run-1", budget=20)
    assert len(result.demos) <= 3


def test_bootstrap_demos_count_respects_k():
    responses = {f"Q{i}": f"A{i}" for i in range(10)}
    mod = _make_module(responses)
    for k in (1, 2, 3):
        opt = BootstrapFewShot(k=k, threshold=0.5)
        result = opt.optimize(mod, _examples(), exact_match, f"run-k{k}", budget=20)
        assert len(result.demos) <= k


def test_bootstrap_no_good_demos_returns_empty():
    mod = _make_module({"Q0": "WRONG"})
    opt = BootstrapFewShot(k=3, threshold=1.0)
    result = opt.optimize(mod, _examples(3), exact_match, "run-empty", budget=10)
    assert result.demos == []


def test_bootstrap_respects_budget():
    mod = _make_module()
    opt = BootstrapFewShot(k=3)
    opt.optimize(mod, _examples(20), exact_match, "run-budget", budget=5)
    assert mod.backend.call_count <= 5


def test_bootstrap_budget_is_relative_not_absolute():
    # Simulate the CombinedOptimizer scenario: backend already has call_count > 0
    # before bootstrap runs. Budget should be measured from the current call count,
    # not from zero — otherwise bootstrap exits immediately.
    responses = {f"Q{i}": f"A{i}" for i in range(10)}
    mod = _make_module(responses)
    mod.backend.call_count = 20  # pretend instruction search already used 20 calls

    opt = BootstrapFewShot(k=3, threshold=0.5)
    result = opt.optimize(mod, _examples(10), exact_match, "run-rel", budget=10)
    # With the fix (relative budget), bootstrap should score up to 10 examples.
    # call_count should be 20 + (up to 10) = at most 30.
    assert mod.backend.call_count <= 30
    # And it should have found some demos (all mock responses are correct).
    assert len(result.demos) > 0


def test_bootstrap_preserves_instruction():
    responses = {f"Q{i}": f"A{i}" for i in range(5)}
    mod = _make_module(responses)
    mod.instruction = "Custom instruction."
    opt = BootstrapFewShot(k=2)
    result = opt.optimize(mod, _examples(5), exact_match, "run-instr", budget=10)
    assert result.instruction == "Custom instruction."
