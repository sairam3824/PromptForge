import pytest

from promptforge import Example, Field, Module, Signature
from promptforge.backends.base import LLMBackend


class MockBackend(LLMBackend):
    """Deterministic backend for tests — returns canned responses keyed by substrings."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = "mock") -> None:
        super().__init__()
        self._responses = responses or {}
        self._default = default

    @property
    def model_name(self) -> str:
        return "mock"

    def generate(self, messages: list[dict], **kwargs) -> str:
        self.call_count += 1
        content = messages[-1]["content"]
        for key, val in self._responses.items():
            if key in content:
                return val
        return self._default


@pytest.fixture
def simple_sig():
    return Signature(
        inputs=[Field("question")],
        outputs=[Field("answer")],
        task_description="Answer the question briefly.",
    )


@pytest.fixture
def multi_sig():
    return Signature(
        inputs=[Field("text")],
        outputs=[Field("name"), Field("age")],
        task_description="Extract name and age.",
    )


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def train_examples():
    return [
        Example(inputs={"question": f"Q{i}"}, outputs={"answer": f"A{i}"})
        for i in range(10)
    ]
