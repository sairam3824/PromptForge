from __future__ import annotations

from typing import Any

from .backends.base import LLMBackend
from .example import Example
from .signature import Signature


class Module:
    """Renders a prompt from (Signature + demos + instruction), calls the backend,
    parses the output back to a field dict."""

    def __init__(
        self,
        signature: Signature,
        backend: LLMBackend,
        demos: list[Example] | None = None,
        instruction: str | None = None,
    ) -> None:
        self.signature = signature
        self.backend = backend
        self.demos: list[Example] = demos or []
        self.instruction: str = instruction or signature.task_description

    def forward(self, **inputs: Any) -> dict[str, Any]:
        prompt = self.signature.render_prompt(
            inputs=inputs,
            demos=self.demos,
            instruction=self.instruction,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.backend.generate(messages)
        return self.signature.parse_output(response)

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        return self.forward(**inputs)

    def clone(
        self,
        demos: list[Example] | None = None,
        instruction: str | None = None,
    ) -> "Module":
        return Module(
            signature=self.signature,
            backend=self.backend,
            demos=demos if demos is not None else list(self.demos),
            instruction=instruction if instruction is not None else self.instruction,
        )
