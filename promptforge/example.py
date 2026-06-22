from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Example:
    inputs: dict[str, Any]
    outputs: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(inputs=d["inputs"], outputs=d["outputs"])

    def to_dict(self) -> dict:
        return {"inputs": self.inputs, "outputs": self.outputs}
