from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class Field:
    name: str
    description: str = ""

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class Signature:
    """Declares inputs, outputs, and a task description; renders/parses prompts."""

    inputs: list[Field]
    outputs: list[Field]
    task_description: str

    @property
    def _multi_output(self) -> bool:
        return len(self.outputs) > 1

    def render_prompt(
        self,
        inputs: dict[str, str],
        demos: list | None = None,
        instruction: str | None = None,
    ) -> str:
        instruction = (instruction or self.task_description).strip()
        parts: list[str] = [instruction]

        if demos:
            parts.append("\nExamples:")
            for i, demo in enumerate(demos, 1):
                parts.append(f"\n[Example {i}]")
                for f in self.inputs:
                    parts.append(f"{f.label}: {demo.inputs.get(f.name, '')}")
                if self._multi_output:
                    out_dict = {f.name: demo.outputs.get(f.name, "") for f in self.outputs}
                    parts.append(f"Output (JSON): {json.dumps(out_dict)}")
                else:
                    out = self.outputs[0]
                    parts.append(f"{out.label}: {demo.outputs.get(out.name, '')}")

        parts.append("\n---")
        for f in self.inputs:
            parts.append(f"{f.label}: {inputs.get(f.name, '')}")

        if self._multi_output:
            keys = [f.name for f in self.outputs]
            parts.append(f"\nOutput the following as a JSON object with keys: {keys}")
            parts.append("Output (JSON):")
        else:
            parts.append(f"{self.outputs[0].label}:")

        return "\n".join(parts)

    def parse_output(self, response: str) -> dict[str, str]:
        response = response.strip()

        if not self._multi_output:
            out_label = self.outputs[0].label + ":"
            lines = response.splitlines()
            first_line = lines[0].strip() if lines else ""
            if first_line.lower().startswith(out_label.lower()):
                value = first_line[len(out_label):].strip()
                # label was on its own line — take the next non-empty line
                if not value:
                    value = next((l.strip() for l in lines[1:] if l.strip()), response)
                return {self.outputs[0].name: value}
            return {self.outputs[0].name: first_line or response}

        # Allow nested braces by matching the outermost { ... }
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {f.name: str(parsed.get(f.name, "")) for f in self.outputs}
            except json.JSONDecodeError:
                pass

        result: dict[str, str] = {}
        for line in response.splitlines():
            for f in self.outputs:
                if line.lower().startswith(f.label.lower() + ":"):
                    result[f.name] = line.split(":", 1)[1].strip()
        return result

    def __repr__(self) -> str:
        ins = ", ".join(f.name for f in self.inputs)
        outs = ", ".join(f.name for f in self.outputs)
        return f"Signature({ins} -> {outs})"
