"""
News headline classification task.

Signature: headline -> category (sports | technology | politics)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from promptforge import Field, Signature

signature = Signature(
    inputs=[Field("headline", "A short news headline")],
    outputs=[Field("category", "One of: sports, technology, or politics")],
    task_description=(
        "Classify the news headline into exactly one of three categories: "
        "sports, technology, or politics. "
        "Reply with only the category name in lowercase."
    ),
)
