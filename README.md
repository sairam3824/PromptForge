# PromptForge

**Automatic prompt optimization for LLM tasks.** Declare what you want, hand over a few labeled examples, and PromptForge searches for the prompt that maximizes your metric — so you stop hand-tuning wording and let a measured search do it.

Inspired by [DSPy](https://github.com/stanfordnlp/dspy). Runs on OpenAI (`gpt-4o-mini` by default) or fully local [Ollama](https://ollama.com) models. Every LLM call is disk-cached, so re-runs cost nothing and finish in seconds.

```text
Baseline prompt  ──►  val score 0.60
                       │  InstructionSearch  (rewrite the instruction)
                       │  BootstrapFewShot   (mine high-scoring demos)
                       ▼
Optimized prompt ──►  val score 0.95
```

---

## Table of Contents

- [Why](#why)
- [How it works](#how-it-works)
- [Install](#install)
- [Quickstart](#quickstart)
- [Define your own task](#define-your-own-task)
- [Python API](#python-api)
- [Optimizers](#optimizers)
- [Metrics](#metrics)
- [Backends](#backends)
- [CLI reference](#cli-reference)
- [How scoring & budget work](#how-scoring--budget-work)
- [Project layout](#project-layout)
- [Development](#development)
- [License](#license)

---

## Why

Prompt quality is the difference between a 60% and a 95% task — but tuning it by hand is slow, subjective, and forgotten the moment the model changes. PromptForge treats the prompt as something you **measure and search**, not something you guess:

- **Declarative** — describe inputs, outputs, and the goal once as a `Signature`. No prompt-string plumbing.
- **Measured** — every candidate is scored on a held-out validation split with a metric you choose.
- **Tracked** — every candidate prompt and its score is logged to SQLite and viewable as a leaderboard.
- **Cheap to iterate** — all model calls are content-hashed and cached to disk; re-running an optimization replays from cache for free.
- **No vendor lock-in** — same code path for OpenAI or a local Ollama model.

---

## How it works

PromptForge optimizes two parts of a prompt:

1. **The instruction** — `InstructionSearch` asks the LLM to propose alternative phrasings of your task description, evaluates each on the validation split, and greedily hill-climbs to the best one within a call budget.
2. **The few-shot demos** — `BootstrapFewShot` runs your training examples through the current prompt, keeps the ones the model already gets right (score ≥ threshold), and injects the best `k` as in-context examples.

The default `combined` optimizer does both: it finds the best instruction first, then bootstraps demos on top of it. Throughout, a `Signature` renders each candidate into a concrete prompt and parses the model's reply back into structured output fields.

---

## Install

```bash
# OpenAI backend (default) + dev tools
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...
```

For free, fully local inference via Ollama:

```bash
pip install -e .
ollama pull qwen2.5:3b
```

Optional extras:

```bash
pip install -e ".[embeddings]"   # enables the `embedding` metric (sentence-transformers)
```

Requires Python ≥ 3.10.

---

## Quickstart

A ready-to-run news-headline classifier (`headline → sports | technology | politics`) lives in `examples/`:

```bash
# 1. Optimize — search for the best prompt using the training data
promptforge optimize \
  --task examples/news_classification/task.py \
  --train examples/news_classification/train.jsonl \
  --output results/best.json

# 2. Evaluate — score the optimized prompt on held-out test data
promptforge eval \
  --task examples/news_classification/task.py \
  --test examples/news_classification/test.jsonl \
  --prompt-file results/best.json \
  --show-errors

# 3. Leaderboard — browse every candidate prompt ranked by score
promptforge leaderboard
```

No OpenAI key? Append `--backend ollama --model qwen2.5:3b` to the `optimize` and `eval` commands.

---

## Define your own task

A task is just a Python file exposing a `signature` and a JSONL dataset.

**`task.py`**
```python
from promptforge import Signature, Field

signature = Signature(
    inputs=[Field("question", "A trivia question")],
    outputs=[Field("answer", "A short factual answer")],
    task_description="Answer the question concisely.",
)
```

**`train.jsonl`** — one JSON object per line:
```jsonl
{"inputs": {"question": "Capital of France?"}, "outputs": {"answer": "Paris"}}
{"inputs": {"question": "Largest planet?"},   "outputs": {"answer": "Jupiter"}}
```

Then optimize:
```bash
promptforge optimize --task task.py --train train.jsonl --metric f1 --budget 50
```

Signatures support **multiple output fields** — declare more than one `Field` in `outputs` and PromptForge will render/parse the answer as a JSON object automatically.

---

## Python API

The CLI is a thin wrapper; everything is usable directly:

```python
from promptforge import (
    Signature, Field, Module, Example,
    OpenAIBackend, exact_match,
    InstructionSearch,
)

signature = Signature(
    inputs=[Field("headline")],
    outputs=[Field("category")],
    task_description="Classify the headline as sports, technology, or politics.",
)

backend = OpenAIBackend(model="gpt-4o-mini")          # calls are auto-cached
module = Module(signature=signature, backend=backend)

# Run the prompt directly
print(module(headline="Lakers win in overtime"))       # -> {"category": "sports"}

# Or optimize it
train = [Example(inputs={"headline": "..."}, outputs={"category": "sports"})]
best = InstructionSearch().optimize(
    module, val_examples=train, metric=exact_match, run_id="demo", budget=30,
)
print(best.instruction)
```

---

## Optimizers

| `--optimizer` | How it works |
|---|---|
| `combined` *(default)* | Instruction search first, then few-shot bootstrap on the winning instruction. Splits the budget between the two phases. |
| `instruction` | LLM proposes `--n-proposals` instruction variants per round and greedily hill-climbs the best on the val split. |
| `bootstrap` | Runs training examples, keeps those scoring ≥ threshold, injects the best `--k` as few-shot demos. |

---

## Metrics

| `--metric` | Description |
|---|---|
| `exact_match` *(default)* | Normalized exact match (lowercased, punctuation-stripped). |
| `f1` | Token-level F1 overlap (SQuAD-style) — good for free-form text. |
| `embedding` | Cosine similarity of sentence embeddings. Requires `pip install -e ".[embeddings]"`. |
| `llm_judge` | An LLM scores the prediction against gold on a 0.0–1.0 scale. |

---

## Backends

```bash
# OpenAI (default) — needs OPENAI_API_KEY
promptforge optimize --backend openai --model gpt-4o-mini ...

# Ollama (local, zero cost)
promptforge optimize --backend ollama --model qwen2.5:3b ...
```

Both run at `temperature=0` for reproducible, cache-friendly results.

---

## CLI reference

```text
promptforge optimize  --task FILE --train FILE [options]
  -m, --metric        exact_match | f1 | embedding | llm_judge   (default: exact_match)
  -b, --budget        max LLM call budget                        (default: 30)
  -o, --optimizer     bootstrap | instruction | combined         (default: combined)
      --backend       openai | ollama                            (default: openai)
      --model         model name                                 (default: gpt-4o-mini)
      --val-split     fraction of train held out for validation  (default: 0.2)
      --k             few-shot demos to keep (bootstrap)          (default: 3)
      --n-proposals   instruction candidates per round           (default: 5)
      --output        save the best prompt to a JSON file
      --run-id        resume / append to an existing run
      --db            SQLite tracker path                         (default: promptforge.db)
      --cache-dir     LLM cache directory                        (default: .promptforge_cache)
      --seed          shuffle/split seed                          (default: 42)

promptforge eval      --task FILE --test FILE [options]
  -p, --prompt-file   optimized prompt JSON from `optimize --output`
      --show-errors   print mispredicted examples
  (also accepts --metric, --backend, --model, --cache-dir)

promptforge leaderboard  [--run-id ID] [--top-n N] [--db PATH]
  Defaults to the most recent run when --run-id is omitted.
```

---

## How scoring & budget work

- **Validation split.** `optimize` shuffles the training file (seeded) and holds out `--val-split` (default 20%) for scoring candidates. Optimizers never train on the val rows.
- **Budget** is a cap on *live* LLM calls. Cache hits don't count, so resumed runs go fast. The `combined` optimizer divides the budget roughly in half between instruction search and bootstrapping.
- **What gets saved.** `--output` writes the single highest-scoring candidate recorded in the tracker (instruction + demos), not merely the optimizer's final state — a later phase can't accidentally save a worse prompt.
- Each run finishes by printing `api_calls` vs `cache_hits` so you can see exactly what the optimization cost.

---

## Project layout

```text
promptforge/
├── cli.py              CLI — optimize / eval / leaderboard
├── signature.py        Field, Signature — prompt rendering & output parsing
├── module.py           Module — render → call backend → parse
├── example.py          Example — a single (inputs, outputs) record
├── metrics.py          exact_match, f1, embedding, llm_judge
├── cache.py            Disk cache (SHA-256 keyed on model + messages)
├── tracker.py          SQLite — logs every candidate + score
├── backends/           base.py, openai_backend.py, ollama_backend.py
└── optimizers/         bootstrap.py, instruction.py, combined.py
examples/
└── news_classification/   task.py · train.jsonl (25) · test.jsonl (10)
tests/                  pytest suite (signatures, metrics, bootstrap, cache)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

---

## License

[MIT](LICENSE) © 2026 Sai Rama Linga Reddy Maruri
