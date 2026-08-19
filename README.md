# ExpertForge

A configurable domain-expert agent. Define an expert domain in a directory, and
the agent classifies each question's intent and complexity, routes it to a
strategy, and answers via an OpenAI-compatible API — from an interactive REPL
or the single-shot `--ask` entry point.

## Features

- **Data-driven domains**: an expert domain is a directory (`domain.json`,
  `intents.yaml`, `intent_mapping.yaml`, `orchestration.yaml`,
  `expert_policy.md`, `prompts/*.md`); adding a domain means writing a new
  directory, not changing code.
- **Intent & complexity routing**: each intent maps to a strategy, and gated
  questions run through the Orchestrator pipeline (Planner → Workers →
  Aggregator).
- **Model tiers**: optional `model_low`/`model_high` tiers for cheap/fast
  answers, falling back to `model`.
- **Golden-dataset evaluation**: classification, routing, answer quality, and
  cost metrics, with result diffing and committed baseline tracking.
- **Observability**: optional token/cost tracing per LLM call and a
  self-contained HTML report.
- **Logging**: optional structlog JSONL logs covering lifecycle, routing,
  errors, and evaluation events.

## Quick Start

```bash
uv sync
cp config.example.json config.json
export AGENT_API_KEY=your_key
```

Edit `config.json` to set `base_url`, `model`, and `domain_dir` for your API
and domain. `base_url` can also be set with the `AGENT_BASE_URL` env var.

Run the interactive REPL:

```bash
uv run python -m agent
```

Ask a single question (prints the answer and exits):

```bash
uv run python -m agent --ask "How do I ..."
```

A specific config file works too: `uv run python -m agent path/to/config.json`.

Evaluate against the golden dataset (needs `AGENT_API_KEY`):

```bash
uv run python -m agent.evaluation run
uv run python -m agent.evaluation run --suite direct teaching
```

## Configuration

`config.json` is a JSON object. This README covers the main sections; it does
not enumerate every field.

| Key | Purpose |
| --- | --- |
| `base_url` | OpenAI-compatible API base URL (env: `AGENT_BASE_URL`). |
| `model` | Default model for classification, answers, and orchestration. |
| `model_low` / `model_high` | Optional cost tiers for `simple` vs `medium`/`complex` questions; fall back to `model`. |
| `domain_dir` | Path to the expert domain directory. |
| `provider` / `provider_capabilities` | Provider name and capability flags (e.g. `supports_json_schema`, `supports_thinking_toggle`). |
| `evaluation.results_dir` | Where evaluation result files are written. |
| `evaluation.judge` | Judge-model client: `base_url`, `model`, `provider`, `provider_capabilities`, `timeout`. `base_url`/`model`/`provider` fall back to the main client's values. The judge uses its own key from the `AGENT_JUDGE_API_KEY` env var (never stored in the config file). |
| `observability` | Token/cost tracing: `enabled`, `data_dir`, optional `phase_map`. Disabled by default. |
| `logging` | structlog JSONL logging: `enabled`, `level` (DEBUG…CRITICAL), `file`. |

With observability enabled, generate the HTML report after a run:

```bash
uv run python -m agent.observability report
```

<details>
<summary>Observability & evaluation commands</summary>

Observability report (trace JSONL → self-contained HTML):

```bash
uv run python -m agent.observability report                 # writes data_dir/report.html
uv run python -m agent.observability report --data-dir DIR  # custom trace directory
uv run python -m agent.observability report --day 2026-08-11  # filter to one day
```

Evaluation runs (all need `AGENT_API_KEY`; a judge block also needs `AGENT_JUDGE_API_KEY`):

```bash
uv run python -m agent.evaluation run                        # full run (all metrics)
uv run python -m agent.evaluation run --skip-quality         # classification/routing/cost only
uv run python -m agent.evaluation run --label my-run         # named result file
uv run python -m agent.evaluation run --results-dir out/     # override the results dir
uv run python -m agent.evaluation run --suite direct teaching  # run specific suites by name
uv run python -m agent.evaluation run --max-per-suite 5      # cap cases per suite
uv run python -m agent.evaluation run --config path.json     # custom agent config
```

Compare two runs (use the paths printed by each run):

```bash
uv run python -m agent.evaluation diff evaluation/results/a.json evaluation/results/b.json
```

Record a committed metrics-only baseline from a run result (prints the delta vs.
an existing baseline):

```bash
uv run python -m agent.evaluation baseline evaluation/results/2026-08-15-a.json
```

</details>

## Development

```bash
uv run pytest -q              # hermetic unit tests (no live API)
uv run pytest tests/live -v   # live end-to-end tests (needs AGENT_API_KEY)
```

Live tests skip automatically when `AGENT_API_KEY` is unset; provide it via an
env var, secret manager, or CI secret — never commit it. No separate lint tool
is configured.
