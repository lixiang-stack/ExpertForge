# ExpertForge

A configurable domain-expert agent. Define an expert domain in a directory, and the
agent classifies each question's intent and complexity, routes it to a strategy, and
answers via an OpenAI-compatible API. Use the interactive
REPL or the single-shot `--ask` entry point.

## Install

```bash
uv sync
```

## Configure

Copy the example config and fill in your API endpoint, model, and domain directory:

```bash
cp config.example.json config.json
```

Edit `config.json`:

- `base_url`: the OpenAI-compatible API base URL.
- `model`: the model used for classification, answers, and orchestration.
- `model_low`: optional low-cost model tier for `simple` questions (falls back to `model`).
- `model_high`: optional high-capability model tier for `medium`/`complex` questions (falls back to `model`).
- `domain_dir`: path to your domain directory (see below).

`classifier_model` is no longer configured — classification uses `model_low`
(falling back to `model`). When `model_low`/`model_high` are empty (the default in
`config.example.json`), all model tiers fall back to `model`.

Then set your API key:

```bash
export AGENT_API_KEY=your_key
```

`base_url` can also be set via the `AGENT_BASE_URL` environment variable.

### Domain directory

Each expert domain lives in its own directory, e.g. `domain/software_engineering/`:

- `domain.json`: domain name, description, and out-of-domain reply.
- `intents.yaml`: the intents the classifier can detect.
- `intent_mapping.yaml`: maps each intent to a strategy.
- `strategies.yaml`: strategy definitions — per-strategy optional `model`,
  `complexity_gate`, and exactly one `default: true` marker (the strategy used for
  unmapped intents).
- `expert_policy.md`: the shared expert identity and answer policy, prepended to
  the system prompt at runtime.
- `prompts/*.md`: one system prompt per strategy. Each file carries only the
  strategy's specific behavior; the shared expert identity/policy lives in
  `expert_policy.md` and is prepended to the system prompt at runtime, with **no**
  placeholders (`{name}`/`{description}`/`{structure}`).

Strategies are fully data-driven: swapping in a new expert domain means writing a new
domain directory — no code changes. Complex questions on a gated strategy run through
an Orchestrator pipeline (Planner → Workers → Aggregator) that builds on the strategy
prompt.

## Run

Interactive REPL:

```bash
uv run python -m agent
```

Single-shot question:

```bash
uv run python -m agent --ask "question"
```

To use a specific config file:

```bash
uv run python -m agent path/to/config.json
```

Type your question at the `you >` prompt. To leave the REPL, type `exit` (or `quit`),
or press `Ctrl-C` / `Ctrl-D`.

## Observability

Optional token/cost-free usage tracking and trace visualization. Enable it in `config.json`:

```json
{
  "observability": { "enabled": true, "data_dir": ".observability", "phase_map": {} }
}
```

- Every LLM call's tokens and latency are recorded automatically (classification, routing, strategy, and orchestration phases) to per-day JSONL files under `data_dir`.
- During a REPL/`--ask` run a compact per-question line is printed after each answer, showing input tokens, output tokens, and elapsed time.
- After a run, generate the HTML report (self-contained, with per-trace step timelines):

```bash
uv run python -m agent.observability report               # writes data_dir/report.html
uv run python -m agent.observability report --day 2026-08-11
```

`phase_map` optionally remaps the built-in phase names (see `agent/observability/patch.py::DEFAULT_PHASES`). Disabled by default; when disabled the agent behaves exactly as before.

## Evaluation

Golden-dataset evaluation for classification, routing, answer quality, and cost:

```bash
uv run python -m agent.evaluation run                      # full run (all metrics)
uv run python -m agent.evaluation run --skip-quality       # classification/routing/cost only
uv run python -m agent.evaluation run --label my-run       # named result file
uv run python -m agent.evaluation run --results-dir out/   # override the results dir
uv run python -m agent.evaluation run --suite direct teaching  # run specific suites by name
uv run python -m agent.evaluation run --max-per-suite 5    # cap cases per suite
```

Datasets live in `evaluation/datasets/<domain>/` — one directory per domain,
containing one YAML file per suite (e.g. `software_engineering/` holds
`classification.yaml`, `routing.yaml`, `direct.yaml`, `teaching.yaml`,
`debugging.yaml`, `analysis.yaml`, `code_snippet.yaml`, and
`orchestration.yaml`). A single YAML file can also be passed directly. Each run
writes a timestamped JSON result to `evaluation/results/` (gitignored). Compare
two runs:

```bash
# compare two run results (use the paths printed by each run)
uv run python -m agent.evaluation diff evaluation/results/2026-08-15-a.json \
                                   evaluation/results/2026-08-15-b.json
```

### Baseline tracking

Full result files (with per-case answers) are gitignored so the repo stays clean.
To keep a small, committed record of where the system stands — and to see how
each iteration moves it — record a **baseline** after a run:

```bash
# 1. run the evaluation; give the run a distinct label so it doesn't overwrite an
#    earlier result (file = {today}-{label}.json), and note the printed path
uv run python -m agent.evaluation run --label after-prompt-fix

# 2. record the baseline from that result file (use the path printed in step 1);
#    if a baseline.json already exists, the new vs. old metrics diff is printed
uv run python -m agent.evaluation baseline evaluation/results/2026-08-15-after-prompt-fix.json
```

The first time only — before any baseline exists — run once and record it as the
starting point (`run --label initial`, then `baseline <its path>`); from then on,
each iteration needs just the single `run` + `baseline` pair above.

`baseline` writes a metrics-only snapshot (classification/routing/answer-quality
accuracy and cost per suite — no case-level answers) to
`evaluation/results/baseline.json`, which is committed. This file is the single
exception to the `evaluation/results/` ignore rule.

**When to record a baseline:** after any change that is meant to affect quality
or cost — prompt edits, model/strategy/domain changes, new evaluation cases —
commit the updated `baseline.json` together with the change that produced it.

**How to evaluate the effect:** every `baseline` run against an existing
`baseline.json` prints a metrics diff (previous vs. new) right away, so you see
whether the change helped. Beyond that, the committed file's git history is the
iteration log: `git log -p evaluation/results/baseline.json` shows each
improvement or regression over time. For a deep per-case comparison of two full
runs, use `diff` above.

Answer-quality judging uses `evaluation.judge_model` from `config.json` (falls back to `model`).
Evaluation is independent of observability: it reads pipeline return values and its
own usage recorder, so disabling observability does not affect evaluation.

## Testing

### Unit tests

`tests/unit/` is the hermetic unit-test tier. It never calls a live API (the
LLM client is mocked) and is the default run:

```bash
uv run pytest -q
```

### Live tests

`live/` exercises the real agent against the LLM API: `test_smoke.py` runs a
single-shot `--ask` question, an out-of-domain question, and a minimal
evaluation run that writes a result file, all end-to-end; `test_integration.py`
runs a complex gated question through the Orchestrator (Planner → Workers →
Aggregator) and a medium question through a strategy.

Live tests run only when you opt in explicitly and `AGENT_API_KEY` is set.
Run them from the repo root:

```bash
uv run pytest live -v
```

### API key security

The live tests need a real `AGENT_API_KEY` and skip automatically when it is
not set. Provide it **only** through a secure
channel — an environment variable, a secret manager, or a CI secret — and
never hardcode it in code or commit it to the repository:

```bash
export AGENT_API_KEY=your_key
```

For example, in CI set `AGENT_API_KEY` as a repository secret; for local
development you can load it from a `.env` file that is git-ignored. The
tests only read the key from `os.environ` and never log or print it.
