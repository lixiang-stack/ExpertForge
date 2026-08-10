# ExpertForge

A configurable domain-expert agent. Define an expert domain in a directory, and the
agent classifies each question's intent and complexity, routes it to one of five
strategy processors, and answers via an OpenAI-compatible API. Use the interactive
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
- `model`: the base model, used for answers unless overridden below.
- `model_low`: low-cost model tier for `simple` questions (falls back to `model`).
- `model_high`: high-capability model tier for `medium`/`complex` questions (falls back to `model`).
- `domain_dir`: path to your domain directory (see below).

`classifier_model` is no longer configured — classification uses `model_low` (falling back to `model`).

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
- `strategies.yaml`: strategy definitions (optional per-strategy model and
  complexity gate).
- `prompts/*.md`: one prompt per strategy (`direct`/`teaching`/`debugging`/
  `analysis`/`code_snippet`), plus `unsupported_complex.md` (still loaded for
  backward compatibility, but complex tasks now run through the Orchestrator).

Complex questions on a gated strategy run through an Orchestrator pipeline
(Planner → Workers → Aggregator) that builds on the strategy prompt and uses the
`model_high` tier.

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
