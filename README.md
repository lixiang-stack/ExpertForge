# ExpertForge

A single-domain configurable AI expert agent CLI. You define an expert domain in a
config file, and the agent classifies each question as in-domain or out-of-domain,
then answers in-domain questions via an OpenAI-compatible API.

## Install

```bash
uv sync
```

## Configure

Copy the example config and fill in your API endpoint, model, and domain:

```bash
cp config.example.json config.json
```

Edit `config.json`:

- `base_url`: the OpenAI-compatible API base URL (e.g. `https://api.example.com/v1`).
- `model`: the model to use for answers.
- `domain.description`: a description of your expert domain (used for classification).

Then set your API key:

```bash
export AGENT_API_KEY=your_key
```

`base_url` can also be set via the `AGENT_BASE_URL` environment variable.

## Run

```bash
uv run python -m agent
```

To use a specific config file:

```bash
uv run python -m agent path/to/config.json
```

Type your question at the `you >` prompt. To leave the REPL, type `exit` (or `quit`),
or press `Ctrl-C` / `Ctrl-D`.
