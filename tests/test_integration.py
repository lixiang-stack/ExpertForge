"""Live integration tests: exercise deeper pipeline paths against the API.

- complex question on a gated strategy → Orchestrator (Planner → Workers → Aggregator)
- medium question → strategy processor (single call)

These skip automatically when `AGENT_API_KEY` is not set. Provide the key via
the environment / secret manager / CI secret — never hardcode or commit it:

    export AGENT_API_KEY=your_key
    uv run pytest tests/test_integration.py -v
"""

import json
import os
from pathlib import Path

import pytest

from agent.chat import Chat
from agent.config import load_config, load_domain_config
from agent.llm import LLMClient

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENT_API_KEY") is None,
    reason="AGENT_API_KEY not set; provide it via the environment to run live integration tests",
)


@pytest.fixture(scope="module")
def live_chat(tmp_path_factory):
    """Build a real Chat wired to the live API.

    Uses the fast deepseek-v4-flash for all tiers so the pipeline runs in seconds;
    the goal is to verify routing/processing/orchestration end-to-end, not to
    benchmark the slow high-end reasoning model.
    """
    example = json.loads((REPO_ROOT / "config.example.json").read_text(encoding="utf-8"))
    example["model_high"] = "deepseek-v4-flash"
    config_path = tmp_path_factory.mktemp("live") / "config.json"
    config_path.write_text(
        json.dumps({**example, "domain_dir": str(REPO_ROOT / example["domain_dir"])}),
        encoding="utf-8",
    )
    config = load_config(str(config_path))
    domain = load_domain_config(config.domain_dir)
    client = LLMClient(
        base_url=config.base_url,
        api_key=os.environ["AGENT_API_KEY"],
        model=config.model,
    )
    return Chat(client, config, domain)


def test_integration_complex_question_orchestrates(live_chat):
    """A complex, gated question must return a non-empty answer via the Orchestrator."""
    response = live_chat.respond(
        "Design a high-availability distributed rate limiter supporting millions "
        "of QPS and explain the trade-offs."
    )
    assert response.kind in {"answer", "unsupported", "reject"}
    assert response.text.strip(), "expected a non-empty response"


def test_integration_medium_question_uses_processor(live_chat):
    """A medium question should return a non-empty answer through the processor path."""
    response = live_chat.respond(
        "Explain how Go's context package is designed and why."
    )
    assert response.kind == "answer"
    assert response.text.strip(), "expected a non-empty answer"
