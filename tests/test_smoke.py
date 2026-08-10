"""Live smoke tests: exercise the real agent end-to-end against the API.

These tests skip automatically when `AGENT_API_KEY` is not set. To run them,
provide the key through the environment (or your secret manager / CI secret) —
never hardcode or commit it:

    export AGENT_API_KEY=your_key
    uv run pytest tests/test_smoke.py -v
"""

import json
import os
from pathlib import Path

import pytest

from agent import agent_cli

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENT_API_KEY") is None,
    reason="AGENT_API_KEY not set; provide it via the environment to run live smoke tests",
)


@pytest.fixture(scope="module")
def live_config(tmp_path_factory):
    """Write a real config.json (base_url/model/domain from config.example.json)."""
    example = json.loads((REPO_ROOT / "config.example.json").read_text(encoding="utf-8"))
    config_path = tmp_path_factory.mktemp("live") / "config.json"
    config_path.write_text(
        json.dumps({**example, "domain_dir": str(REPO_ROOT / example["domain_dir"])}),
        encoding="utf-8",
    )
    return str(config_path)


def test_smoke_single_question_returns_answer(live_config, capsys):
    assert agent_cli.main([live_config, "--ask", "What is Go defer?"]) == 0
    out = capsys.readouterr().out
    assert out.strip(), "expected a non-empty answer"


def test_smoke_out_of_domain_rejects(live_config, capsys):
    assert agent_cli.main([live_config, "--ask", "Recommend a restaurant in San Francisco."]) == 0
    out = capsys.readouterr().out
    assert out.strip(), "expected a non-empty out-of-domain reply"
