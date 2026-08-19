"""Live smoke tests: exercise the real agent end-to-end against the API.

These tests skip automatically when `AGENT_API_KEY` is not set. To run them,
provide the key through the environment (or your secret manager / CI secret) —
never hardcode or commit it:

    export AGENT_API_KEY=your_key
    uv run pytest tests/live -v
"""

import json
import os
from pathlib import Path

import pytest

from agent import agent_cli
from tests.helpers import absolutize_domain_dir, resolve_live_config_src

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENT_API_KEY") is None,
    reason="AGENT_API_KEY not set; provide it via the environment to run live smoke tests",
)


@pytest.fixture(scope="module")
def live_config(tmp_path_factory):
    """Write a real config.json, preferring the user's config.json over the example."""
    config_path = tmp_path_factory.mktemp("live") / "config.json"
    config_path.write_text(
        json.dumps(absolutize_domain_dir(resolve_live_config_src(REPO_ROOT), REPO_ROOT)),
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


def test_smoke_evaluation_writes_result(live_config, tmp_path, monkeypatch):
    """A tiny dataset slice runs end-to-end and produces a result file."""
    import agent.evaluation.__main__ as eval_main

    dataset_dir = tmp_path / "software_engineering"
    dataset_dir.mkdir()
    dataset = dataset_dir / "smoke.yaml"
    dataset.write_text(
        'cases:\n'
        '  - id: smoke-1\n'
        '    question: "What is Go defer?"\n'
        '    tier: classification\n'
        '    smoke: true\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    monkeypatch.setenv("AGENT_API_KEY", os.environ["AGENT_API_KEY"])
    rc = eval_main.main(["run", "--config", live_config,
                         "--dataset", str(dataset_dir),
                         "--label", "smoke",
                         "--results-dir", str(results_dir)])
    assert rc == 0
    files = list(results_dir.glob("*-smoke.json"))
    assert files, "expected a result file to be written"
    assert json.loads(files[0].read_text(encoding="utf-8"))["metrics"]["n_cases"] == 1
