import json

from agent.chat import Chat
from agent.config import AgentConfig, load_domain_config


def _write_finance_domain(tmp_path):
    base = tmp_path / "finance"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "Finance Advice",
        "description": "Personal finance, investment, and risk guidance.",
        "out_of_domain_reply": "Out of finance domain.",
    }), encoding="utf-8")
    (base / "intents.yaml").write_text(
        "- id: portfolio_review\n  description: review an investment portfolio\n"
        "- id: risk_check\n  description: assess financial risk\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text(
        "portfolio_review: advise\nrisk_check: risk_assessment\n", encoding="utf-8"
    )
    (base / "strategies.yaml").write_text(
        "advise:\n  default: true\n  complexity_gate: true\n"
        "risk_assessment:\n  complexity_gate: true\n",
        encoding="utf-8",
    )
    (base / "prompts" / "advise.md").write_text(
        "You are a finance advisor in the Finance Advice domain.\n\n"
        "Personal finance, investment, and risk guidance.\n\n"
        "Structure:\n- Summary\n- Options\n- Recommendation\n- Risks\n",
        encoding="utf-8",
    )
    (base / "prompts" / "risk_assessment.md").write_text(
        "You are a risk assessor in the Finance Advice domain.\n\n"
        "Personal finance, investment, and risk guidance.\n\n"
        "Structure:\n- Risk factors\n- Likelihood\n- Mitigations\n",
        encoding="utf-8",
    )
    (base / "prompts" / "unsupported_complex.md").write_text("unsupported", encoding="utf-8")
    return str(base)


def _config(domain_dir):
    return AgentConfig(base_url="https://x", model="m", classifier_model="m", domain_dir=domain_dir)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_custom_strategy_answers_without_code_changes(tmp_path):
    domain = load_domain_config(_write_finance_domain(tmp_path))
    assert domain.default_strategy == "advise"
    client = FakeClient([
        '{"in_domain": true, "intent": "portfolio_review", "complexity": "simple", "reason": "ok"}',
        "the finance advice",
    ])
    chat = Chat(client, _config(domain_dir=str(tmp_path / "finance")), domain)
    resp = chat.respond("Should I diversify into bonds?")
    assert resp.kind == "answer"
    assert resp.text == "the finance advice"


def test_custom_strategy_orchestrates_complex(tmp_path):
    domain = load_domain_config(_write_finance_domain(tmp_path))
    client = FakeClient([
        '{"in_domain": true, "intent": "risk_check", "complexity": "complex", "reason": "ok"}',
        '{"tasks": [{"title": "r1", "instruction": "identify risks"}]}',
        "risk worker output",
        "final risk answer",
    ])
    chat = Chat(client, _config(domain_dir=str(tmp_path / "finance")), domain)
    resp = chat.respond("Assess my retirement risk profile")
    assert resp.kind == "answer"
    assert resp.text == "final risk answer"