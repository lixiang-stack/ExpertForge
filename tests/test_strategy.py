from agent.config import DomainConfig, IntentDef, StrategyDef
from agent.strategy import Strategy, build_registry


def _prompts():
    return {
        "direct": "Direct answer prompt.",
        "teaching": "Teaching prompt.",
    }


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out.",
        intents={},
        intent_mapping={},
        strategies={"direct": StrategyDef("direct", default=True),
                    "teaching": StrategyDef("teaching", complexity_gate=True)},
        default_strategy="direct",
        prompts=_prompts(),
    )


class FakeClient:
    def __init__(self, text="answer"):
        self.text = text
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False):
        self.calls.append((messages, model))
        return self.text


def test_build_registry_builds_each_strategy():
    registry = build_registry(_domain())
    assert set(registry) == {"direct", "teaching"}
    assert isinstance(registry["teaching"], Strategy)
    assert registry["direct"].build_system_prompt() == "Direct answer prompt."


def test_build_system_prompt_returns_template_verbatim():
    p = Strategy("direct", "You are an agent in the X domain.\n- Approach\n- Code snippet")
    prompt = p.build_system_prompt()
    assert prompt == "You are an agent in the X domain.\n- Approach\n- Code snippet"
    assert "{name}" not in prompt
    assert "{description}" not in prompt
    assert "{structure}" not in prompt


def test_process_single_call_returns_string():
    client = FakeClient("answer")
    p = Strategy("direct", "You are an agent in the X domain.")
    out = p.process(client, "q", [("旧问", "旧答")])
    assert out == "answer"
    assert len(client.calls) == 1
    messages, model = client.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "旧问"}
    assert messages[2] == {"role": "assistant", "content": "旧答"}
    assert messages[-1]["content"] == "q"
