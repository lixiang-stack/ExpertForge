from agent.config import DomainConfig, IntentDef, StrategyDef
from agent.processors.analysis import AnalysisProcessor
from agent.processors.code_snippet import CodeSnippetProcessor
from agent.processors.debugging import DebuggingProcessor
from agent.processors.direct import DirectAnswerProcessor
from agent.processors.registry import build_registry
from agent.processors.teaching import TeachingProcessor


def _prompts():
    return {
        "direct": "Direct {name} {description} {structure}",
        "teaching": "Teach {name} {description} {structure}",
        "debugging": "Debug {name} {description} {structure}",
        "analysis": "Analyze {name} {description} {structure}",
        "code_snippet": "Code {name} {description} {structure}",
    }


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out.",
        intents={},
        intent_mapping={},
        strategies={},
        prompts=_prompts(),
    )


class FakeClient:
    def __init__(self, text="answer"):
        self.text = text
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False):
        self.calls.append((messages, model))
        return self.text


def test_direct_structure_empty():
    p = DirectAnswerProcessor("X {structure}", "软件工程", "sw")
    assert "{structure}" not in p.build_system_prompt()
    assert "Concept" not in p.build_system_prompt()


def test_teaching_structure():
    p = TeachingProcessor("X {structure}", "软件工程", "sw")
    prompt = p.build_system_prompt()
    assert "Concept" in prompt
    assert "Common misconceptions" in prompt


def test_debugging_structure():
    p = DebuggingProcessor("X {structure}", "软件工程", "sw")
    assert "Possible causes" in p.build_system_prompt()


def test_analysis_structure():
    p = AnalysisProcessor("X {structure}", "软件工程", "sw")
    assert "Trade-offs" in p.build_system_prompt()


def test_code_snippet_structure():
    p = CodeSnippetProcessor("X {structure}", "软件工程", "sw")
    assert "Approach" in p.build_system_prompt()


def test_process_single_call_returns_string():
    client = FakeClient("answer")
    p = DirectAnswerProcessor("X {structure}", "软件工程", "sw")
    out = p.process(client, "q", [("旧问", "旧答")])
    assert out == "answer"
    assert len(client.calls) == 1
    messages, model = client.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "旧问"}
    assert messages[2] == {"role": "assistant", "content": "旧答"}
    assert messages[-1]["content"] == "q"


def test_build_registry():
    registry = build_registry(_domain())
    assert set(registry) == {"direct", "teaching", "debugging", "analysis", "code_snippet"}
    assert isinstance(registry["teaching"], TeachingProcessor)
    assert registry["direct"].build_system_prompt() == "Direct 软件工程 sw "
