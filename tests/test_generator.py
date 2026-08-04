from agent.generator import build_messages, build_system_prompt


def test_build_system_prompt_contains_domain():
    prompt = build_system_prompt("软件工程", "software engineering")
    assert "软件工程" in prompt
    assert "software engineering" in prompt
    assert "multiple angles" in prompt


def test_build_messages_structure():
    messages = build_messages("sys", [("问1", "答1")], "问2")
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "答1"},
        {"role": "user", "content": "问2"},
    ]


def test_build_messages_truncates_history():
    history = [(f"q{i}", f"a{i}") for i in range(30)]
    messages = build_messages("sys", history, "final", max_turns=5)
    assert len(messages) == 1 + 10 + 1
    assert messages[1]["content"] == "q25"
    assert messages[-1]["content"] == "final"
