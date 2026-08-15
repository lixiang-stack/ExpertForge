import pytest

from agent.evaluation.dataset import (
    COMPLEXITY_LEVELS,
    DatasetError,
    EvalCase,
    Suite,
    is_in_domain,
    load_suites,
)


def _dataset_path(tmp_path, yaml_text):
    d = tmp_path / "software_engineering"
    d.mkdir(exist_ok=True)
    path = d / "se.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return str(path)


_VALID = """
domain: software_engineering
cases:
  - id: se-001
    question: "What is dependency injection?"
    expected:
      domain: software_engineering
      intent: concept_explain
      complexity: simple
      strategy: teaching
      orchestrate: false
    answer_quality: true
    reference: "Dependency injection passes dependencies into a component."
  - id: se-002
    question: "Recommend a restaurant in Tokyo."
    expected:
      domain: other
      intent: null
      complexity: null
      strategy: reject
      orchestrate: false
"""


def test_load_suites_valid(tmp_path):
    suites = load_suites(_dataset_path(tmp_path, _VALID))
    assert len(suites) == 1
    s = suites[0]
    assert s.name == "se"
    assert s.domain == "software_engineering"
    assert len(s.cases) == 2
    c = s.cases[0]
    assert c.id == "se-001"
    assert c.question == "What is dependency injection?"
    assert c.expected_domain == "software_engineering"
    assert c.expected_intent == "concept_explain"
    assert c.expected_complexity == "simple"
    assert c.expected_strategy == "teaching"
    assert c.expected_orchestrate is False
    assert c.answer_quality is True
    assert c.reference == "Dependency injection passes dependencies into a component."


def test_load_suites_out_of_domain_case_fields(tmp_path):
    suites = load_suites(_dataset_path(tmp_path, _VALID))
    s = suites[0]
    c = s.cases[1]
    assert c.expected_domain == "other"
    assert c.expected_intent is None
    assert c.expected_complexity is None
    assert c.expected_strategy == "reject"
    assert is_in_domain(c, s) is False
    assert is_in_domain(s.cases[0], s) is True


def test_load_suites_answer_quality_defaults_true(tmp_path):
    path = _dataset_path(tmp_path,
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n')
    s = load_suites(str(path))[0]
    assert s.cases[0].answer_quality is True
    assert s.cases[0].reference is None
    assert s.cases[0].expected_orchestrate is False


def test_load_suites_missing_file():
    with pytest.raises(DatasetError):
        load_suites("/nonexistent/se.yaml")


def test_load_suites_bad_yaml(tmp_path):
    with pytest.raises(DatasetError):
        load_suites(_dataset_path(tmp_path, ":: not: [valid"))


def test_load_suites_missing_cases(tmp_path):
    with pytest.raises(DatasetError):
        load_suites(_dataset_path(tmp_path, "cases: not_a_list\n"))


def test_load_suites_invalid_complexity(tmp_path):
    with pytest.raises(DatasetError):
        load_suites(_dataset_path(tmp_path,
            'domain: software_engineering\n'
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: faq\n'
            '      complexity: huge\n'
            '      strategy: direct\n'))


def test_load_committed_software_engineering_suites():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "evaluation" / "datasets" / "software_engineering"
    suites = load_suites(str(path))
    names = [s.name for s in suites]
    assert names == ["analysis", "classification", "code_snippet", "debugging",
                     "direct", "orchestration", "routing", "teaching"]
    assert all(s.domain == "software_engineering" for s in suites)
    all_cases = [c for s in suites for c in s.cases]
    assert len(all_cases) >= 20
    ids = [c.id for c in all_cases]
    assert len(ids) == len(set(ids))  # no cross-suite duplication
    intents = {c.expected_intent for c in all_cases}
    assert {"faq", "concept_explain", "tutorial", "learning_guide", "summarization",
            "troubleshooting", "performance_analysis", "comparison", "architecture_design",
            "code_review", "generate_code"} <= intents
    strategies = {c.expected_strategy for c in all_cases}
    assert {"direct", "teaching", "debugging", "analysis", "code_snippet"} <= strategies
    assert {"simple", "medium", "complex"} <= {c.expected_complexity for c in all_cases}
    assert any(c.expected_orchestrate for c in all_cases)
    assert any(c.expected_domain == "other" for c in all_cases)


def _suite_dir(tmp_path, name="software_engineering"):
    d = tmp_path / name
    d.mkdir()
    (d / "direct.yaml").write_text(
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    (d / "teaching.yaml").write_text(
        'cases:\n'
        '  - id: b\n'
        '    question: "q2"\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: concept_explain\n'
        '      complexity: medium\n'
        '      strategy: teaching\n',
        encoding="utf-8",
    )
    return d


def test_load_suites_directory(tmp_path):
    d = _suite_dir(tmp_path)
    suites = load_suites(str(d))
    assert len(suites) == 2
    assert suites[0].name == "direct"
    assert suites[1].name == "teaching"
    assert suites[0].domain == "software_engineering"
    assert suites[1].domain == "software_engineering"
    assert len(suites[0].cases) == 1
    assert suites[0].cases[0].id == "a"


def test_load_suites_empty_directory(tmp_path):
    d = tmp_path / "software_engineering"
    d.mkdir()
    with pytest.raises(DatasetError):
        load_suites(str(d))


def test_load_suites_directory_missing_cases(tmp_path):
    d = _suite_dir(tmp_path)
    (d / "bad.yaml").write_text("not_a_mapping: true\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_suites(str(d))


def test_load_suites_single_file(tmp_path):
    d = _suite_dir(tmp_path)
    p = d / "direct.yaml"
    suites = load_suites(str(p))
    assert len(suites) == 1
    assert suites[0].name == "direct"
    assert suites[0].domain == "software_engineering"