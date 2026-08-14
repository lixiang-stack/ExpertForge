import pytest

from agent.evaluation.dataset import (
    COMPLEXITY_LEVELS,
    Dataset,
    DatasetError,
    EvalCase,
    load_dataset,
)


def _dataset_path(tmp_path, yaml_text):
    path = tmp_path / "se.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return str(path)


_VALID = """
domain: software_engineering
cases:
  - id: se-001
    question: "What is dependency injection?"
    category: knowledge
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
    category: boundary
    expected:
      domain: other
      intent: null
      complexity: null
      strategy: reject
      orchestrate: false
"""


def test_load_dataset_valid(tmp_path):
    ds = load_dataset(_dataset_path(tmp_path, _VALID))
    assert isinstance(ds, Dataset)
    assert ds.domain == "software_engineering"
    assert len(ds.cases) == 2
    c = ds.cases[0]
    assert c.id == "se-001"
    assert c.question == "What is dependency injection?"
    assert c.category == "knowledge"
    assert c.expected_domain == "software_engineering"
    assert c.expected_intent == "concept_explain"
    assert c.expected_complexity == "simple"
    assert c.expected_strategy == "teaching"
    assert c.expected_orchestrate is False
    assert c.answer_quality is True
    assert c.reference == "Dependency injection passes dependencies into a component."


def test_out_of_domain_case_fields():
    from agent.evaluation.dataset import is_in_domain

    import tempfile
    path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    with open(path, "w", encoding="utf-8") as f:
        f.write(_VALID)
    ds = load_dataset(path)
    c = ds.cases[1]
    assert c.expected_domain == "other"
    assert c.expected_intent is None
    assert c.expected_complexity is None
    assert c.expected_strategy == "reject"
    assert is_in_domain(c, ds) is False
    assert is_in_domain(ds.cases[0], ds) is True


def test_load_dataset_answer_quality_defaults_true(tmp_path):
    path = tmp_path / "se.yaml"
    path.write_text(
        'domain: software_engineering\n'
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    category: knowledge\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    ds = load_dataset(str(path))
    assert ds.cases[0].answer_quality is True
    assert ds.cases[0].reference is None
    assert ds.cases[0].expected_orchestrate is False


def test_load_dataset_missing_file():
    with pytest.raises(DatasetError):
        load_dataset("/nonexistent/se.yaml")


def test_load_dataset_bad_yaml(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path, ":: not: [valid"))


def test_load_dataset_missing_domain(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path, "cases: []\n"))


def test_load_dataset_missing_cases(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path, "domain: se\n"))


def test_load_dataset_invalid_complexity(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path,
            'domain: software_engineering\n'
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    category: knowledge\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: faq\n'
            '      complexity: huge\n'
            '      strategy: direct\n'))


def test_load_dataset_unknown_category(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path,
            'domain: software_engineering\n'
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    category: weird\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: faq\n'
            '      complexity: simple\n'
            '      strategy: direct\n'))


def test_load_committed_software_engineering_dataset():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "evaluation" / "datasets" / "software_engineering.yaml"
    ds = load_dataset(str(path))
    assert ds.domain == "software_engineering"
    assert len(ds.cases) >= 40
    categories = {c.category for c in ds.cases}
    assert {"knowledge", "problem_solving", "evaluation", "generation", "boundary"} <= categories
    intents = {c.expected_intent for c in ds.cases}
    assert {"faq", "concept_explain", "tutorial", "learning_guide", "summarization",
            "troubleshooting", "performance_analysis", "comparison", "architecture_design",
            "code_review", "generate_code"} <= intents
    strategies = {c.expected_strategy for c in ds.cases}
    assert {"direct", "teaching", "debugging", "analysis", "code_snippet"} <= strategies
    assert {"simple", "medium", "complex"} <= {c.expected_complexity for c in ds.cases}
    assert any(c.expected_orchestrate for c in ds.cases)
    assert any(c.expected_domain == "other" for c in ds.cases)
