from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = ("knowledge", "problem_solving", "evaluation", "generation", "boundary")
COMPLEXITY_LEVELS = ("simple", "medium", "complex")
OUT_OF_DOMAIN = "other"
REJECT_STRATEGY = "reject"


class DatasetError(Exception):
    """Raised when a golden dataset is missing or invalid."""


@dataclass
class EvalCase:
    id: str
    question: str
    category: str
    expected_domain: str
    expected_intent: str | None
    expected_complexity: str | None
    expected_strategy: str
    expected_orchestrate: bool
    answer_quality: bool
    reference: str | None


@dataclass
class Dataset:
    domain: str
    cases: list[EvalCase]


def is_in_domain(case: EvalCase, dataset: Dataset) -> bool:
    return case.expected_domain == dataset.domain


def _read_yaml(path: Path) -> object:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise DatasetError(f"Invalid dataset YAML: {path}: {e}")


def _validate_case(raw: object, dataset_domain: str) -> EvalCase:
    if not isinstance(raw, dict):
        raise DatasetError(f"Dataset case must be a mapping, got: {raw!r}")
    cid = raw.get("id")
    question = raw.get("question")
    category = raw.get("category")
    if not isinstance(cid, str) or not cid:
        raise DatasetError(f"Case missing string 'id': {raw!r}")
    if not isinstance(question, str) or not question:
        raise DatasetError(f"Case {cid} missing string 'question'")
    if category not in CATEGORIES:
        raise DatasetError(f"Case {cid} has unknown category {category!r}")
    expected = raw.get("expected")
    if not isinstance(expected, dict):
        raise DatasetError(f"Case {cid} missing 'expected' mapping")
    exp_domain = expected.get("domain")
    if not isinstance(exp_domain, str) or not exp_domain:
        raise DatasetError(f"Case {cid} missing expected.domain")
    if exp_domain not in (dataset_domain, OUT_OF_DOMAIN):
        raise DatasetError(f"Case {cid} expected.domain {exp_domain!r} must be "
                           f"{dataset_domain!r} or {OUT_OF_DOMAIN!r}")
    in_domain = exp_domain == dataset_domain
    intent = expected.get("intent")
    complexity = expected.get("complexity")
    strategy = expected.get("strategy")
    orchestrate = expected.get("orchestrate", False)
    if not isinstance(strategy, str) or not strategy:
        raise DatasetError(f"Case {cid} missing expected.strategy")
    if orchestrate not in (True, False):
        raise DatasetError(f"Case {cid} expected.orchestrate must be a boolean")
    if in_domain:
        if not isinstance(intent, str) or not intent:
            raise DatasetError(f"In-domain case {cid} missing expected.intent")
        if complexity not in COMPLEXITY_LEVELS:
            raise DatasetError(f"Case {cid} invalid complexity {complexity!r}")
    else:
        intent = None
        complexity = None
        if strategy != REJECT_STRATEGY:
            raise DatasetError(f"Out-of-domain case {cid} expected.strategy must be "
                               f"{REJECT_STRATEGY!r}")
        if orchestrate is not False:
            raise DatasetError(f"Out-of-domain case {cid} must have orchestrate: false")
    answer_quality = raw.get("answer_quality", True)
    if answer_quality not in (True, False):
        raise DatasetError(f"Case {cid} answer_quality must be a boolean")
    reference = raw.get("reference")
    return EvalCase(
        id=cid,
        question=question,
        category=category,
        expected_domain=exp_domain,
        expected_intent=intent,
        expected_complexity=complexity,
        expected_strategy=strategy,
        expected_orchestrate=bool(orchestrate),
        answer_quality=bool(answer_quality),
        reference=reference if isinstance(reference, str) else None,
    )


def load_dataset(path: str) -> Dataset:
    raw = _read_yaml(Path(path))
    if not isinstance(raw, dict):
        raise DatasetError(f"Dataset top-level must be a mapping: {path}")
    domain = raw.get("domain")
    if not isinstance(domain, str) or not domain:
        raise DatasetError(f"Dataset missing string 'domain': {path}")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise DatasetError(f"Dataset 'cases' must be a list: {path}")
    return Dataset(domain=domain, cases=[_validate_case(c, domain) for c in cases_raw])
