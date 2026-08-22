from __future__ import annotations

import json
from pathlib import Path

import yaml

from .config import (
    COMPLEXITY_LEVELS,
    ComplexityLevelDef,
    ComplexityPolicy,
    ConfigError,
    DomainConfig,
    EvaluatorPolicy,
    IntentDef,
    OrchestrationPolicy,
)

DOMAIN_FILE_CONTRACT = (
    "A domain directory must contain: 'domain.json' (name, description, "
    "out_of_domain_reply), 'intents.yaml' (list of {id, description, "
    "positive_examples, negative_examples, boundaries}), 'orchestration.yaml' "
    "(enabled, min_complexity, intents, max_workers, topology, evaluator), "
    "'intent_mapping.yaml' (intent id -> strategy id), 'prompts/*.md' (one "
    "file per strategy), optional 'complexity.yaml' (list of simple|medium|"
    "complex levels) and optional 'expert_policy.md'."
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Domain config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid domain config JSON: {path}: {e}")
    if not isinstance(data, dict):
        raise ConfigError(f"Domain config must be a JSON object: {path}")
    return data


def _read_yaml(path: Path) -> object:
    if not path.is_file():
        raise ConfigError(f"Domain config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid domain config YAML: {path}: {e}")


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_meta(base: Path) -> tuple[str, str, str]:
    meta = _read_json(base / "domain.json")
    name = meta.get("name") or ""
    description = meta.get("description")
    if not description:
        raise ConfigError(f"Missing 'description' in {base / 'domain.json'}")
    out_of_domain_reply = meta.get("out_of_domain_reply") or (
        f"This question falls outside my expert domain ({name}) "
        "and I cannot provide a professional answer."
    )
    return name, description, out_of_domain_reply


def _parse_intents(base: Path) -> dict[str, IntentDef]:
    intents: dict[str, IntentDef] = {}
    intents_data = _read_yaml(base / "intents.yaml")
    if intents_data is None:
        intents_data = []
    if not isinstance(intents_data, list):
        raise ConfigError(f"intents.yaml must contain a list: {base / 'intents.yaml'}")
    for item in intents_data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ConfigError(f"Invalid intent entry in {base / 'intents.yaml'}: {item}")
        iid = item["id"]
        intents[iid] = IntentDef(
            id=iid,
            description=item.get("description") or "",
            positive_examples=_str_list(item.get("positive_examples")),
            negative_examples=_str_list(item.get("negative_examples")),
            boundaries=_str_list(item.get("boundaries")),
        )
    return intents


def _parse_orchestration(base: Path, intents: dict[str, IntentDef]) -> OrchestrationPolicy:
    orch_path = base / "orchestration.yaml"
    if not orch_path.is_file():
        raise ConfigError(f"orchestration.yaml not found: {orch_path}")
    orch_data = _read_yaml(orch_path)
    if not isinstance(orch_data, dict):
        raise ConfigError(f"orchestration.yaml must contain a mapping: {orch_path}")
    orch_intents = orch_data.get("intents")
    if not isinstance(orch_intents, list) or not orch_intents:
        raise ConfigError(f"orchestration.yaml 'intents' must be a non-empty list: {orch_path}")
    if not all(isinstance(i, str) and i in intents for i in orch_intents):
        raise ConfigError(f"orchestration.yaml 'intents' references unknown intent: {orch_path}")
    min_complexity = orch_data.get("min_complexity", "complex")
    if min_complexity not in COMPLEXITY_LEVELS:
        raise ConfigError(f"Unknown 'min_complexity' {min_complexity!r} in {orch_path}")
    max_workers = orch_data.get("max_workers", 4)
    if not isinstance(max_workers, int) or max_workers <= 0:
        raise ConfigError(f"orchestration.yaml 'max_workers' must be a positive int: {orch_path}")
    topology = orch_data.get("topology", "map_reduce")
    if topology not in ("map_reduce", "critique"):
        raise ConfigError(
            f"orchestration.yaml 'topology' must be 'map_reduce' or 'critique': {orch_path}"
        )
    ev = orch_data.get("evaluator") or {}
    if not isinstance(ev, dict):
        raise ConfigError(f"orchestration.yaml 'evaluator' must be a mapping: {orch_path}")
    min_score = ev.get("min_dimension_score", 3)
    max_rounds = ev.get("max_rounds", 1)
    if not isinstance(min_score, int) or not 1 <= min_score <= 5:
        raise ConfigError(f"orchestration.yaml 'min_dimension_score' must be an int in 1..5: {orch_path}")
    if not isinstance(max_rounds, int) or max_rounds < 0:
        raise ConfigError(f"orchestration.yaml 'max_rounds' must be a non-negative int: {orch_path}")
    return OrchestrationPolicy(
        enabled=bool(orch_data.get("enabled", True)),
        min_complexity=min_complexity,
        intents=orch_intents,
        max_workers=max_workers,
        topology=topology,
        evaluator=EvaluatorPolicy(
            enabled=bool(ev.get("enabled", True)),
            min_dimension_score=min_score,
            max_rounds=max_rounds,
        ),
    )


def _parse_complexity(base: Path) -> ComplexityPolicy | None:
    complexity = None
    complexity_path = base / "complexity.yaml"
    if complexity_path.is_file():
        complexity_data = _read_yaml(complexity_path)
        if not isinstance(complexity_data, list):
            raise ConfigError(f"complexity.yaml must contain a list: {complexity_path}")
        levels: list[ComplexityLevelDef] = []
        for item in complexity_data:
            if not isinstance(item, dict) or not isinstance(item.get("level"), str):
                raise ConfigError(
                    f"Invalid complexity level entry in {complexity_path}: {item}"
                )
            if item["level"] not in COMPLEXITY_LEVELS:
                raise ConfigError(
                    f"Unknown complexity level {item['level']!r} in {complexity_path}"
                )
            levels.append(ComplexityLevelDef(
                level=item["level"],
                description=item.get("description") or "",
                dimensions=_str_list(item.get("dimensions")),
                positive_examples=_str_list(item.get("positive_examples")),
                negative_examples=_str_list(item.get("negative_examples")),
                boundaries=_str_list(item.get("boundaries")),
            ))
        if [l.level for l in levels] != list(COMPLEXITY_LEVELS):
            raise ConfigError(
                f"complexity.yaml must define each level exactly once in order "
                f"simple, medium, complex: {complexity_path}"
            )
        complexity = ComplexityPolicy(levels=levels)
    return complexity


def _parse_intent_mapping(base: Path, intents: dict[str, IntentDef]) -> dict[str, str]:
    mapping_data = _read_yaml(base / "intent_mapping.yaml")
    if mapping_data is None:
        mapping_data = {}
    if not isinstance(mapping_data, dict):
        raise ConfigError(
            f"intent_mapping.yaml must contain a mapping: {base / 'intent_mapping.yaml'}"
        )
    intent_mapping: dict[str, str] = {}
    for intent_id, strategy_id in mapping_data.items():
        if not isinstance(strategy_id, str):
            raise ConfigError(f"Invalid mapping for intent '{intent_id}'")
        if intent_id not in intents:
            raise ConfigError(
                f"Mapping references unknown intent '{intent_id}' in {base / 'intent_mapping.yaml'}"
            )
        intent_mapping[intent_id] = strategy_id

    for intent_id in intents:
        if intent_id not in intent_mapping:
            raise ConfigError(
                f"intent_mapping.yaml is missing a strategy for intent '{intent_id}'"
            )
    return intent_mapping


def _parse_prompts(base: Path, intent_mapping: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    prompt_dir = base / "prompts"
    strategies = sorted(p.stem for p in prompt_dir.glob("*.md"))
    if not strategies:
        raise ConfigError(f"No strategy prompt files found in {prompt_dir}")
    prompts: dict[str, str] = {}
    for sid in strategies:
        prompts[sid] = _read_prompt(prompt_dir / f"{sid}.md")
    for intent_id, strategy_id in intent_mapping.items():
        if strategy_id not in strategies:
            raise ConfigError(
                f"Mapping for intent '{intent_id}' references unknown strategy "
                f"'{strategy_id}': no {strategy_id}.md in {prompt_dir}"
            )
    return strategies, prompts


def _parse_expert_policy(base: Path) -> str:
    expert_policy = ""
    expert_policy_path = base / "expert_policy.md"
    if expert_policy_path.is_file():
        expert_policy = expert_policy_path.read_text(encoding="utf-8")
    return expert_policy


def load_domain_config(domain_dir: str) -> DomainConfig:
    base = Path(domain_dir)
    name, description, out_of_domain_reply = _parse_meta(base)
    intents = _parse_intents(base)
    orchestration = _parse_orchestration(base, intents)
    complexity = _parse_complexity(base)
    intent_mapping = _parse_intent_mapping(base, intents)
    strategies, prompts = _parse_prompts(base, intent_mapping)
    expert_policy = _parse_expert_policy(base)
    return DomainConfig(
        name=name,
        description=description,
        out_of_domain_reply=out_of_domain_reply,
        intents=intents,
        intent_mapping=intent_mapping,
        strategies=strategies,
        prompts=prompts,
        complexity=complexity,
        expert_policy=expert_policy,
        orchestration=orchestration,
    )
