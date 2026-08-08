# ExpertForge 策略路由实现计划（Strategy Routing）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ExpertForge 从"单领域直接回答"升级为"分类 + 执行策略"框架：领域配置目录化（yaml + markdown），Domain/Intent/Complexity 分类，intent→strategy 映射，五个一次性生成的策略处理器，Clarification 前置阶段，Chat Interface + `--ask` 单次入口。

**Architecture:** `config.json` 只保留连接信息与 `domain_dir` 指针；领域内容拆到独立目录。交互层（`repl.py`/`agent_cli.py`）驱动 `Chat` 引擎；`Chat` 调 `Router.route`（Domain → Intent → Complexity → 映射 → 复杂度 gate → 澄清标志），再路由到 `processors` 里的策略处理器。所有处理器一次性生成完整字符串（非流式），Clarification 是路由前置阶段，单次确认。

**Tech Stack:** Python 3.10+，`openai>=1.0` + 新增 `pyyaml>=6.0`；测试用 `pytest`；环境由 `uv` 管理。

## Global Constraints

- Python >= 3.10，使用 `from __future__ import annotations`。
- 运行时依赖：`openai>=1.0`、`pyyaml>=6.0`；开发依赖：`pytest>=8.0`。
- 发送给 LLM 的 prompt、通用错误与异常信息一律使用英文；面向用户的界面文案（欢迎语、澄清提示）可用中文。
- 配置键名固定：根 `config.json` 为 `base_url`、`model`、`classifier_model`（null 回退到 `model`）、`domain_dir`；领域目录内为 `domain.json`（`name`/`description`/`out_of_domain_reply`）、`intents.yaml`（`id`/`description`/`needs_clarification`）、`intent_mapping.yaml`、`strategies.yaml`（`model` 可选、`complexity_gate`）、`prompts/<id>.md`（含 `{name}`、`{description}`、`{structure}` 占位符）+ `prompts/clarify.md` + `prompts/unsupported_complex.md`。
- 环境变量：`AGENT_API_KEY`（必需）、`AGENT_BASE_URL`（可选，覆盖 `base_url`）。
- 分类调用：Domain/Intent/Complexity 各一次 `chat_completion`（`disable_thinking=True`）；解析失败用更严格 prompt 重试一次；仍失败降级：Domain → `in_domain=false` 拒绝、Intent → 空 id（路由回退 `direct`）、Complexity → `medium`。
- 复杂度 gate：strategy 配置 `complexity_gate: true` 且 `complexity == "complex"` 时，路由归为 `complex_unsupported`（不调 LLM 生成实答，输出占位提示）。
- 澄清标志：intent 配置 `needs_clarification: true` 时路由置 `needs_clarification=True`；但一旦归为 `complex_unsupported` 则澄清标志强制为 `False`。
- 一次性生成：所有处理器调用 `chat_completion`（非流式）返回完整 `str`。
- 历史上限：最近 20 轮。
- 退出命令：`exit` / `quit`。
- 测试运行方式：项目根目录执行 `uv run pytest`；入口 `uv run python -m agent [config.json] [--ask 'question']`。
- 删除不再使用的 `agent/generator.py` 与 `tests/test_generator.py`。

---

### Task 1: 配置连接信息重构（AgentConfig → domain_dir）

**Files:**
- Modify: `pyproject.toml`
- Modify: `agent/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass class AgentConfig`，字段：`base_url: str`、`model: str`、`classifier_model: str`、`domain_dir: str`。
  - `def load_config(path: str | None = None) -> AgentConfig` —— `path` 缺省为 `"config.json"`；`AGENT_BASE_URL` 覆盖 `base_url`；`classifier_model` 缺失回退到 `model`；`domain_dir` 必填（非空字符串），缺失抛 `ConfigError`；文件不存在/非法 JSON/缺 `base_url`/缺 `model` 抛 `ConfigError`。
  - `def get_api_key() -> str`（保持原样）。
  - 所有 `ConfigError` 消息为英文。
- Consumes: 无（`ConfigError` 定义沿用）。
- 后续 Task 2 在 `config.py` 中追加 `DomainConfig`；Task 4+ 使用 `AgentConfig`。

- [ ] **Step 1: 添加 pyyaml 依赖**

```bash
uv add pyyaml
```

Expected: `pyyaml` 加入 `pyproject.toml` 与 `uv.lock`，`uv sync` 自动执行。

- [ ] **Step 2: 改写失败测试（重写 `tests/test_config.py`）**

```python
import json

import pytest

from agent.config import AgentConfig, ConfigError, get_api_key, load_config


def _write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_load_config_basic(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "classifier_model": "classifier-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert isinstance(cfg, AgentConfig)
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "model-a"
    assert cfg.classifier_model == "classifier-a"
    assert cfg.domain_dir == "domain/software_engineering"


def test_classifier_model_falls_back_to_model(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "model-a"


def test_env_base_url_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BASE_URL", "https://env.example.com/v1")
    path = _write_config(tmp_path, {
        "base_url": "https://file.example.com/v1",
        "model": "m",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.base_url == "https://env.example.com/v1"


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.json")


def test_invalid_json_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = tmp_path / "config.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_missing_base_url_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"model": "m", "domain_dir": "d"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_model_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"base_url": "https://x/v1", "domain_dir": "d"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_domain_dir_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"base_url": "https://x/v1", "model": "m"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_get_api_key(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "secret")
    assert get_api_key() == "secret"


def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_api_key()
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL（`AttributeError` / 断言失败，因 `AgentConfig` 字段变化）。

- [ ] **Step 4: 实现 `agent/config.py`**

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_dir: str


def load_config(path: str | None = None) -> AgentConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"Config file not found: {config_path}. "
            "Create one by copying config.example.json."
        )
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid config JSON: {e}")

    if not isinstance(raw, dict):
        raise ConfigError("Config top-level must be a JSON object.")

    base_url = os.environ.get("AGENT_BASE_URL") or raw.get("base_url")
    if not base_url:
        raise ConfigError("Missing 'base_url' in config or AGENT_BASE_URL env var.")

    model = raw.get("model")
    if not model:
        raise ConfigError("Missing 'model' in config.")

    classifier_model = raw.get("classifier_model") or model

    domain_dir = raw.get("domain_dir")
    if not isinstance(domain_dir, str) or not domain_dir:
        raise ConfigError("Missing 'domain_dir' in config.")

    return AgentConfig(
        base_url=base_url,
        model=model,
        classifier_model=classifier_model,
        domain_dir=domain_dir,
    )


def get_api_key() -> str:
    api_key = os.environ.get("AGENT_API_KEY")
    if not api_key:
        raise ConfigError(
            "AGENT_API_KEY environment variable is not set. "
            "Set it with: export AGENT_API_KEY=your_key"
        )
    return api_key
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（9 项）。

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml uv.lock agent/config.py tests/test_config.py
git commit -m "refactor: config keeps connection info plus domain_dir"
```

---

### Task 2: 领域配置加载（DomainConfig）

**Files:**
- Modify: `agent/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces（追加到 `agent/config.py`）：
  - `@dataclass class IntentDef`：`id: str`、`description: str`、`needs_clarification: bool = False`
  - `@dataclass class StrategyDef`：`id: str`、`model: str | None = None`、`complexity_gate: bool = False`
  - `@dataclass class DomainConfig`：`name: str`、`description: str`、`out_of_domain_reply: str`、`intents: dict[str, IntentDef]`、`intent_mapping: dict[str, str]`、`strategies: dict[str, StrategyDef]`、`prompts: dict[str, str]`
  - `def load_domain_config(domain_dir: str) -> DomainConfig`：
    - 读 `domain_dir/domain.json`（`name` 可空，`description` 必填，`out_of_domain_reply` 缺省生成英文默认语）。
    - 读 `domain_dir/intents.yaml`（list，每项 `{id, description, needs_clarification?}`）。
    - 读 `domain_dir/intent_mapping.yaml`（dict intent→strategy），校验 intent 必须存在于 `intents`、strategy 必须存在于 `strategies`。
    - 读 `domain_dir/strategies.yaml`（dict id→{model?, complexity_gate?}）。
    - 读 `domain_dir/prompts/<strategy>.md` 全部策略 + `clarify.md` + `unsupported_complex.md`。
    - 缺文件 / 非法 JSON / 非法 YAML / 结构错误抛 `ConfigError`。
  - 新增 import：`from pathlib import Path`、`import yaml`。
- Consumes: `ConfigError`（Task 1）。
- 后续 Task 3-9 使用 `DomainConfig`、`IntentDef`、`StrategyDef`、`load_domain_config`。

- [ ] **Step 1: 写失败测试（追加到 `tests/test_config.py`）**

```python
from agent.config import IntentDef, StrategyDef, load_domain_config


def _write_domain(tmp_path, **overrides):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "软件工程",
        "description": "software engineering",
        "out_of_domain_reply": "Out of domain.",
    }, ensure_ascii=False), encoding="utf-8")
    (base / "intents.yaml").write_text(
        "- id: concept_explain\n  description: explain a concept\n"
        "- id: faq\n  description: quick question\n  needs_clarification: true\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text(
        "concept_explain: teaching\nfaq: direct\n", encoding="utf-8"
    )
    (base / "strategies.yaml").write_text(
        "teaching:\n  complexity_gate: true\ndirect:\n  model: model-direct\n",
        encoding="utf-8",
    )
    (base / "prompts" / "teaching.md").write_text(
        "teach {name} {description} {structure}", encoding="utf-8"
    )
    (base / "prompts" / "direct.md").write_text(
        "direct {name} {description} {structure}", encoding="utf-8"
    )
    (base / "prompts" / "clarify.md").write_text("clarify", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("unsupported", encoding="utf-8")
    return str(base)


def test_load_domain_config_basic(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    assert isinstance(domain, DomainConfig)
    assert domain.name == "软件工程"
    assert domain.description == "software engineering"
    assert domain.out_of_domain_reply == "Out of domain."
    assert set(domain.intents) == {"concept_explain", "faq"}
    assert domain.intents["faq"].needs_clarification is True
    assert domain.intents["concept_explain"].needs_clarification is False
    assert domain.intent_mapping == {"concept_explain": "teaching", "faq": "direct"}
    assert domain.strategies["teaching"].complexity_gate is True
    assert domain.strategies["direct"].model == "model-direct"
    assert domain.strategies["direct"].complexity_gate is False
    assert "teach {name}" in domain.prompts["teaching"]
    assert "clarify" in domain.prompts["clarify"]
    assert "unsupported" in domain.prompts["unsupported_complex"]


def test_load_domain_config_out_of_domain_reply_default(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "软件工程", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("", encoding="utf-8")
    (base / "prompts" / "clarify.md").write_text("c", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.out_of_domain_reply == (
        "This question falls outside my expert domain (软件工程) "
        "and I cannot provide a professional answer."
    )


def test_load_domain_config_missing_domain_json(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(str(tmp_path / "no-such-dir"))


def test_load_domain_config_bad_yaml(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(":: not: [valid", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("", encoding="utf-8")
    (base / "prompts" / "clarify.md").write_text("c", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_mapping_unknown_intent(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("- id: faq\n  description: q\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("bogus_intent: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d {structure}", encoding="utf-8")
    (base / "prompts" / "clarify.md").write_text("c", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_missing_prompt(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("- id: faq\n  description: q\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d {structure}", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL（`ImportError: cannot import name 'DomainConfig'`）。

- [ ] **Step 3: 实现 `DomainConfig` 等（追加到 `agent/config.py`）**

在 `get_api_key` 之后追加：

```python
from pathlib import Path

import yaml


@dataclass
class IntentDef:
    id: str
    description: str
    needs_clarification: bool = False


@dataclass
class StrategyDef:
    id: str
    model: str | None = None
    complexity_gate: bool = False


@dataclass
class DomainConfig:
    name: str
    description: str
    out_of_domain_reply: str
    intents: dict[str, IntentDef]
    intent_mapping: dict[str, str]
    strategies: dict[str, StrategyDef]
    prompts: dict[str, str]


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


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_domain_config(domain_dir: str) -> DomainConfig:
    base = Path(domain_dir)
    meta = _read_json(base / "domain.json")
    name = meta.get("name") or ""
    description = meta.get("description")
    if not description:
        raise ConfigError(f"Missing 'description' in {base / 'domain.json'}")
    out_of_domain_reply = meta.get("out_of_domain_reply") or (
        f"This question falls outside my expert domain ({name}) "
        "and I cannot provide a professional answer."
    )

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
            needs_clarification=bool(item.get("needs_clarification", False)),
        )

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

    strategies_data = _read_yaml(base / "strategies.yaml")
    if strategies_data is None:
        strategies_data = {}
    if not isinstance(strategies_data, dict):
        raise ConfigError(f"strategies.yaml must contain a mapping: {base / 'strategies.yaml'}")
    strategies: dict[str, StrategyDef] = {}
    for sid, item in strategies_data.items():
        if isinstance(item, dict):
            model = item.get("model")
            strategies[sid] = StrategyDef(
                id=sid,
                model=model if isinstance(model, str) and model else None,
                complexity_gate=bool(item.get("complexity_gate", False)),
            )
        else:
            strategies[sid] = StrategyDef(id=sid)

    for intent_id, strategy_id in intent_mapping.items():
        if strategy_id not in strategies:
            raise ConfigError(
                f"Mapping for intent '{intent_id}' references unknown strategy "
                f"'{strategy_id}' in {base / 'intent_mapping.yaml'}"
            )

    prompts: dict[str, str] = {}
    prompt_dir = base / "prompts"
    for sid in strategies:
        prompts[sid] = _read_prompt(prompt_dir / f"{sid}.md")
    prompts["clarify"] = _read_prompt(prompt_dir / "clarify.md")
    prompts["unsupported_complex"] = _read_prompt(prompt_dir / "unsupported_complex.md")

    return DomainConfig(
        name=name,
        description=description,
        out_of_domain_reply=out_of_domain_reply,
        intents=intents,
        intent_mapping=intent_mapping,
        strategies=strategies,
        prompts=prompts,
    )
```

注意：`yaml`、`Path` 的 import 需放在文件顶部（`from pathlib import Path` 在 `from dataclasses import dataclass` 之后；`import yaml` 在 `import os` 附近）。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（15 项：Task1 9 + Task2 6）。

- [ ] **Step 5: 提交**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: load domain config directory (yaml + markdown prompts)"
```

---

### Task 3: 分类器扩展（Intent + Complexity）

**Files:**
- Modify: `agent/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Produces（在 `agent/classifier.py` 中）：
  - `@dataclass class IntentClassification`：`intent_id: str`、`reason: str`
  - `@dataclass class ComplexityClassification`：`level: str`、`reason: str`
  - `def classify_intent(client, question: str, domain_name: str, domain_description: str, intents: list[str], *, model: str | None = None) -> IntentClassification`
    - prompt 枚举可选 intents；解析 `{"intent": ..., "reason": ...}`；`intent` 不在白名单则视为解析失败重试；失败降级 `IntentClassification("", "Unreliable classification: classifier output could not be parsed")`。
  - `def classify_complexity(client, question: str, domain_name: str, domain_description: str, *, model: str | None = None) -> ComplexityClassification`
    - 解析 `{"complexity": "simple"|"medium"|"complex", "reason": ...}`；非法 level 视为解析失败重试；失败降级 `ComplexityClassification("medium", "Unreliable classification: classifier output could not be parsed")`。
  - 内部：抽 `_classify_json(client, prompt, parser, *, model) -> object | None`（严格 prompt 重试一次，供三个分类器共用）。
- Consumes: `LLMClient`（Task 3 既有）、既有 `Classification`/`classify_question`（保持原行为与既有测试通过）。
- 后续 Task 4 `Router` 使用 `classify_intent`、`classify_complexity`、`classify_question`。

- [ ] **Step 1: 写失败测试（追加到 `tests/test_classifier.py`）**

```python
from agent.classifier import (
    Classification,
    classify_complexity,
    classify_intent,
    classify_question,
)
```

（替换原 import 为上述；保留 `FakeClient` 与既有 6 个测试不变。）

```python
def test_classify_intent_success():
    client = FakeClient(['{"intent": "concept_explain", "reason": "why"}'])
    result = classify_intent(client, "why is interface like this", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == "concept_explain"
    assert result.reason == "why"
    assert client.calls[0][2] is True


def test_classify_intent_unknown_retries():
    client = FakeClient(
        ['{"intent": "bogus", "reason": "x"}', '{"intent": "faq", "reason": "y"}']
    )
    result = classify_intent(client, "q", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == "faq"
    assert len(client.calls) == 2


def test_classify_intent_unreliable_falls_back_empty():
    client = FakeClient(["garbage", "garbage"])
    result = classify_intent(client, "q", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == ""
    assert "Unreliable" in result.reason
    assert len(client.calls) == 2


def test_classify_complexity_success():
    client = FakeClient(['{"complexity": "complex", "reason": "big scope"}'])
    result = classify_complexity(client, "design a redis cluster", "软件工程", "sw")
    assert result.level == "complex"


def test_classify_complexity_invalid_level_retries():
    client = FakeClient(
        ['{"complexity": "huge", "reason": "x"}', '{"complexity": "medium", "reason": "y"}']
    )
    result = classify_complexity(client, "q", "软件工程", "sw")
    assert result.level == "medium"
    assert len(client.calls) == 2


def test_classify_complexity_unreliable_defaults_medium():
    client = FakeClient(["garbage", "garbage"])
    result = classify_complexity(client, "q", "软件工程", "sw")
    assert result.level == "medium"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL（`ImportError`，`classify_intent` 未定义）。

- [ ] **Step 3: 实现（重写 `agent/classifier.py`）**

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm import LLMClient


@dataclass
class Classification:
    in_domain: bool
    reason: str


@dataclass
class IntentClassification:
    intent_id: str
    reason: str


@dataclass
class ComplexityClassification:
    level: str
    reason: str


_CLASSIFY_PROMPT = """You are a domain boundary judge. Given an expert domain, decide whether the user's question belongs to that domain.

Expert domain name: {name}
Expert domain description: {description}

Rules:
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"in_domain": true or false, "reason": "one-sentence justification"}}

User question: {question}
"""

_INTENT_PROMPT = """You are an intent judge for an expert agent in the {name} domain.

Domain description: {description}

Available intents:
{intents}

Rules:
- Choose the single intent that best matches the user's goal.
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"intent": "one of the intent ids above", "reason": "one-sentence justification"}}

User question: {question}
"""

_COMPLEXITY_PROMPT = """You are a task complexity judge for an expert agent in the {name} domain.

Domain description: {description}

Complexity levels:
- simple: answerable in a short, direct response
- medium: requires some structured explanation
- complex: large scope, multiple steps or subsystems

Rules:
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"complexity": "simple" or "medium" or "complex", "reason": "one-sentence justification"}}

User question: {question}
"""

_STRICT_REMINDER = "\nReminder: output ONLY the JSON object above and no other text."


def _build_classify_prompt(name: str, description: str, question: str) -> str:
    return _CLASSIFY_PROMPT.format(name=name, description=description, question=question)


def _classify_json(client, prompt: str, parser, *, model: str | None = None):
    for strict in (False, True):
        text = client.chat_completion(
            [{"role": "system", "content": prompt + (_STRICT_REMINDER if strict else "")}],
            model=model,
            disable_thinking=True,
        )
        parsed = parser(text)
        if parsed is not None:
            return parsed
    return None


def _parse_classification(text: str) -> Classification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("in_domain"), bool):
        return None
    reason = data.get("reason")
    return Classification(
        in_domain=data["in_domain"],
        reason=reason if isinstance(reason, str) else "",
    )


def _parse_intent(text: str, allowed: set[str]) -> IntentClassification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("intent"), str):
        return None
    if data["intent"] not in allowed:
        return None
    reason = data.get("reason")
    return IntentClassification(
        intent_id=data["intent"],
        reason=reason if isinstance(reason, str) else "",
    )


def _parse_complexity(text: str) -> ComplexityClassification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("complexity"), str):
        return None
    if data["complexity"] not in {"simple", "medium", "complex"}:
        return None
    reason = data.get("reason")
    return ComplexityClassification(
        level=data["complexity"],
        reason=reason if isinstance(reason, str) else "",
    )


def classify_question(
    client: LLMClient,
    question: str,
    domain_name: str,
    domain_description: str,
    *,
    model: str | None = None,
) -> Classification:
    prompt = _build_classify_prompt(domain_name, domain_description, question)
    result = _classify_json(client, prompt, _parse_classification, model=model)
    if result is not None:
        return result
    return Classification(
        in_domain=False, reason="Unreliable classification: classifier output could not be parsed"
    )


def classify_intent(
    client: LLMClient,
    question: str,
    domain_name: str,
    domain_description: str,
    intents: list[str],
    *,
    model: str | None = None,
) -> IntentClassification:
    prompt = _INTENT_PROMPT.format(
        name=domain_name,
        description=domain_description,
        intents="\n".join(f"- {i}" for i in intents),
        question=question,
    )
    result = _classify_json(client, prompt, lambda t: _parse_intent(t, set(intents)), model=model)
    if result is not None:
        return result
    return IntentClassification(
        intent_id="", reason="Unreliable classification: classifier output could not be parsed"
    )


def classify_complexity(
    client: LLMClient,
    question: str,
    domain_name: str,
    domain_description: str,
    *,
    model: str | None = None,
) -> ComplexityClassification:
    prompt = _COMPLEXITY_PROMPT.format(
        name=domain_name, description=domain_description, question=question
    )
    result = _classify_json(client, prompt, _parse_complexity, model=model)
    if result is not None:
        return result
    return ComplexityClassification(
        level="medium", reason="Unreliable classification: classifier output could not be parsed"
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: PASS（既有 6 + 新增 6 = 12 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/classifier.py tests/test_classifier.py
git commit -m "feat: add intent and complexity classifiers"
```

---

### Task 4: 路由（Router）

**Files:**
- Create: `agent/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Produces：
  - 常量 `DEFAULT_STRATEGY = "direct"`、`COMPLEX_UNSUPPORTED = "complex_unsupported"`
  - `@dataclass class RouteResult`：`in_domain: bool`、`strategy: str`、`intent: str | None = None`、`complexity: str | None = None`、`needs_clarification: bool = False`、`reject_reason: str = ""`
  - `class Router`：`__init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig)`；`route(self, question: str) -> RouteResult`：
    1. `classify_question`，`in_domain=false` → `RouteResult(in_domain=False, strategy="reject", reject_reason=reason)`。
    2. `classify_intent` + `classify_complexity`（均用 `config.classifier_model`）。
    3. `strategy = domain.intent_mapping.get(intent_id, DEFAULT_STRATEGY)`。
    4. 若对应 `StrategyDef.complexity_gate` 为真且 `complexity == "complex"` → `strategy = COMPLEX_UNSUPPORTED`，此时 `needs_clarification = False`。
    5. `needs_clarification = intent_def.needs_clarification`（intent 不存在则 `False`）。
- Consumes: `classify_question`/`classify_intent`/`classify_complexity`（Task 3）、`AgentConfig`/`DomainConfig`（Task 1/2）、`LLMClient`（既有）。
- 后续 Task 6 `Chat` 使用 `Router`、`RouteResult`、`COMPLEX_UNSUPPORTED`。

- [ ] **Step 1: 写失败测试**

```python
from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.router import COMPLEX_UNSUPPORTED, Router


def _domain(**overrides):
    default = {
        "name": "软件工程",
        "description": "sw",
        "out_of_domain_reply": "Out.",
        "intents": {
            "concept_explain": IntentDef("concept_explain", "explain"),
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug", needs_clarification=True),
            "architecture_design": IntentDef("architecture_design", "arch"),
        },
        "intent_mapping": {
            "concept_explain": "teaching",
            "faq": "direct",
            "troubleshooting": "debugging",
            "architecture_design": "analysis",
        },
        "strategies": {
            "teaching": StrategyDef("teaching", complexity_gate=True),
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
            "analysis": StrategyDef("analysis", complexity_gate=True),
        },
        "prompts": {},
    }
    default.update(overrides)
    return DomainConfig(**default)


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(self, messages, model=None, disable_thinking=False):
        return self.responses.pop(0)


def test_route_in_domain_simple_strategy():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "concept_explain", "reason": "ok"}',
        '{"complexity": "simple", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("q")
    assert result.in_domain is True
    assert result.strategy == "teaching"
    assert result.intent == "concept_explain"
    assert result.complexity == "simple"
    assert result.needs_clarification is False


def test_route_out_of_domain():
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'])
    result = Router(client, _config(), _domain()).route("weather?")
    assert result.in_domain is False
    assert result.strategy == "reject"
    assert result.reject_reason == "unrelated"


def test_route_unknown_intent_defaults_to_direct():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "", "reason": "unreliable"}',
        '{"complexity": "simple", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "direct"


def test_route_needs_clarification():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("my program hangs")
    assert result.needs_clarification is True


def test_route_complex_gated_to_unsupported():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "architecture_design", "reason": "ok"}',
        '{"complexity": "complex", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("design a big system")
    assert result.strategy == COMPLEX_UNSUPPORTED
    assert result.needs_clarification is False


def test_route_complex_ungated_strategy_stays():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "faq", "reason": "ok"}',
        '{"complexity": "complex", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "direct"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'agent.router'`）。

- [ ] **Step 3: 实现 `agent/router.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from .classifier import classify_complexity, classify_intent, classify_question
from .config import AgentConfig, DomainConfig
from .llm import LLMClient

DEFAULT_STRATEGY = "direct"
COMPLEX_UNSUPPORTED = "complex_unsupported"


@dataclass
class RouteResult:
    in_domain: bool
    strategy: str
    intent: str | None = None
    complexity: str | None = None
    needs_clarification: bool = False
    reject_reason: str = ""


class Router:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain

    def route(self, question: str) -> RouteResult:
        domain_result = classify_question(
            self.client,
            question,
            self.domain.name,
            self.domain.description,
            model=self.config.classifier_model,
        )
        if not domain_result.in_domain:
            return RouteResult(
                in_domain=False, strategy="reject", reject_reason=domain_result.reason
            )

        intent_result = classify_intent(
            self.client,
            question,
            self.domain.name,
            self.domain.description,
            list(self.domain.intents),
            model=self.config.classifier_model,
        )
        complexity_result = classify_complexity(
            self.client,
            question,
            self.domain.name,
            self.domain.description,
            model=self.config.classifier_model,
        )

        strategy = self.domain.intent_mapping.get(intent_result.intent_id, DEFAULT_STRATEGY)
        strategy_def = self.domain.strategies.get(strategy)
        if strategy_def and strategy_def.complexity_gate and complexity_result.level == "complex":
            strategy = COMPLEX_UNSUPPORTED

        needs_clarification = False
        if strategy != COMPLEX_UNSUPPORTED:
            intent_def = self.domain.intents.get(intent_result.intent_id)
            needs_clarification = bool(intent_def and intent_def.needs_clarification)

        return RouteResult(
            in_domain=True,
            strategy=strategy,
            intent=intent_result.intent_id or None,
            complexity=complexity_result.level,
            needs_clarification=needs_clarification,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_router.py -v`
Expected: PASS（6 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/router.py tests/test_router.py
git commit -m "feat: add intent/complexity router with strategy mapping"
```

---

### Task 5: 策略处理器（Processors）

**Files:**
- Create: `agent/processors/__init__.py`
- Create: `agent/processors/base.py`
- Create: `agent/processors/direct.py`
- Create: `agent/processors/teaching.py`
- Create: `agent/processors/debugging.py`
- Create: `agent/processors/analysis.py`
- Create: `agent/processors/coding.py`
- Create: `agent/processors/registry.py`
- Test: `tests/test_processors.py`

**Interfaces:**
- Produces：
  - `class Processor`（base.py）：`strategy_id` 类属性（默认 `"base"`）；`__init__(self, prompt_template: str, domain_name: str, domain_description: str)`；属性 `structure: str`（默认 `""`）；`build_system_prompt(self) -> str`（模板中 `{structure}` 替换为 structure，再 `.format(name=..., description=...)`）；`build_messages(self, history: list[tuple[str, str]], question: str, *, max_turns: int = 20) -> list[dict]`（system + 最近 max_turns 轮历史 + 当前问题）；`process(self, client, question: str, history, *, model: str | None = None) -> str`（调用 `client.chat_completion(messages, model=model)`，返回完整字符串，一次性生成）。
  - 五个子类（各自 `strategy_id` 与 `structure`，详见 Step 3）：
    - `DirectAnswerProcessor`（`direct`，structure 为空）
    - `TeachingProcessor`（`teaching`）
    - `DebuggingProcessor`（`debugging`）
    - `AnalysisProcessor`（`analysis`）
    - `CodingProcessor`（`coding`）
  - `def build_registry(domain: DomainConfig) -> dict[str, Processor]`（registry.py）：`{strategy_id: cls(domain.prompts[strategy_id], domain.name, domain.description)}`。
- Consumes: `DomainConfig`（Task 2）。
- 后续 Task 6 `Chat` 使用 `build_registry`。

- [ ] **Step 1: 写失败测试**

```python
from agent.config import DomainConfig, IntentDef, StrategyDef
from agent.processors.analysis import AnalysisProcessor
from agent.processors.coding import CodingProcessor
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
        "coding": "Code {name} {description} {structure}",
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


def test_coding_structure():
    p = CodingProcessor("X {structure}", "软件工程", "sw")
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
    assert set(registry) == {"direct", "teaching", "debugging", "analysis", "coding"}
    assert isinstance(registry["teaching"], TeachingProcessor)
    assert registry["direct"].build_system_prompt() == "Direct 软件工程 sw "
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_processors.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现各处理器文件**

`agent/processors/__init__.py`：空文件。

`agent/processors/base.py`：

```python
from __future__ import annotations


class Processor:
    strategy_id = "base"

    def __init__(self, prompt_template: str, domain_name: str, domain_description: str):
        self.prompt_template = prompt_template
        self.domain_name = domain_name
        self.domain_description = domain_description

    @property
    def structure(self) -> str:
        return ""

    def build_system_prompt(self) -> str:
        template = self.prompt_template.replace("{structure}", self.structure)
        return template.format(name=self.domain_name, description=self.domain_description)

    def build_messages(
        self,
        history: list[tuple[str, str]],
        question: str,
        *,
        max_turns: int = 20,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.build_system_prompt()}]
        for user_text, assistant_text in history[-max_turns:]:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": question})
        return messages

    def process(self, client, question: str, history: list[tuple[str, str]], *, model: str | None = None) -> str:
        return client.chat_completion(self.build_messages(history, question), model=model)
```

`agent/processors/direct.py`：

```python
from __future__ import annotations

from .base import Processor


class DirectAnswerProcessor(Processor):
    strategy_id = "direct"
```

`agent/processors/teaching.py`：

```python
from __future__ import annotations

from .base import Processor


class TeachingProcessor(Processor):
    strategy_id = "teaching"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Concept\n"
            "- Why it is designed this way\n"
            "- How it works\n"
            "- Concrete example\n"
            "- Common misconceptions\n"
            "- Summary"
        )
```

`agent/processors/debugging.py`：

```python
from __future__ import annotations

from .base import Processor


class DebuggingProcessor(Processor):
    strategy_id = "debugging"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Problem analysis\n"
            "- Possible causes\n"
            "- Verification steps\n"
            "- Fix suggestions\n"
            "- Best practices"
        )
```

`agent/processors/analysis.py`：

```python
from __future__ import annotations

from .base import Processor


class AnalysisProcessor(Processor):
    strategy_id = "analysis"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Comparison dimensions\n"
            "- Key differences\n"
            "- Trade-offs\n"
            "- Recommendation"
        )
```

`agent/processors/coding.py`：

```python
from __future__ import annotations

from .base import Processor


class CodingProcessor(Processor):
    strategy_id = "coding"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Approach\n"
            "- Code with inline explanation\n"
            "- Key considerations\n"
            "- Best practices"
        )
```

`agent/processors/registry.py`：

```python
from __future__ import annotations

from ..config import DomainConfig
from .analysis import AnalysisProcessor
from .base import Processor
from .coding import CodingProcessor
from .debugging import DebuggingProcessor
from .direct import DirectAnswerProcessor
from .teaching import TeachingProcessor

PROCESSOR_CLASSES = {
    "direct": DirectAnswerProcessor,
    "teaching": TeachingProcessor,
    "debugging": DebuggingProcessor,
    "analysis": AnalysisProcessor,
    "coding": CodingProcessor,
}


def build_registry(domain: DomainConfig) -> dict[str, Processor]:
    return {
        sid: cls(domain.prompts[sid], domain.name, domain.description)
        for sid, cls in PROCESSOR_CLASSES.items()
    }
```

（registry.py 需 `from .base import Processor` 以用于类型标注，或在 `PROCESSOR_CLASSES` 前导入 `Processor`。）

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_processors.py -v`
Expected: PASS（7 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/processors/ tests/test_processors.py
git commit -m "feat: add five strategy processors with single-shot generation"
```

---

### Task 6: Chat Interface

**Files:**
- Create: `agent/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Produces：
  - `@dataclass class ChatResponse`：`kind: str`、`text: str`；`kind` ∈ `"answer" | "clarification" | "reject" | "unsupported" | "error"`
  - `class Chat`：
    - `__init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig)`；持有 `router`、`processors`（`build_registry`）、`history: list[tuple[str, str]]`、`_pending: str | None`。
    - `respond(self, question: str, *, allow_clarification: bool = True) -> ChatResponse`：
      1. `route = self.router.route(question)`。
      2. `not in_domain` → `ChatResponse("reject", out_of_domain_reply + (f" ({reject_reason})" if reason else ""))`。
      3. `strategy == "complex_unsupported"` → `ChatResponse("unsupported", prompts["unsupported_complex"])`。
      4. `route.needs_clarification and allow_clarification` → 存 `_pending = question`，返回 `ChatResponse("clarification", self._ask_clarification(question, route))`。
      5. 否则取 processor；策略 model 取 `domain.strategies[strategy].model or config.model`；`process` 生成答案，追加 `(question, answer)` 到 `history`，返回 `ChatResponse("answer", answer)`。
      6. 处理器缺失 → `ChatResponse("error", ...)`。
    - `answer_clarification(self, supplementary: str) -> ChatResponse`：取 `_pending` 合并为 `question + "\n\nAdditional context: " + supplementary`，`respond(merged, allow_clarification=False)`。
    - `_ask_clarification(self, question: str, route) -> str`：用 `prompts["clarify"].format(question=..., intent=route.intent or "unknown", complexity=route.complexity or "unknown")` 调 `client.chat_completion(model=config.classifier_model, disable_thinking=True)`。
- Consumes: `Router`/`RouteResult`/`COMPLEX_UNSUPPORTED`（Task 4）、`build_registry`（Task 5）、`AgentConfig`/`DomainConfig`（Task 1/2）、`LLMClient`（既有）。
- 后续 Task 7/8 使用 `Chat`、`ChatResponse`。

- [ ] **Step 1: 写失败测试**

```python
from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug", needs_clarification=True),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies={
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
        },
        prompts={
            "direct": "Direct {name} {description} {structure}",
            "debugging": "Debug {name} {description} {structure}",
            "clarify": "What do you mean by {question} ({intent}/{complexity})?",
            "unsupported_complex": "Needs orchestrator.",
        },
    )


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(self, messages, model=None, disable_thinking=False):
        return self.responses.pop(0)


def test_respond_reject():
    chat = Chat(FakeClient(['{"in_domain": false, "reason": "unrelated"}']), _config(), _domain())
    resp = chat.respond("weather?")
    assert resp.kind == "reject"
    assert resp.text == "Out of domain. (unrelated)"


def test_respond_answer_appends_history():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "faq", "reason": "ok"}',
        '{"complexity": "simple", "reason": "ok"}',
        "the answer",
    ]), _config(), _domain())
    resp = chat.respond("what is defer")
    assert resp.kind == "answer"
    assert resp.text == "the answer"
    assert chat.history == [("what is defer", "the answer")]


def test_respond_clarification_then_answer():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
        "clarify question",
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
        "the final answer",
    ]), _config(), _domain())
    resp = chat.respond("my go program hangs")
    assert resp.kind == "clarification"
    assert "clarify question" in resp.text
    resp2 = chat.answer_clarification("it hangs on startup")
    assert resp2.kind == "answer"
    assert resp2.text == "the final answer"
    assert chat.history == [
        ("my go program hangs\n\nAdditional context: it hangs on startup", "the final answer")
    ]


def test_respond_skips_clarification_when_disallowed():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
        "direct answer",
    ]), _config(), _domain())
    resp = chat.respond("my program hangs", allow_clarification=False)
    assert resp.kind == "answer"
    assert resp.text == "direct answer"


def test_respond_unsupported_complex():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "complex", "reason": "ok"}',
    ]), _config(), _domain())
    resp = chat.respond("huge debugging task")
    assert resp.kind == "unsupported"
    assert resp.text == "Needs orchestrator."
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_chat.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'agent.chat'`）。

- [ ] **Step 3: 实现 `agent/chat.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig, DomainConfig
from .llm import LLMClient
from .processors.registry import build_registry
from .router import COMPLEX_UNSUPPORTED, Router


@dataclass
class ChatResponse:
    kind: str
    text: str


class Chat:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain
        self.router = Router(client, config, domain)
        self.processors = build_registry(domain)
        self.history: list[tuple[str, str]] = []
        self._pending: str | None = None

    def respond(self, question: str, *, allow_clarification: bool = True) -> ChatResponse:
        route = self.router.route(question)
        if not route.in_domain:
            text = self.domain.out_of_domain_reply
            if route.reject_reason:
                text += f" ({route.reject_reason})"
            return ChatResponse(kind="reject", text=text)
        if route.strategy == COMPLEX_UNSUPPORTED:
            return ChatResponse(
                kind="unsupported", text=self.domain.prompts["unsupported_complex"]
            )
        if route.needs_clarification and allow_clarification:
            self._pending = question
            return ChatResponse(
                kind="clarification", text=self._ask_clarification(question, route)
            )
        processor = self.processors.get(route.strategy)
        if processor is None:
            return ChatResponse(kind="error", text=f"No processor for strategy '{route.strategy}'")
        model = self.config.model
        strategy_def = self.domain.strategies.get(route.strategy)
        if strategy_def and strategy_def.model:
            model = strategy_def.model
        answer = processor.process(self.client, question, self.history, model=model)
        self.history.append((question, answer))
        return ChatResponse(kind="answer", text=answer)

    def answer_clarification(self, supplementary: str) -> ChatResponse:
        pending = self._pending
        self._pending = None
        if pending is None:
            return ChatResponse(kind="answer", text="")
        merged = pending + "\n\nAdditional context: " + supplementary
        return self.respond(merged, allow_clarification=False)

    def _ask_clarification(self, question: str, route) -> str:
        prompt = self.domain.prompts["clarify"].format(
            question=question,
            intent=route.intent or "unknown",
            complexity=route.complexity or "unknown",
        )
        return self.client.chat_completion(
            [{"role": "system", "content": prompt}],
            model=self.config.classifier_model,
            disable_thinking=True,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_chat.py -v`
Expected: PASS（5 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/chat.py tests/test_chat.py
git commit -m "feat: add chat interface with clarification phase"
```

---

### Task 7: 重写 REPL（Chat 驱动）

**Files:**
- Modify: `agent/repl.py`
- Test: `tests/test_repl.py`

**Interfaces:**
- Produces：
  - `def run_repl(client: LLMClient, config: AgentConfig, domain: DomainConfig) -> None`
    - 打印 `ExpertForge | Domain: {domain.name} | Type exit or quit to leave`；循环 `input("you > ")`；`exit`/`quit` 退出；空输入跳过；`EOFError`/`KeyboardInterrupt` 退出。
    - 每轮：`Chat(client, config, domain)` 实例持有（在循环外创建一次）；`chat.respond(question)`；`kind == "clarification"` 时打印 `expert > {text}`，`input("you > ")` 收取补充并 `chat.answer_clarification(supplementary)`；最终打印 `expert > {response.text}`。
    - `LLMError` 捕获打印 `[error] {e}`，继续循环。
- Consumes: `Chat`/`ChatResponse`（Task 6）、`AgentConfig`/`DomainConfig`（Task 1/2）、`LLMClient`/`LLMError`（既有）。
- 后续 Task 8 `agent_cli` 使用 `run_repl`。

- [ ] **Step 1: 写失败测试（重写 `tests/test_repl.py`）**

```python
from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.llm import LLMError
from agent.repl import run_repl


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="m", domain_dir="d")


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies={"direct": StrategyDef("direct")},
        prompts={
            "direct": "Direct {name} {description} {structure}",
            "clarify": "clarify",
            "unsupported_complex": "unsupported",
        },
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(self, messages, model=None, disable_thinking=False):
        return self.responses.pop(0)


def test_repl_answers(monkeypatch, capsys):
    inputs = iter(["What is defer?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "faq", "reason": "ok"}',
        '{"complexity": "simple", "reason": "ok"}',
        "the answer",
    ])
    run_repl(client, _config(), _domain())
    out = capsys.readouterr().out
    assert "the answer" in out


def test_repl_rejects(monkeypatch, capsys):
    inputs = iter(["What is the weather?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'])
    run_repl(client, _config(), _domain())
    out = capsys.readouterr().out
    assert "Out of domain." in out


def test_repl_error_does_not_crash(monkeypatch, capsys):
    inputs = iter(["question", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class ErrorClient:
        def chat_completion(self, messages, model=None, disable_thinking=False):
            raise LLMError("network error")

    run_repl(ErrorClient(), _config(), _domain())
    out = capsys.readouterr().out
    assert "network error" in out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_repl.py -v`
Expected: FAIL（`TypeError`：`run_repl` 新签名 / 旧实现不匹配）。

- [ ] **Step 3: 实现 `agent/repl.py`**

```python
from __future__ import annotations

from .chat import Chat
from .config import AgentConfig, DomainConfig
from .llm import LLMClient, LLMError


def run_repl(client: LLMClient, config: AgentConfig, domain: DomainConfig) -> None:
    chat = Chat(client, config, domain)
    print(f"ExpertForge | Domain: {domain.name} | Type exit or quit to leave")

    while True:
        try:
            question = input("you > ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        try:
            response = chat.respond(question)
            if response.kind == "clarification":
                print("expert > " + response.text)
                try:
                    supplementary = input("you > ").strip()
                except EOFError:
                    print("\nBye.")
                    break
                except KeyboardInterrupt:
                    print("\nBye.")
                    break
                response = chat.answer_clarification(supplementary)
            print("expert > " + response.text)
        except LLMError as e:
            print(f"[error] {e}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_repl.py -v`
Expected: PASS（3 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/repl.py tests/test_repl.py
git commit -m "refactor: drive REPL through Chat interface"
```

---

### Task 8: CLI 入口（--ask）+ 删除 generator

**Files:**
- Modify: `agent/agent_cli.py`
- Delete: `agent/generator.py`
- Delete: `tests/test_generator.py`
- Test: `tests/test_agent_cli.py`

**Interfaces:**
- Produces：
  - `def main(argv: list[str] | None = None) -> int`
    - `-h/--help`：打印 `Usage: python -m agent [config_file_path] [--ask 'question']`，返回 0。
    - 解析参数：`--ask <question>` 提取 `ask`，其余为 positional（第一个作为 config 路径）。
    - 依次 `load_config` → `load_domain_config(config.domain_dir)` → `get_api_key`；任一 `ConfigError` → stderr `Config error: {e}`，返回 1。
    - 创建 `LLMClient(base_url, api_key, model)`。
    - 有 `ask`：`Chat(client, config, domain).respond(ask, allow_clarification=False)`，打印 `response.text`。
    - 无 `ask`：`run_repl(client, config, domain)`。
    - `KeyboardInterrupt` → 打印 `\nBye.`；返回 0。
- Consumes: `Chat`（Task 6）、`run_repl`（Task 7）、`load_config`/`load_domain_config`/`get_api_key`/`ConfigError`（Task 1/2）、`LLMClient`（既有）。
- 删除 `agent/generator.py` 与 `tests/test_generator.py`（不再使用，`build_messages` 由 `Processor.build_messages` 取代）。

- [ ] **Step 1: 写失败测试（重写 `tests/test_agent_cli.py`）**

```python
import json

from agent import agent_cli


def _write_root_config(tmp_path, domain_dir):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://x/v1",
        "model": "m",
        "domain_dir": domain_dir,
    }), encoding="utf-8")
    return str(path)


def _write_domain(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "软件工程", "description": "d", "out_of_domain_reply": "Out.",
    }, ensure_ascii=False), encoding="utf-8")
    (base / "intents.yaml").write_text("- id: faq\n  description: quick\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text(
        "Direct {name} {description} {structure}", encoding="utf-8"
    )
    (base / "prompts" / "clarify.md").write_text("clarify", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("unsupported", encoding="utf-8")
    return str(base)


def test_main_missing_config_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    assert agent_cli.main([str(tmp_path / "no-such.json")]) == 1
    assert "Config error" in capsys.readouterr().err


def test_main_missing_api_key_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))
    assert agent_cli.main([str(config_path)]) == 1
    assert "AGENT_API_KEY" in capsys.readouterr().err


def test_main_help_exits_0(capsys):
    assert agent_cli.main(["-h"]) == 0
    assert "Usage" in capsys.readouterr().out


def test_main_runs_repl(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": iter(["exit"]).__next__())

    class FakeClient:
        def chat_completion(self, messages, model=None, disable_thinking=False):
            return '{"in_domain": true, "reason": "ok"}'

    monkeypatch.setattr(agent_cli, "LLMClient", lambda *a, **k: FakeClient())
    assert agent_cli.main([str(config_path)]) == 0
    assert "ExpertForge" in capsys.readouterr().out


def test_main_ask_prints_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))

    class FakeClient:
        def __init__(self, *a, **k):
            self.responses = [
                '{"in_domain": true, "reason": "ok"}',
                '{"intent": "faq", "reason": "ok"}',
                '{"complexity": "simple", "reason": "ok"}',
                "one-shot answer",
            ]

        def chat_completion(self, messages, model=None, disable_thinking=False):
            return self.responses.pop(0)

    monkeypatch.setattr(agent_cli, "LLMClient", lambda *a, **k: FakeClient())
    assert agent_cli.main([str(config_path), "--ask", "what is defer"]) == 0
    assert "one-shot answer" in capsys.readouterr().out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_agent_cli.py -v`
Expected: FAIL（`load_config` 缺 `domain_dir` 报 `ConfigError` / 断言失败）。

- [ ] **Step 3: 实现 `agent/agent_cli.py`**

```python
from __future__ import annotations

import sys

from .chat import Chat
from .config import ConfigError, get_api_key, load_config, load_domain_config
from .llm import LLMClient
from .repl import run_repl


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] in {"-h", "--help"}:
        print("Usage: python -m agent [config_file_path] [--ask 'question']")
        return 0

    ask: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--ask" and i + 1 < len(args):
            ask = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1
    config_path = positional[0] if positional else None

    try:
        config = load_config(config_path)
        domain = load_domain_config(config.domain_dir)
        api_key = get_api_key()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model)
    try:
        if ask is not None:
            response = Chat(client, config, domain).respond(ask, allow_clarification=False)
            print(response.text)
        else:
            run_repl(client, config, domain)
    except KeyboardInterrupt:
        print("\nBye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 删除 generator**

```bash
git rm agent/generator.py tests/test_generator.py
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_agent_cli.py tests/test_repl.py -v`
Expected: PASS（agent_cli 5 + repl 3）。

- [ ] **Step 6: 提交**

```bash
git add agent/agent_cli.py agent/generator.py tests/test_agent_cli.py tests/test_generator.py
git commit -m "feat: add --ask single-shot entry, drop legacy generator"
```

---

### Task 9: 示例领域配置 + config.example.json + README + 全量回归

**Files:**
- Create: `domain/software_engineering/domain.json`
- Create: `domain/software_engineering/intents.yaml`
- Create: `domain/software_engineering/intent_mapping.yaml`
- Create: `domain/software_engineering/strategies.yaml`
- Create: `domain/software_engineering/prompts/direct.md`
- Create: `domain/software_engineering/prompts/teaching.md`
- Create: `domain/software_engineering/prompts/debugging.md`
- Create: `domain/software_engineering/prompts/analysis.md`
- Create: `domain/software_engineering/prompts/coding.md`
- Create: `domain/software_engineering/prompts/clarify.md`
- Create: `domain/software_engineering/prompts/unsupported_complex.md`
- Modify: `config.example.json`
- Modify: `README.md`
- Modify: `.gitignore`（加入 `domain/` 若未忽略；保留 `config.json` 忽略）

**Interfaces:**
- 无新代码接口；提供可运行的示例领域配置，供 `--ask`/REPL 冒烟与用户上手。

- [ ] **Step 1: 创建领域示例目录**

创建目录 `domain/software_engineering/`，并在其中写：

`domain.json`：

```json
{
  "name": "Software Engineering",
  "description": "Covers software design, development, testing, operations, and performance optimization.",
  "out_of_domain_reply": "This question falls outside my expert domain (Software Engineering) and I cannot provide a professional answer."
}
```

`intents.yaml`：

```yaml
- id: concept_explain
  description: Explain a concept, design rationale, or "why" question
  needs_clarification: false
- id: tutorial
  description: Learn a topic step by step
  needs_clarification: false
- id: learning_guide
  description: Create a learning path or guide
  needs_clarification: false
- id: faq
  description: Quick factual or definitional question
  needs_clarification: false
- id: summarization
  description: Summarize provided content
  needs_clarification: false
- id: troubleshooting
  description: Diagnose and resolve a specific problem
  needs_clarification: true
- id: comparison
  description: Compare two or more technologies or approaches
  needs_clarification: false
- id: performance_analysis
  description: Analyze performance characteristics or bottlenecks
  needs_clarification: false
- id: architecture_design
  description: Design system or component architecture
  needs_clarification: true
- id: generate_code
  description: Write code to accomplish a task
  needs_clarification: true
- id: code_review
  description: Review provided code
  needs_clarification: false
```

`intent_mapping.yaml`：

```yaml
concept_explain: teaching
tutorial: teaching
learning_guide: teaching
faq: direct
summarization: direct
troubleshooting: debugging
comparison: analysis
performance_analysis: analysis
architecture_design: analysis
generate_code: coding
code_review: coding
```

`strategies.yaml`：

```yaml
direct:
  model: null
  complexity_gate: false
teaching:
  model: null
  complexity_gate: true
debugging:
  model: null
  complexity_gate: true
analysis:
  model: null
  complexity_gate: true
coding:
  model: null
  complexity_gate: true
```

`prompts/direct.md`：

```markdown
You are an expert Agent in the {name} domain.

{description}

{structure}

Answering requirements:
- Answer authoritatively and professionally.
- Adjust the structure of your answer to fit each question; do not force a fixed template.
- Only answer questions within this domain.
```

`prompts/teaching.md`：

```markdown
You are an expert Agent in the {name} domain.

{description}

{structure}

Answering requirements:
- Answer authoritatively and professionally.
- Explain the topic thoroughly and insightfully.
- Only answer questions within this domain.
```

`prompts/debugging.md`：

```markdown
You are an expert Agent in the {name} domain.

{description}

{structure}

Answering requirements:
- Answer authoritatively and professionally.
- Be systematic: analyze before proposing fixes.
- Only answer questions within this domain.
```

`prompts/analysis.md`：

```markdown
You are an expert Agent in the {name} domain.

{description}

{structure}

Answering requirements:
- Answer authoritatively and professionally.
- Compare objectively and point out trade-offs.
- Only answer questions within this domain.
```

`prompts/coding.md`：

```markdown
You are an expert Agent in the {name} domain.

{description}

{structure}

Answering requirements:
- Answer authoritatively and professionally.
- Produce clear, idiomatic code with inline explanation.
- Only answer questions within this domain.
```

`prompts/clarify.md`：

```markdown
The user's question is ambiguous or lacks essential context. Ask ONE short, specific question that gets the missing information needed to answer well.

User question: {question}
Detected intent: {intent}
Task complexity: {complexity}
```

`prompts/unsupported_complex.md`：

```markdown
This task requires a full orchestrator pipeline (planning, parallel workers, aggregation, evaluation) that is not yet supported. Please rephrase as a more focused question or split it into smaller steps.
```

- [ ] **Step 2: 更新 `config.example.json`**

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "classifier_model": "deepseek-v4-flash",
  "domain_dir": "domain/software_engineering"
}
```

- [ ] **Step 3: 更新 `README.md`**

覆盖：项目简介（可配置领域专家 Agent、分类→策略路由、五个策略处理器、--ask 单次入口）；安装（`uv sync`）；配置（复制 `config.example.json` 为 `config.json` 并填写 `base_url`、`model`、`domain_dir`；`export AGENT_API_KEY=...`；领域目录结构说明：`domain/software_engineering/` 下 `domain.json`/`intents.yaml`/`intent_mapping.yaml`/`strategies.yaml`/`prompts/*.md`）；运行（`uv run python -m agent` 交互式、`uv run python -m agent --ask "question"` 单次）；退出命令（`exit`/`quit`）。

- [ ] **Step 4: 全量回归测试**

Run: `uv run pytest -v`
Expected: 全部 PASS（config 15 + llm 5 + classifier 12 + router 6 + processors 7 + chat 5 + repl 3 + agent_cli 5 = 58；以实际收集为准）。

- [ ] **Step 5: 端到端冒烟（无 API Key 路径）**

Run: `env -u AGENT_API_KEY uv run python -m agent --ask "What is Go defer?"`
Expected: 退出码 1，stderr 打印 `Config error: AGENT_API_KEY environment variable is not set. ...`（说明配置加载与示例领域目录可用，链路到 API key 检查为止）。

- [ ] **Step 6: 提交**

```bash
git add domain/ config.example.json README.md .gitignore
git commit -m "feat: add example domain config and update docs"
```

---

### 自审记录

- **规格覆盖**：领域目录化（Task 1/2/9）、Intent/Complexity 两次分类（Task 3）、intent→strategy 映射与复杂度 gate（Task 4）、Clarification 前置阶段单次确认（Task 6）、五个处理器一次性生成（Task 5）、Chat Interface 与 `--ask`（Task 6/8）、classifier_model + 每 strategy 可选 model（Task 6 中 `strategy_def.model or config.model`）、非流式一次性生成（Task 5 全程 `chat_completion`）、配置化（yaml+md，Task 2/9）、错误降级（Task 3/6/8）、四类测试 + 冒烟（Task 1-9）、README（Task 9）、Orchestrator 延后为 `complex_unsupported` 占位（Task 4/6）。
- **占位符扫描**：所有步骤含完整代码与预期输出，无 TBD/TODO。
- **类型一致性**：`AgentConfig`（Task 1）在 Task 4/6/7/8 中字段引用一致；`DomainConfig`/`IntentDef`/`StrategyDef`（Task 2）在 Task 4/5/6/7/8 中一致；`RouteResult`/`Router`/`COMPLEX_UNSUPPORTED`（Task 4）在 Task 6 中一致；`Chat`/`ChatResponse`（Task 6）在 Task 7/8 中一致；`build_registry`（Task 5）在 Task 6 中一致；`Processor.build_messages` 取代原 `generator.build_messages`（Task 8 删除 generator）。策略 id 常量统一：映射值来自 yaml，处理器 `strategy_id` 与 `registry` 的 `PROCESSOR_CLASSES` 键一致（direct/teaching/debugging/analysis/coding）。
