import json
from pathlib import Path

import pytest

from tests.helpers import absolutize_domain_dir, resolve_live_config_src


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_resolve_prefers_user_config(tmp_path):
    _write(tmp_path / "config.json", {"base_url": "gemini", "model": "g"})
    _write(tmp_path / "config.example.json", {"base_url": "deepseek", "model": "d"})
    assert resolve_live_config_src(tmp_path)["base_url"] == "gemini"


def test_resolve_falls_back_to_example(tmp_path):
    _write(tmp_path / "config.example.json", {"base_url": "deepseek", "model": "d"})
    assert resolve_live_config_src(tmp_path)["base_url"] == "deepseek"


def test_resolve_raises_when_neither_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_live_config_src(tmp_path)


def test_absolutize_relative_domain_dir(tmp_path):
    cfg = absolutize_domain_dir({"domain_dir": "domain/software_engineering"}, tmp_path)
    assert cfg["domain_dir"] == str(tmp_path / "domain/software_engineering")


def test_absolutize_keeps_absolute_domain_dir(tmp_path):
    cfg = absolutize_domain_dir({"domain_dir": "/abs/path"}, tmp_path)
    assert cfg["domain_dir"] == "/abs/path"


def test_absolutize_does_not_mutate_input(tmp_path):
    src = {"domain_dir": "domain/software_engineering"}
    absolutize_domain_dir(src, tmp_path)
    assert src["domain_dir"] == "domain/software_engineering"
