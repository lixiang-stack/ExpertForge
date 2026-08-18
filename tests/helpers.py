import json
from pathlib import Path


def resolve_live_config_src(root: Path) -> dict:
    """Return the preferred live-test config: the user's config.json when it
    exists, otherwise config.example.json, both resolved under ``root``."""
    for name in ("config.json", "config.example.json"):
        path = root / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No config.json or config.example.json under {root}")


def absolutize_domain_dir(config: dict, root: Path) -> dict:
    """Return a copy of ``config`` with ``domain_dir`` resolved against ``root``."""
    merged = dict(config)
    domain_dir = merged.get("domain_dir")
    path = Path(domain_dir) if domain_dir else root
    merged["domain_dir"] = str(path if path.is_absolute() else root / path)
    return merged
