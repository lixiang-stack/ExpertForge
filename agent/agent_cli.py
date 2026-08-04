from __future__ import annotations

import sys

from .config import ConfigError, get_api_key, load_config
from .llm import LLMClient
from .repl import run_repl


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] in {"-h", "--help"}:
        print("Usage: python -m agent [config_file_path]")
        return 0

    config_path = args[0] if args else None
    try:
        config = load_config(config_path)
        api_key = get_api_key()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model)
    try:
        run_repl(client, config)
    except KeyboardInterrupt:
        print("\nBye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())