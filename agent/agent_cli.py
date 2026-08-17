from __future__ import annotations

import sys

from .chat import Chat
from .config import ConfigError, effective_timeout, get_api_key, load_config, load_domain_config
from .llm import LLMClient
from .observability import install
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

    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model,
                       timeout=effective_timeout(config))
    client, _obs_plugin = install(client, config, domain)
    try:
        if ask is not None:
            response = Chat(client, config, domain).respond(ask)
            print(response.text)
        else:
            run_repl(client, config, domain)
    except KeyboardInterrupt:
        print("\nBye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
