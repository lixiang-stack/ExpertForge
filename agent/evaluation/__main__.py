from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import ConfigError, get_api_key, load_config, load_domain_config
from agent.llm import LLMClient

from .dataset import DatasetError, load_dataset
from .diff import diff_runs, load_result
from .metrics import compute_metrics
from .report import format_summary, serialize_results, write_result
from .runner import run_evaluation


def _default_dataset(domain_dir: str) -> str:
    return f"evaluation/datasets/{Path(domain_dir).name}.yaml"


def _cmd_run(args) -> int:
    try:
        config = load_config(args.config)
        domain = load_domain_config(config.domain_dir)
        api_key = get_api_key()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1
    dataset_path = args.dataset or _default_dataset(config.domain_dir)
    try:
        dataset = load_dataset(dataset_path)
    except DatasetError as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1
    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model)
    results = run_evaluation(config, domain, dataset, client, skip_quality=args.skip_quality)
    metrics = compute_metrics(dataset, results)
    judge_model = (config.evaluation.judge_model if config.evaluation else None) or config.model
    record = serialize_results(
        results, metrics,
        domain=dataset.domain, label=args.label, model=config.model,
        judge_model=judge_model, skip_quality=args.skip_quality,
        dataset_path=dataset_path,
    )
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = "evaluation/results"
        eval_cfg = getattr(config, "evaluation", None)
        if eval_cfg is not None:
            results_dir = eval_cfg.results_dir
    path = write_result(results_dir, record, label=args.label)
    print(format_summary(record))
    print(f"Result written to: {path}")
    return 0


def _cmd_diff(args) -> int:
    try:
        a = load_result(args.run_a)
        b = load_result(args.run_b)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Diff error: {e}", file=sys.stderr)
        return 1
    print(diff_runs(a, b))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the golden dataset")
    run_p.add_argument("--dataset", default=None, help="path to dataset YAML")
    run_p.add_argument("--label", default="run", help="run label for the result file")
    run_p.add_argument("--skip-quality", action="store_true",
                       help="classification/routing/cost only, no answer generation")
    run_p.add_argument("--config", default=None, help="path to agent config.json")
    run_p.add_argument("--results-dir", default=None,
                       help="directory for result JSONs (default: config evaluation.results_dir, "
                            "else evaluation/results)")
    run_p.set_defaults(func=_cmd_run)

    diff_p = sub.add_parser("diff", help="compare two run results")
    diff_p.add_argument("run_a", help="first result JSON")
    diff_p.add_argument("run_b", help="second result JSON")
    diff_p.set_defaults(func=_cmd_diff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
