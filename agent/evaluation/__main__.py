from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import ConfigError, get_api_key, load_config, load_domain_config
from agent.llm import LLMClient

from .dataset import DatasetError, Suite, load_suites
from .diff import diff_runs, load_result
from .metrics import compute_metrics
from .report import format_summary, serialize_results, slim_record, write_baseline, write_result
from .runner import run_evaluation


def _default_dataset(domain_dir: str) -> str:
    return f"evaluation/datasets/{Path(domain_dir).name}"


def _cmd_run(args) -> int:
    if args.max_per_suite is not None and args.max_per_suite < 1:
        print("--max-per-suite must be >= 1", file=sys.stderr)
        return 1
    try:
        config = load_config(args.config)
        domain = load_domain_config(config.domain_dir)
        api_key = get_api_key()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1
    dataset_path = args.dataset or _default_dataset(config.domain_dir)
    try:
        suites = load_suites(dataset_path)
    except DatasetError as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1
    if args.suite:
        wanted = set(args.suite)
        suites = [s for s in suites if s.name in wanted]
        if not suites:
            print(f"No suites matched: {', '.join(args.suite)}", file=sys.stderr)
            return 1
    if args.max_per_suite is not None:
        suites = [Suite(name=s.name, domain=s.domain, cases=s.cases[:args.max_per_suite])
                  for s in suites]
    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model,
                       timeout=config.timeout,
                       provider=config.provider,
                       capability_overrides=config.provider_capabilities)
    results_by_suite: dict[str, list] = {}
    for s in suites:
        results_by_suite[s.name] = run_evaluation(
            config, domain, s, client, skip_quality=args.skip_quality
        )
    metrics_by_suite = {
        s.name: compute_metrics(s, results_by_suite[s.name]) for s in suites
    }
    all_results = [r for rs in results_by_suite.values() for r in rs]
    all_cases = [c for s in suites for c in s.cases]
    merged = Suite(name="all", domain=suites[0].domain, cases=all_cases)
    metrics = compute_metrics(merged, all_results)
    judge_model = (config.evaluation.judge_model if config.evaluation else None) or config.model
    record = serialize_results(
        all_results, metrics, metrics_by_suite,
        domain=merged.domain, label=args.label, model=config.model,
        judge_model=judge_model, skip_quality=args.skip_quality,
        dataset_path=dataset_path, suites=[s.name for s in suites],
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


def _cmd_baseline(args) -> int:
    try:
        record = load_result(args.run)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Baseline error: {e}", file=sys.stderr)
        return 1
    path = args.out or str(Path("evaluation/results") / "baseline.json")
    prev = None
    if Path(path).is_file():
        try:
            prev = load_result(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            prev = None
    write_baseline(path, slim_record(record))
    print(f"Baseline written to: {path}")
    if prev is not None:
        print(diff_runs(prev, slim_record(record)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the golden dataset")
    run_p.add_argument("--dataset", default=None,
                       help="path to dataset directory (or single YAML file)")
    run_p.add_argument("--label", default="run", help="run label for the result file")
    run_p.add_argument("--skip-quality", action="store_true",
                       help="classification/routing/cost only, no answer generation")
    run_p.add_argument("--config", default=None, help="path to agent config.json")
    run_p.add_argument("--results-dir", default=None,
                       help="directory for result JSONs (default: config evaluation.results_dir, "
                            "else evaluation/results)")
    run_p.add_argument("--suite", nargs="+", default=None,
                       help="suites to run by name (default: all suites)")
    run_p.add_argument("--max-per-suite", type=int, default=None,
                       help="cap cases per suite (default: unlimited)")
    run_p.set_defaults(func=_cmd_run)

    diff_p = sub.add_parser("diff", help="compare two run results")
    diff_p.add_argument("run_a", help="first result JSON")
    diff_p.add_argument("run_b", help="second result JSON")
    diff_p.set_defaults(func=_cmd_diff)

    baseline_p = sub.add_parser("baseline", help="record a metrics-only baseline from a run result")
    baseline_p.add_argument("run", help="result JSON to record as the baseline")
    baseline_p.add_argument("--out", default=None,
                            help="output path (default: evaluation/results/baseline.json)")
    baseline_p.set_defaults(func=_cmd_baseline)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
