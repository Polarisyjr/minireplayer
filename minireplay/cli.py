from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle import load_bundle
from .config import load_config
from .errors import ReplayError
from .llm_store import REPLAY_MODES
from .report import build_report
from .serving import assert_forced_capable, running_vllm_containers, start, stop
from .supervisor import record_bundle, replay_bundle
from .util import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minireplay")
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="record one native sweep into a bundle")
    record.add_argument("--config", type=Path, required=True)
    record.add_argument("--out", type=Path, required=True, help="run evidence directory")
    record.add_argument("--bundle", type=Path, required=True, help="bundle directory to create")
    record.add_argument("--run-id")

    replay = commands.add_parser("replay", help="replay one bundle")
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--mode", choices=REPLAY_MODES, default="tool-only")
    replay.add_argument(
        "--fast-claim",
        action="store_true",
        help="claim slots by position only, skipping the argument digest compare",
    )
    replay.add_argument("--run-id")

    report = commands.add_parser("report", help="compare repeated replays of one bundle")
    report.add_argument("--bundle", type=Path, required=True)
    report.add_argument("--run", type=Path, action="append", required=True)
    report.add_argument("--source", type=Path)
    report.add_argument("--out", type=Path)

    comparison = commands.add_parser(
        "plot-comparison",
        help="plot one recording and one or more validated replays",
    )
    comparison.add_argument("--bundle", type=Path, required=True)
    comparison.add_argument("--source", type=Path, required=True)
    comparison.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="replay directory; repeat for every replay to compare",
    )
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--prefix", help="output filename stem (derived by default)")
    comparison.add_argument("--label", help="workload label shown in figure titles")
    comparison.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("svg", "png"),
        help="output format; repeat to select both (default: svg and png)",
    )

    validate = commands.add_parser("validate", help="validate a bundle")
    validate.add_argument("--bundle", type=Path, required=True)

    up = commands.add_parser(
        "vllm-up", help="start a vLLM fleet that forced decoding can use"
    )
    up.add_argument("--config", type=Path, required=True)

    down = commands.add_parser("vllm-down", help="stop the vLLM fleet")
    down.add_argument("--config", type=Path, required=True)

    return parser


def _run(arguments: argparse.Namespace) -> int:
    if arguments.command == "record":
        bundle = record_bundle(
            config=load_config(arguments.config),
            output=arguments.out,
            bundle_output=arguments.bundle,
            run_id=arguments.run_id,
        )
        print(
            json.dumps(
                {
                    "bundle_id": bundle.manifest["bundle_id"],
                    "counts": bundle.manifest["counts"],
                    "actors": len(bundle.manifest["actors"]),
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "replay":
        run = replay_bundle(
            config=load_config(arguments.config),
            bundle_dir=arguments.bundle,
            output=arguments.out,
            replay_mode=arguments.mode,
            fast_claim=arguments.fast_claim,
            run_id=arguments.run_id,
        )
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "makespan_seconds": run.metrics["makespan_seconds"],
                    "unattributed_gap_seconds": run.timeline["unattributed_gap_seconds"],
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "report":
        report = build_report(
            bundle_dir=arguments.bundle,
            run_dirs=list(arguments.run),
            source_dir=arguments.source,
        )
        if arguments.out is not None:
            atomic_write_json(arguments.out, report)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["valid"] else 1

    if arguments.command == "plot-comparison":
        # Keep matplotlib initialization out of record/replay command startup.
        from .comparison_plot import render_comparison

        result = render_comparison(
            bundle_dir=arguments.bundle,
            source_dir=arguments.source,
            run_dirs=list(arguments.run),
            output_dir=arguments.out,
            prefix=arguments.prefix,
            label=arguments.label,
            formats=arguments.formats or ("svg", "png"),
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    if arguments.command == "validate":
        bundle = load_bundle(arguments.bundle)
        print(
            json.dumps(
                {
                    "valid": True,
                    "bundle_id": bundle.manifest["bundle_id"],
                    "adapter": bundle.adapter,
                    "counts": bundle.manifest["counts"],
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "vllm-up":
        config = load_config(arguments.config)
        spec = config.serving_spec()
        start(spec)
        containers = running_vllm_containers()
        for container in containers:
            assert_forced_capable(container)
        print(
            json.dumps(
                {
                    "image": spec.image,
                    "containers": containers,
                    "audit": str(spec.audit_path_on_host),
                    "forced_capable": True,
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "vllm-down":
        stop(load_config(arguments.config).repo)
        print(json.dumps({"stopped": True}))
        return 0

    raise ReplayError(f"unknown command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except (ReplayError, OSError, ValueError) as exc:
        print(f"minireplay: {exc}", file=sys.stderr)
        return 2
