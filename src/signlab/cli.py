"""Command line interface for signlab-eval.

Exit codes are part of the contract, because this is meant to sit in CI:

===  ==========================================================================
0    everything passed
1    a quality gate failed (``--fail-under``)
2    the dataset is invalid (validation errors, or it could not be loaded)
3    usage or I/O error
===  ==========================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import DatasetError, load_dataset
from .metrics import evaluate
from .predictors import UltralyticsPredictor, load_predictions
from .report import metrics_section, write_html, write_json
from .validate import validate_dataset
from .version import __version__

EXIT_OK = 0
EXIT_GATE = 1
EXIT_INVALID = 2
EXIT_USAGE = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signlab",
        description="Validate YOLO datasets and evaluate detectors reproducibly.",
    )
    parser.add_argument("--version", action="version", version=f"signlab-eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="check a dataset for label and leakage problems")
    validate.add_argument("--dataset", required=True, type=Path, help="dataset root containing classes.yaml")
    validate.add_argument("--min-box-px", type=float, default=4.0)
    validate.add_argument("--max-aspect", type=float, default=20.0)
    validate.add_argument("--imbalance-ratio", type=float, default=20.0)
    validate.add_argument("--no-leakage", action="store_true", help="skip train/val overlap detection")
    validate.add_argument("--out", type=Path, help="write a JSON report here")

    evaluate_cmd = sub.add_parser("evaluate", help="score detections against a labelled split")
    source = evaluate_cmd.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="JSON Lines file of detections")
    source.add_argument("--weights", type=Path, help="Ultralytics checkpoint (needs the ultralytics extra)")
    evaluate_cmd.add_argument("--dataset", required=True, type=Path)
    evaluate_cmd.add_argument("--split", default="val")
    evaluate_cmd.add_argument("--conf", type=float, default=0.001, help="score floor for --weights runs")
    evaluate_cmd.add_argument("--class-ids", help="comma separated subset of class ids to score")
    evaluate_cmd.add_argument("--fail-under-map50", type=float, help="exit 1 when mAP@50 is below this")
    evaluate_cmd.add_argument("--fail-under-map50-95", type=float, help="exit 1 when mAP@50-95 is below this")
    evaluate_cmd.add_argument("--out", type=Path, help="write a JSON report here")
    evaluate_cmd.add_argument("--html", type=Path, help="also write a self-contained HTML report")

    report = sub.add_parser("report", help="merge validation and metrics JSON into one HTML report")
    report.add_argument("--validation", type=Path, help="JSON written by 'signlab validate --out'")
    report.add_argument("--metrics", type=Path, help="JSON written by 'signlab evaluate --out'")
    report.add_argument("--html", required=True, type=Path)

    return parser


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    report = validate_dataset(
        dataset,
        min_box_px=args.min_box_px,
        max_aspect=args.max_aspect,
        check_leakage=not args.no_leakage,
        imbalance_ratio=args.imbalance_ratio,
    )
    print(report.render())

    if args.out:
        write_json({"tool": "signlab-eval", "version": __version__, **report.to_dict()}, args.out)
        print(f"\nwrote {args.out}")
    return EXIT_OK if report.ok else EXIT_INVALID


def _cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    try:
        split = dataset.split(args.split)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    if args.predictions:
        try:
            predictor = load_predictions(args.predictions)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
    else:
        try:
            predictor = UltralyticsPredictor(args.weights, conf=args.conf)
        except (OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE

    class_ids = None
    if args.class_ids:
        try:
            class_ids = [int(part) for part in args.class_ids.split(",") if part.strip()]
        except ValueError:
            print("error: --class-ids must be comma separated integers", file=sys.stderr)
            return EXIT_USAGE

    per_image = {}
    for sample in split.samples:
        if sample.error:
            continue
        per_image[sample.image_id] = (predictor.predict(sample.image_path), sample.ground_truth())

    report = evaluate(per_image, class_ids=class_ids)
    print(report.render())

    payload = {"tool": "signlab-eval", "version": __version__, "split": args.split, **report.to_dict()}

    failures = []
    if args.fail_under_map50 is not None and report.map50 < args.fail_under_map50:
        failures.append(f"mAP@50 {report.map50:.4f} < {args.fail_under_map50}")
    if args.fail_under_map50_95 is not None and report.map50_95 < args.fail_under_map50_95:
        failures.append(f"mAP@50-95 {report.map50_95:.4f} < {args.fail_under_map50_95}")
    if failures:
        print("\nquality gate failed: " + "; ".join(failures), file=sys.stderr)

    if args.out:
        write_json(payload, args.out)
        print(f"\nwrote {args.out}")
    if args.html:
        sections = [metrics_section(payload)]
        write_html("signlab-eval report", sections, args.html)
        print(f"wrote {args.html}")

    return EXIT_GATE if failures else EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    sections: list[tuple[str, dict]] = []
    title = "signlab-eval report"
    for label, path in (("Validation", args.validation), ("Metrics", args.metrics)):
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        except json.JSONDecodeError as exc:
            print(f"error: {path} is not valid JSON ({exc})", file=sys.stderr)
            return EXIT_USAGE
        title = f"signlab-eval - {label.lower()}"
        sections.append((label, payload))

    if not sections:
        print("error: pass --validation and/or --metrics", file=sys.stderr)
        return EXIT_USAGE

    write_html(title, sections, args.html)
    print(f"wrote {args.html}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {"validate": _cmd_validate, "evaluate": _cmd_evaluate, "report": _cmd_report}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())