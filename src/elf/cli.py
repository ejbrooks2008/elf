"""Command-line interface for elf.

Provides subcommands for dataset prep, labeling, training, evaluation, serving, and export.
This is a lightweight scaffold; each command delegates to the corresponding module.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Optional

from elf.data import curate, labeling
from elf.training import runner as training_runner
from elf.evaluation import runner as eval_runner
from elf.serve import app as serve_app
from elf.training import checkpointing


def _path_type(path_str: str) -> pathlib.Path:
    return pathlib.Path(path_str).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elf training framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare-data
    prep = subparsers.add_parser("prepare-data", help="Ingest, clean, and split datasets")
    prep.add_argument("--config", type=_path_type, required=False, help="Path to data prep config (YAML/JSON)")
    prep.set_defaults(func=lambda args: curate.prepare_data(config_path=args.config))

    # label
    label = subparsers.add_parser("label", help="Launch labeling/curation workflow")
    label.add_argument("--backend", choices=["fiftyone", "label-studio"], default="fiftyone")
    label.add_argument("--config", type=_path_type, required=False)
    label.set_defaults(func=lambda args: labeling.launch_labeling(backend=args.backend, config_path=args.config))

    # train
    train = subparsers.add_parser("train", help="Train a model with LoRA/DoRA adapters")
    train.add_argument("--config", type=_path_type, required=True, help="Training config file")
    train.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    train.set_defaults(func=lambda args: training_runner.train(config_path=args.config, resume=args.resume))

    # eval
    ev = subparsers.add_parser("eval", help="Evaluate a trained model")
    ev.add_argument("--config", type=_path_type, required=True, help="Evaluation config file")
    ev.set_defaults(func=lambda args: eval_runner.run_eval(config_path=args.config))

    # serve
    serve = subparsers.add_parser("serve", help="Launch inference/serving API or UI")
    serve.add_argument("--config", type=_path_type, required=False, help="Serving config file")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=lambda args: serve_app.serve(config_path=args.config, host=args.host, port=args.port))

    # export
    exp = subparsers.add_parser("export", help="Export adapters or full models for deployment")
    exp.add_argument("--input", type=_path_type, required=True, help="Checkpoint or adapter path")
    exp.add_argument("--output", type=_path_type, required=True, help="Export destination")
    exp.add_argument("--format", choices=["safetensors", "diffusers"], default="safetensors")
    exp.set_defaults(func=lambda args: checkpointing.export_artifacts(args.input, args.output, args.format))

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
