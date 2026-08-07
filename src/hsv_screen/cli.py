from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import affinity_input, docking, final_select, ligands, postdock3d, prescreen_pool, qc, scoring, screen2d, screen3d, standardize
from .config import load_config, resolve_path
from .reference import validate_and_prepare
from .distributed_pipeline import run as run_distributed_pipeline
from .distributed_runtime import launch as launch_distributed


STAGES = {
    "references": lambda config, force: validate_and_prepare(config, prepare_pdbqt=True),
    "standardize": standardize.run,
    "score2d": screen2d.run,
    "pool3d": prescreen_pool.run,
    "score3d": screen3d.run,
    "select": final_select.run,
    "ligands": ligands.run,
    "prepare-smina": docking.prepare,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="H9E937 8V1Q-WT staged screening pipeline")
    parser.add_argument("command", choices=[
        "all", "complete", "smina", *STAGES, "qc", "path", "collect-docking",
        "postdock3d", "prepare-affinity", "integrated-score", "final-score"
    ])
    parser.add_argument("--config", type=Path, default=Path("config/full.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profile", default="wt", help="Integrated-scoring profile (WT-only workflow: wt)")
    parser.add_argument("--top-n", type=int, default=200, help="Number of top compounds to export")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if args.force and args.command == "all":
        output_root = resolve_path(config, config["output_dir"])
        if output_root.exists():
            shutil.rmtree(output_root)
    if args.command == "path":
        print(resolve_path(config, config["output_dir"]))
        return
    if args.command == "qc":
        result = qc.run(config)
    elif args.command == "complete":
        result = launch_distributed(args.config, "complete", run_distributed_pipeline)
    elif args.command == "smina":
        result = launch_distributed(args.config, "smina", run_distributed_pipeline)
    elif args.command == "collect-docking":
        result = docking.collect(config, args.force)
    elif args.command == "postdock3d":
        result = postdock3d.run(config, args.force)
    elif args.command == "prepare-affinity":
        result = affinity_input.run(config, args.force)
    elif args.command == "integrated-score":
        result = scoring.run(config, args.force, args.profile, args.top_n)
    elif args.command == "final-score":
        result = scoring.run(config, args.force, args.profile, args.top_n)
    elif args.command == "all":
        result = None
        for name, function in STAGES.items():
            print(f"\n=== {name} ===", flush=True)
            result = function(config, args.force)
        result = qc.run(config)
    else:
        result = STAGES[args.command](config, args.force)
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
