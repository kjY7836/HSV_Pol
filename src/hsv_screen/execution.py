from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import (
    affinity_input,
    docking,
    final_select,
    ligands,
    postdock3d,
    prescreen_pool,
    qc,
    screen2d,
    screen3d,
    standardize,
)
from .config import load_config, resolve_path
from .reference import validate_and_prepare


PREPARATION_STAGES = [
    ("references", lambda config: validate_and_prepare(config, prepare_pdbqt=True)),
    ("standardize", standardize.run),
    ("score2d", screen2d.run),
    ("pool3d", prescreen_pool.run),
    ("score3d", screen3d.run),
    ("select", final_select.run),
    ("ligands", ligands.run),
    ("prepare-smina", docking.prepare),
]


def prepare_pipeline(config: dict) -> dict:
    for name, function in PREPARATION_STAGES:
        print(f"\n=== {name} ===", flush=True)
        function(config)
    return qc.run(config)


def _smina_binary() -> Path:
    adjacent = Path(sys.executable).resolve().parent / "smina"
    candidate = adjacent if adjacent.is_file() else Path(shutil.which("smina") or "")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"smina was not found next to the current Python ({adjacent}) or on PATH")
    return candidate.resolve()


def _run_smina_job(binary: Path, row: dict[str, str]) -> str:
    command = [
        str(binary), "--receptor", row["receptor"], "--ligand", row["ligands"],
        "--center_x", row["center_x"], "--center_y", row["center_y"],
        "--center_z", row["center_z"], "--size_x", row["size_x"],
        "--size_y", row["size_y"], "--size_z", row["size_z"],
        "--cpu", row["cpu"], "--exhaustiveness", row["exhaustiveness"],
        "--num_modes", row["num_modes"], "--seed", row["seed"],
        "--out", row["output"], "--log", row["log"],
    ]
    process = subprocess.run(
        command, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(
            f"Smina failed for protocol={row['protocol_id']} chunk={row['chunk']} "
            f"(log: {row['log']}):\n{process.stderr[-2000:]}")
    return row["chunk"]


def run_smina_and_postprocess(config: dict) -> dict:
    root = resolve_path(config, config["output_dir"])
    manifest_path = root / "07_smina" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run the preparation stages first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("production_ready", False):
        raise RuntimeError(f"8V1Q-WT production gate failed; inspect {manifest_path}")

    with Path(manifest["jobs"]).open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    if not jobs:
        raise ValueError("Smina job table is empty")

    binary = _smina_binary()
    parallel = int(config["docking"]["parallel_jobs"])
    if parallel != 64:
        raise ValueError(f"This workflow requires docking.parallel_jobs=64, found {parallel}")
    print(
        f"\n=== smina: {len(jobs)} chunks, 64 concurrent one-CPU jobs ===",
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=min(64, len(jobs))) as executor:
        futures = [executor.submit(_run_smina_job, binary, row) for row in jobs]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == len(jobs) or completed % 10 == 0:
                print(f"[smina] {completed}/{len(jobs)} chunks", flush=True)

    print("\n=== collect-docking ===", flush=True)
    docking_summary = docking.collect(config, force=True)
    print("\n=== postdock3d ===", flush=True)
    postdock_summary = postdock3d.run(config, force=True)
    print("\n=== prepare-affinity ===", flush=True)
    affinity_summary = affinity_input.run(config, force=True)
    return {
        "status": "WAITING_FOR_EXTERNAL_AFFINITY",
        "docking": docking_summary,
        "postdock_3d": postdock_summary,
        "affinity_input": affinity_summary,
    }


def run_complete(config_path: str | Path) -> dict:
    config, _ = load_config(config_path)
    if int(config["workers"]) != 64:
        raise ValueError(f"This workflow requires workers=64, found {config['workers']}")
    preparation = prepare_pipeline(config)
    result = run_smina_and_postprocess(config)
    result["preparation_qc"] = preparation
    return result
