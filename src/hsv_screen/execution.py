from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def smina_binary() -> Path:
    adjacent = Path(sys.executable).resolve().parent / "smina"
    candidate = adjacent if adjacent.is_file() else Path(shutil.which("smina") or "")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"smina was not found next to the current Python ({adjacent}) or on PATH")
    return candidate.resolve()


def run_smina_job(binary: Path, row: dict[str, str]) -> str:
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
