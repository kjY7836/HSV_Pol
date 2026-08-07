#!/usr/bin/env python3
"""Generate, dock, and affinity-score the five bundled demonstration ligands."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_affinity import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    predict,
    write_rows,
)


RECEPTOR_PDB = ROOT / "receptors" / "8V1Q_WT_UL30_DNA_no_water.pdb"
RECEPTOR_PDBQT = ROOT / "receptors" / "8V1Q_WT_UL30_DNA_no_water.pdbqt"
EXAMPLE_CSV = ROOT / "examples" / "example_smiles.csv"


def find_smina(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    adjacent = Path(sys.executable).resolve().parent / "smina"
    candidates.append(adjacent)
    discovered = shutil.which("smina")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "smina was not found beside the active Python executable or on PATH; "
        "pass --smina /absolute/path/to/smina"
    )


def build_example_sdf(csv_path: Path, output: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise ValueError(f"The demonstration requires exactly five ligands, found {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    try:
        for index, row in enumerate(rows):
            ligand_id = row["ligand_id"].strip()
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError(f"Invalid SMILES for {ligand_id}")
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 20260806 + index
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise RuntimeError(f"3D embedding failed for {ligand_id}")
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            mol.SetProp("_Name", ligand_id)
            mol.SetProp("compound_id", ligand_id)
            mol.SetProp("example_name", row["name"])
            mol.SetProp("source_smiles", row["smiles"])
            writer.write(mol)
    finally:
        writer.close()
    return rows


def write_best_per_ligand(
    path: Path,
    predictions: list[dict],
    input_rows: list[dict[str, str]],
) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["ligand_id"])].append(row)
    input_by_id = {row["ligand_id"]: row for row in input_rows}
    if set(grouped) != set(input_by_id):
        raise ValueError(
            f"Smina ligand IDs differ from the five inputs: "
            f"observed={sorted(grouped)}, expected={sorted(input_by_id)}"
        )
    summary_rows: list[dict] = []
    for ligand_id in input_by_id:
        poses = grouped[ligand_id]
        best_affinity = max(poses, key=lambda row: float(row["predicted_paffinity"]))
        scored = [row for row in poses if row["smina_score"] is not None]
        best_smina = min(scored, key=lambda row: float(row["smina_score"])) if scored else None
        summary_rows.append(
            {
                "ligand_id": ligand_id,
                "name": input_by_id[ligand_id]["name"],
                "pose_count": len(poses),
                "best_predicted_paffinity": best_affinity["predicted_paffinity"],
                "best_affinity_pose_record": best_affinity["ligand_record_index"],
                "best_affinity_pose_smina_score": best_affinity["smina_score"],
                "best_smina_score": best_smina["smina_score"] if best_smina else None,
                "best_smina_pose_predicted_paffinity": (
                    best_smina["predicted_paffinity"] if best_smina else None
                ),
            }
        )
    write_rows(path, summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bundled five-ligand Smina -> 3DMPG demonstration."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "results")
    parser.add_argument("--smina", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=1, help="CPU count passed to Smina.")
    parser.add_argument("--exhaustiveness", type=int, default=12)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (RECEPTOR_PDB, RECEPTOR_PDBQT, EXAMPLE_CSV, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    smina = find_smina(args.smina)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_sdf = args.output_dir / "five_examples_3d.sdf"
    docked_sdf = args.output_dir / "five_examples_smina.sdf"
    smina_log = args.output_dir / "smina.log"
    predictions_csv = args.output_dir / "pose_affinity_predictions.csv"
    best_csv = args.output_dir / "best_per_ligand.csv"
    summary_json = args.output_dir / "run_summary.json"

    input_rows = build_example_sdf(EXAMPLE_CSV, input_sdf)
    command = [
        str(smina),
        "--receptor",
        str(RECEPTOR_PDBQT),
        "--ligand",
        str(input_sdf),
        "--center_x",
        "147.660",
        "--center_y",
        "145.083",
        "--center_z",
        "124.388",
        "--size_x",
        "24",
        "--size_y",
        "24",
        "--size_z",
        "28",
        "--cpu",
        str(args.cpu),
        "--exhaustiveness",
        str(args.exhaustiveness),
        "--num_modes",
        str(args.num_modes),
        "--seed",
        str(args.seed),
        "--out",
        str(docked_sdf),
        "--log",
        str(smina_log),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    predictions, model_summary = predict(
        RECEPTOR_PDB,
        docked_sdf,
        args.checkpoint,
        device_name=args.device,
        pocket_radius=10.0,
        batch_size=1,
        threads=args.threads,
    )
    write_rows(predictions_csv, predictions)
    write_best_per_ligand(best_csv, predictions, input_rows)
    summary = {
        "example_count": 5,
        "pose_count": len(predictions),
        "smina_command": command,
        "smina_log": str(smina_log.resolve()),
        "docked_sdf": str(docked_sdf.resolve()),
        "pose_predictions": str(predictions_csv.resolve()),
        "best_per_ligand": str(best_csv.resolve()),
        "model_run": model_summary,
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
