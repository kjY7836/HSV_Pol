from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem

from .config import resolve_path
from .io_utils import atomic_csv, atomic_json, prepare_stage_dir
from .screen3d import (
    build_feature_model,
    init_worker,
    parse_pocket_atoms,
    score_conformation_similarity,
)


FIELDS = [
    "pose_id", "postdock_3d_status", "postdock_3d_similarity",
    "postdock_3d_feature_score", "postdock_3d_shape_similarity", "postdock_o3a_score",
]


def _score_pose(task: tuple[int, str, str]) -> tuple[int, dict]:
    index, pose_id, mol_block = task
    mol = Chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=False)
    if mol is None or mol.GetNumConformers() != 1:
        return index, {
            "pose_id": pose_id, "postdock_3d_status": "invalid_pose",
            "postdock_3d_similarity": 0.0, "postdock_3d_feature_score": 0.0,
            "postdock_3d_shape_similarity": 0.0, "postdock_o3a_score": 0.0,
        }
    combined, feature, shape, o3a = score_conformation_similarity(mol, 0)
    return index, {
        "pose_id": pose_id,
        "postdock_3d_status": "ok" if combined >= 0 else "alignment_failed",
        "postdock_3d_similarity": max(0.0, combined),
        "postdock_3d_feature_score": feature,
        "postdock_3d_shape_similarity": shape,
        "postdock_o3a_score": o3a,
    }


def run(config: dict, force: bool = False) -> dict:
    root = resolve_path(config, config["output_dir"])
    stage = root / "09_postdock_3d"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)

    poses_path = root / "08_docking_results" / "docked_poses.sdf"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing consolidated docked-pose SDF: {poses_path}")
    refs = config["references"]
    reference = Chem.SDMolSupplier(
        str(resolve_path(config, refs["bound_ligand_sdf"])), removeHs=False)[0]
    if reference is None:
        raise ValueError("Experimental PNU reference cannot be parsed")
    settings = config["structure_3d"]
    pocket = parse_pocket_atoms(
        resolve_path(config, refs["pdb_file"]), refs["ligand_resname"], reference,
        float(settings["pocket_radius"]))
    features = build_feature_model(reference, pocket, settings)
    reference_block = Chem.MolToMolBlock(reference)
    init_worker(reference_block, pocket, features, settings)
    self_score = score_conformation_similarity(Chem.Mol(reference), 0)
    if self_score[0] < 0.99 or self_score[1] < 0.99 or self_score[2] < 0.99:
        raise AssertionError(f"Post-docking PNU 3D self-check failed: {self_score}")

    tasks = []
    seen: set[str] = set()
    for index, mol in enumerate(Chem.SDMolSupplier(str(poses_path), removeHs=False)):
        if mol is None:
            raise ValueError(f"Invalid pose record {index + 1} in {poses_path}")
        pose_id = mol.GetProp("pose_id") if mol.HasProp("pose_id") else ""
        if not pose_id or pose_id in seen:
            raise ValueError(f"Missing or duplicate pose_id at record {index + 1}: {pose_id}")
        seen.add(pose_id)
        tasks.append((index, pose_id, Chem.MolToMolBlock(mol)))
    if not tasks:
        raise ValueError("No docked poses are available for post-docking 3D similarity")

    started = time.time()
    rows: list[dict | None] = [None] * len(tasks)
    statuses: Counter[str] = Counter()
    with mp.Pool(
            int(config["workers"]), initializer=init_worker,
            initargs=(reference_block, pocket, features, settings)) as pool:
        for index, result in pool.imap(_score_pose, tasks, chunksize=32):
            rows[index] = result
            statuses[result["postdock_3d_status"]] += 1
    output_rows = [row for row in rows if row is not None]
    success = statuses.get("ok", 0)
    coverage = success / len(tasks)
    minimum = float(config["postdock_3d"]["minimum_pose_coverage"])
    if coverage < minimum:
        raise ValueError(
            f"Post-docking 3D similarity coverage {coverage:.3f} is below required {minimum:.3f}")
    output = stage / "pose_3d_similarity.csv"
    atomic_csv(output, output_rows, FIELDS)
    summary = {
        "stage": "postdock_3d_similarity", "input_pose_count": len(tasks),
        "statuses": dict(statuses), "pose_coverage": coverage,
        "reference": str(resolve_path(config, refs["bound_ligand_sdf"])),
        "method": "Docked conformation is Crippen-O3A aligned to experimental PNU; score is 0.60 pharmacophore-feature match + 0.40 shape similarity. The 8V1Q placement is not replaced or re-docked.",
        "pnu_self_check": {
            "combined": self_score[0], "feature": self_score[1],
            "shape": self_score[2], "o3a": self_score[3],
        },
        "output": str(output), "elapsed_seconds": time.time() - started,
    }
    atomic_json(done, summary)
    return summary
