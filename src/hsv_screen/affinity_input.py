from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from .config import resolve_path
from .io_utils import atomic_csv, atomic_json, prepare_stage_dir


FIELDS = [
    "pose_id", "parent_structure_key", "compound_id", "source_pool",
    "selection_channel", "state_id", "state_smiles", "protocol_id", "receptor_id",
    "box_id", "mutant_id", "pose_rank", "receptor_pdb", "ligand_pose_sdf",
    "ligand_record_index", "smina_score", "docking_percentile",
    "postdock_3d_status", "postdock_3d_similarity", "postdock_3d_feature_score",
    "postdock_3d_shape_similarity", "postdock_o3a_score",
    "pnu_ecfp_similarity", "pnu_fcfp_similarity", "pharm2d_similarity",
    "pnu_consensus", "quality_score", "library_evidence_score",
    "structure3d_score", "structure3d_feature_score", "structure3d_shape_similarity",
    "structure3d_clash_fraction", "qed", "mw", "clogp", "tpsa", "hbd", "hba",
    "rotatable_bonds", "heavy_atoms", "formal_charge", "fraction_csp3",
    "pains_alert", "brenk_alert", "soft_alert_count", "predicted_paffinity",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + ".partial")
    shutil.copyfile(source, partial)
    os.replace(partial, destination)


def _docking_percentiles(rows: list[dict]) -> dict[str, float]:
    """Rank Smina scores within each protocol; more negative is better."""
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["protocol_id"]].append((row["pose_id"], float(row["smina_score"])))
    result: dict[str, float] = {}
    for values in grouped.values():
        ordered = sorted(values, key=lambda item: item[1])
        size = len(ordered)
        positions: dict[float, list[int]] = defaultdict(list)
        for position, (_, score) in enumerate(ordered):
            positions[score].append(position)
        for pose_id, score in ordered:
            average_position = sum(positions[score]) / len(positions[score])
            result[pose_id] = 1.0 if size == 1 else 1.0 - average_position / (size - 1)
    return result


def run(config: dict, force: bool = False) -> dict:
    root = resolve_path(config, config["output_dir"])
    stage = root / "10_affinity_input"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)

    parents_path = root / "05_final_selection" / "selected.csv"
    states_path = root / "06_ligand_states" / "ligand_states.csv"
    docking_path = root / "08_docking_results" / "docking_poses.csv"
    similarity_path = root / "09_postdock_3d" / "pose_3d_similarity.csv"
    for path in (parents_path, states_path, docking_path, similarity_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing affinity-package input: {path}")

    parents = _read_csv(parents_path)
    states = _read_csv(states_path)
    docking = _read_csv(docking_path)
    similarities = _read_csv(similarity_path)
    if not docking:
        raise ValueError("No docked poses are available for the affinity package")

    parent_by_key = {row["structure_key"]: row for row in parents}
    state_by_id = {row["state_id"]: row for row in states}
    similarity_by_pose = {row["pose_id"]: row for row in similarities}
    if len(parent_by_key) != len(parents) or len(state_by_id) != len(states):
        raise ValueError("Duplicate parent or ligand-state identifiers")
    if len(similarity_by_pose) != len(similarities):
        raise ValueError("Duplicate pose_id in post-docking 3D scores")

    pose_ids = [row["pose_id"] for row in docking]
    if len(pose_ids) != len(set(pose_ids)):
        raise ValueError("Duplicate pose_id in docking results")
    missing_similarity = set(pose_ids) - set(similarity_by_pose)
    if missing_similarity:
        raise ValueError(
            f"{len(missing_similarity)} docked poses lack post-docking scores; "
            f"example: {sorted(missing_similarity)[0]}")

    states_by_parent: dict[str, set[str]] = defaultdict(set)
    for row in docking:
        states_by_parent[row["parent_structure_key"]].add(row["state_id"])
    invalid_state_counts = {
        key: sorted(values) for key, values in states_by_parent.items() if len(values) != 1}
    if invalid_state_counts:
        raise ValueError(
            f"Affinity package requires exactly one ligand state per parent; "
            f"found {len(invalid_state_counts)} violations")

    receptor_sources = {Path(row["receptor_pdb"]).resolve() for row in docking}
    ligand_sources = {Path(row["ligand_pose_sdf"]).resolve() for row in docking}
    if len(receptor_sources) != 1 or len(ligand_sources) != 1:
        raise ValueError("Affinity package expects one WT receptor and one consolidated pose SDF")
    receptor_source = next(iter(receptor_sources))
    ligand_source = next(iter(ligand_sources))
    if not receptor_source.is_file() or not ligand_source.is_file():
        raise FileNotFoundError("Docking result structure paths are missing")

    receptor_copy = stage / "receptor" / receptor_source.name
    ligand_copy = stage / "ligands" / "docked_poses.sdf"
    _copy_atomic(receptor_source, receptor_copy)
    _copy_atomic(ligand_source, ligand_copy)
    sdf_count = sum(
        mol is not None for mol in Chem.SDMolSupplier(str(ligand_copy), removeHs=False))
    if sdf_count != len(docking):
        raise ValueError(
            f"Affinity ligand SDF contains {sdf_count} valid records, expected {len(docking)}")

    percentiles = _docking_percentiles(docking)
    rows = []
    for pose in docking:
        parent = parent_by_key.get(pose["parent_structure_key"])
        state = state_by_id.get(pose["state_id"])
        if parent is None or state is None:
            raise ValueError(f"Untraceable docking pose {pose['pose_id']}")
        similarity = similarity_by_pose[pose["pose_id"]]
        row = {
            **parent,
            **pose,
            **similarity,
            "parent_structure_key": pose["parent_structure_key"],
            "state_smiles": state["state_smiles"],
            "receptor_pdb": str(receptor_copy.resolve()),
            "ligand_pose_sdf": str(ligand_copy.resolve()),
            "docking_percentile": percentiles[pose["pose_id"]],
            "predicted_paffinity": "",
        }
        rows.append(row)

    manifest = stage / "affinity_predictions.csv"
    atomic_csv(manifest, rows, FIELDS)
    readme = stage / "README.md"
    readme.write_text(
        "# Affinity model handoff\n\n"
        "This directory is a self-contained, post-docking input package for an external "
        "PDBbind-trained affinity model. The receptor is fixed at "
        "`receptor/8V1Q_WT_UL30_DNA_no_water.pdb`; ligand coordinates and bond orders are "
        "stored in the multi-record `ligands/docked_poses.sdf`.\n\n"
        "Use `ligand_record_index` (1-based) to select the SDF record paired with "
        "`receptor_pdb`. All non-affinity descriptors and screening scores are already "
        "populated in `affinity_predictions.csv`. Fill only `predicted_paffinity`; do not "
        "change `pose_id` or reorder/delete records unless the configured 95% pose and parent "
        "coverage gates remain satisfied. PDBQT is a docking-engine input and is not an "
        "affinity-model structure input.\n",
        encoding="utf-8",
    )
    summary = {
        "stage": "affinity_input", "pose_count": len(rows),
        "parent_count": len(states_by_parent), "states_per_parent": 1,
        "receptor_pdb": str(receptor_copy), "ligand_pose_sdf": str(ligand_copy),
        "prediction_table": str(manifest), "prediction_column": "predicted_paffinity",
        "prediction_values_initialized_blank": True,
        "non_affinity_scores_complete": True,
    }
    atomic_json(done, summary)
    return summary
