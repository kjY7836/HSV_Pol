from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import zlib
from collections import Counter
from pathlib import Path

from rdkit import Chem

from .config import resolve_path
from .io_utils import atomic_csv, atomic_json, prepare_stage_dir


JOB_FIELDS = [
    "protocol_id", "receptor_id", "box_id", "mutant_id", "chunk", "ligands",
    "receptor", "output", "log", "center_x", "center_y", "center_z", "size_x",
    "size_y", "size_z", "cpu", "exhaustiveness", "num_modes",
    "seed",
]
POSE_FIELDS = [
    "pose_id", "parent_structure_key", "state_id", "protocol_id", "receptor_id",
    "box_id", "mutant_id", "pose_rank", "smina_score",
    "receptor_pdb", "ligand_pose_sdf", "ligand_record_index",
    "source_docked_sdf", "source_record_index",
]


def _scope_accepts(mol: Chem.Mol, protocol: dict, selected_parents: set[str]) -> bool:
    parent = mol.GetProp("parent_structure_key") if mol.HasProp("parent_structure_key") else ""
    channel = mol.GetProp("selection_channel") if mol.HasProp("selection_channel") else ""
    if not parent or not channel:
        raise ValueError("Ligand state lacks parent_structure_key or selection_channel")
    allowed = set(protocol["scope"]["selection_channels"])
    if "*" not in allowed and channel not in allowed:
        return False
    limit = int(protocol["scope"].get("max_parents", 0) or 0)
    if parent not in selected_parents and limit and len(selected_parents) >= limit:
        return False
    selected_parents.add(parent)
    return True


def _write_protocol_chunks(root_states: Path, stage: Path, protocol: dict,
                           chunk_size: int) -> tuple[list[dict], dict]:
    protocol_id = protocol["protocol_id"]
    chunks_dir = stage / "ligand_chunks" / protocol_id
    outputs_dir = stage / "outputs" / protocol_id
    logs_dir = stage / "logs" / protocol_id
    chunks_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, Chem.SDWriter] = {}
    selected_parents: set[str] = set()
    state_count = 0
    try:
        for mol in Chem.SDMolSupplier(str(root_states), removeHs=False):
            if mol is None:
                raise ValueError(f"Invalid SDF record while preparing {protocol_id}")
            if not _scope_accepts(mol, protocol, selected_parents):
                continue
            mol.SetProp("_Name", mol.GetProp("state_id"))
            state_count += 1
            chunk = (state_count - 1) // chunk_size + 1
            if chunk not in writers:
                writers[chunk] = Chem.SDWriter(str(chunks_dir / f"ligands_{chunk:05d}.sdf"))
            writers[chunk].write(mol)
    finally:
        for writer in writers.values():
            writer.close()
    if state_count == 0:
        raise AssertionError(f"Docking protocol {protocol_id} selected no ligand states")

    center = [float(value) for value in protocol["box"]["center"]]
    size = [float(value) for value in protocol["box"]["size"]]
    jobs = []
    receptor = resolve_path({"_project_dir": protocol["_project_dir"]}, protocol["receptor_pdbqt"])
    for chunk in range(1, len(writers) + 1):
        jobs.append({
            "protocol_id": protocol_id, "receptor_id": protocol["receptor_id"],
            "box_id": protocol["box_id"], "mutant_id": protocol.get("mutant_id", "WT"),
            "chunk": chunk, "ligands": chunks_dir / f"ligands_{chunk:05d}.sdf",
            "receptor": receptor, "output": outputs_dir / f"docked_{chunk:05d}.sdf",
            "log": logs_dir / f"smina_{chunk:05d}.log",
            "center_x": center[0], "center_y": center[1], "center_z": center[2],
            "size_x": size[0], "size_y": size[1], "size_z": size[2],
            "cpu": int(protocol["_cpu_per_job"]),
            "exhaustiveness": int(protocol["exhaustiveness"]),
            "num_modes": int(protocol["num_modes"]),
            "seed": (int(protocol["_seed"]) + zlib.crc32(protocol_id.encode()) + chunk) % 2147483647,
        })
    return jobs, {
        "protocol_id": protocol_id, "parent_count": len(selected_parents),
        "state_count": state_count, "chunk_count": len(writers),
        "scope": protocol["scope"], "box": protocol["box"],
    }


def prepare(config: dict, force: bool = False) -> dict:
    root = resolve_path(config, config["output_dir"])
    stage = root / "07_smina"
    done = stage / "manifest.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    root_states = root / "06_ligand_states" / "ligand_states.sdf"
    if not root_states.exists():
        raise FileNotFoundError(f"Missing ligand-state SDF: {root_states}")
    input_state_count = sum(1 for mol in Chem.SDMolSupplier(str(root_states), removeHs=False) if mol is not None)
    settings = config["docking"]
    enabled = [dict(protocol) for protocol in settings["protocols"] if protocol.get("enabled", False)]
    if not enabled:
        raise ValueError("No docking protocol is enabled")

    all_jobs: list[dict] = []
    protocol_stats = []
    for protocol in enabled:
        protocol["_project_dir"] = config["_project_dir"]
        protocol["_cpu_per_job"] = settings["cpu_per_job"]
        protocol["_seed"] = config["seed"]
        receptor = resolve_path(config, protocol["receptor_pdbqt"])
        if not receptor.exists():
            raise FileNotFoundError(f"Enabled protocol {protocol['protocol_id']} lacks receptor: {receptor}")
        jobs, stats = _write_protocol_chunks(
            root_states, stage, protocol, int(settings["chunk_size"]))
        all_jobs.extend(jobs)
        stats["receptor_pdbqt"] = str(receptor)
        protocol_stats.append(stats)

    jobs_path = stage / "jobs.tsv"
    with jobs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=JOB_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_jobs)

    required = set(settings.get("production_required_protocols", []))
    enabled_ids = {protocol["protocol_id"] for protocol in enabled}
    production_ready = required.issubset(enabled_ids)
    adjacent_engine = Path(sys.executable).resolve().parent / settings["engine"]
    engine_path = str(adjacent_engine) if adjacent_engine.exists() else shutil.which(settings["engine"])
    manifest = {
        "stage": "smina_preparation", "input_state_count": input_state_count,
        "scheduled_state_count": sum(item["state_count"] for item in protocol_stats),
        "job_count": len(all_jobs),
        "protocols": protocol_stats, "jobs": str(jobs_path),
        "smina_available": bool(engine_path),
        "smina_path": engine_path, "production_ready": production_ready,
        "missing_required_protocols": sorted(required - enabled_ids),
        "production_gate": "Only the required 8V1Q WT protocol may be enabled; its explicit local box and receptor files must be present.",
    }
    atomic_json(done, manifest)
    return manifest


def _affinity_property(mol: Chem.Mol) -> float:
    for name in ("minimizedAffinity", "smina_affinity", "affinity"):
        if mol.HasProp(name):
            return float(mol.GetProp(name))
    names = list(mol.GetPropNames())
    raise ValueError(f"Docked pose lacks a recognized Smina affinity property; found {names}")


def collect(config: dict, force: bool = False) -> dict:
    root = resolve_path(config, config["output_dir"])
    stage = root / "08_docking_results"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    protocols = {item["protocol_id"]: item for item in config["docking"]["protocols"] if item.get("enabled")}
    states_path = root / "06_ligand_states" / "ligand_states.csv"
    with states_path.open(encoding="utf-8", newline="") as handle:
        state_rows = list(csv.DictReader(handle))
    state_metadata = {row["state_id"]: row for row in state_rows if row.get("embed_status", "").startswith("ok")}
    if len(state_metadata) != sum(row.get("embed_status", "").startswith("ok") for row in state_rows):
        raise ValueError("Ligand-state IDs are not unique")
    jobs_path = root / "07_smina" / "jobs.tsv"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing Smina job manifest: {jobs_path}")
    with jobs_path.open(encoding="utf-8", newline="") as handle:
        expected_jobs = list(csv.DictReader(handle, delimiter="\t"))
    expected_output_paths = [row.get("output", "") for row in expected_jobs]
    if not expected_output_paths or any(not path for path in expected_output_paths):
        raise ValueError("Smina job manifest has no valid output paths")
    if len(expected_output_paths) != len(set(expected_output_paths)):
        raise ValueError("Smina job manifest contains duplicate output paths")
    jobs_by_protocol: dict[str, list[Path]] = {protocol_id: [] for protocol_id in protocols}
    for row in expected_jobs:
        protocol_id = row.get("protocol_id", "")
        if protocol_id not in protocols:
            raise ValueError(f"Smina job manifest contains unknown protocol {protocol_id}")
        jobs_by_protocol[protocol_id].append(Path(row["output"]))
    empty_protocols = [protocol_id for protocol_id, paths in jobs_by_protocol.items() if not paths]
    if empty_protocols:
        raise ValueError(f"Enabled protocols have no Smina jobs: {empty_protocols}")
    missing_outputs = [row["output"] for row in expected_jobs if not Path(row["output"]).exists()]
    if missing_outputs:
        raise FileNotFoundError(
            f"{len(missing_outputs)} expected Smina outputs are missing; first: {missing_outputs[0]}")
    expected_states: dict[str, set[str]] = {protocol_id: set() for protocol_id in protocols}
    for row in expected_jobs:
        ligand_path = Path(row.get("ligands", ""))
        if not ligand_path.is_file():
            raise FileNotFoundError(f"Smina input chunk is missing: {ligand_path}")
        protocol_id = row["protocol_id"]
        for record_index, mol in enumerate(
                Chem.SDMolSupplier(str(ligand_path), removeHs=False), start=1):
            if mol is None:
                raise ValueError(
                    f"Invalid Smina input record {record_index} in {ligand_path}")
            state_id = (mol.GetProp("state_id") if mol.HasProp("state_id")
                        else (mol.GetProp("_Name") if mol.HasProp("_Name") else ""))
            if not state_id or state_id not in state_metadata:
                raise ValueError(
                    f"Untraceable Smina input state {state_id!r} in {ligand_path}")
            if state_id in expected_states[protocol_id]:
                raise ValueError(
                    f"State {state_id} is scheduled more than once for protocol {protocol_id}")
            expected_states[protocol_id].add(state_id)
    required_protocols = set(config["docking"].get("production_required_protocols", []))
    for protocol_id in required_protocols:
        missing_from_schedule = set(state_metadata) - expected_states.get(protocol_id, set())
        if missing_from_schedule:
            raise ValueError(
                f"Required protocol {protocol_id} did not schedule "
                f"{len(missing_from_schedule)} embedded states; "
                f"first={sorted(missing_from_schedule)[0]}")
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    observed_states: dict[str, set[str]] = {protocol_id: set() for protocol_id in protocols}
    pose_counts: dict[str, Counter[str]] = {
        protocol_id: Counter() for protocol_id in protocols}
    consolidated = stage / "docked_poses.sdf"
    consolidated_partial = Path(str(consolidated) + ".partial")
    pose_writer = Chem.SDWriter(str(consolidated_partial))
    ligand_record_index = 0
    try:
        for protocol_id, protocol in protocols.items():
            # Read exactly the files declared by jobs.tsv.  Untracked stale SDF
            # files must never leak into a new affinity input set.
            files = sorted(jobs_by_protocol[protocol_id])
            for path in files:
                for source_record_index, mol in enumerate(
                        Chem.SDMolSupplier(str(path), removeHs=False), start=1):
                    if mol is None:
                        raise ValueError(f"Invalid docked pose in {path}")
                    state_id = (mol.GetProp("state_id") if mol.HasProp("state_id")
                                else (mol.GetProp("_Name") if mol.HasProp("_Name") else ""))
                    metadata = state_metadata.get(state_id)
                    parent = (mol.GetProp("parent_structure_key") if mol.HasProp("parent_structure_key")
                              else (metadata or {}).get("parent_structure_key", ""))
                    if not state_id or not parent:
                        raise ValueError(f"Docked pose in {path} lacks state/parent traceability")
                    if state_id not in expected_states[protocol_id]:
                        raise ValueError(
                            f"Smina output contains unscheduled state {state_id} for {protocol_id}")
                    observed_states[protocol_id].add(state_id)
                    pose_counts[protocol_id][state_id] += 1
                    pose_rank = pose_counts[protocol_id][state_id]
                    pose_id = f"{protocol_id}|{state_id}|{pose_rank}"
                    ligand_record_index += 1
                    receptor_pdb = str(resolve_path(config, protocol["receptor_pdb"]))
                    for name, value in {
                        "pose_id": pose_id, "parent_structure_key": parent,
                        "state_id": state_id, "protocol_id": protocol_id,
                        "receptor_id": protocol["receptor_id"], "box_id": protocol["box_id"],
                        "mutant_id": protocol.get("mutant_id", "WT"),
                        "receptor_pdb": receptor_pdb,
                        "compound_id": (metadata or {}).get("compound_id", ""),
                        "source_pool": (metadata or {}).get("source_pool", ""),
                        "selection_channel": (metadata or {}).get("selection_channel", ""),
                        "state_smiles": (metadata or {}).get("state_smiles", ""),
                    }.items():
                        mol.SetProp(name, str(value))
                    mol.SetProp("_Name", pose_id)
                    pose_writer.write(mol)
                    rows.append({
                        "pose_id": pose_id, "parent_structure_key": parent, "state_id": state_id,
                        "protocol_id": protocol_id, "receptor_id": protocol["receptor_id"],
                        "box_id": protocol["box_id"], "mutant_id": protocol.get("mutant_id", "WT"),
                        "pose_rank": pose_rank, "smina_score": _affinity_property(mol),
                        "receptor_pdb": receptor_pdb, "ligand_pose_sdf": str(consolidated),
                        "ligand_record_index": ligand_record_index,
                        "source_docked_sdf": str(path), "source_record_index": source_record_index,
                    })
                    counts[protocol_id] += 1
    finally:
        pose_writer.close()
    missing_states = {
        protocol_id: sorted(expected_states[protocol_id] - observed_states[protocol_id])
        for protocol_id in protocols
        if expected_states[protocol_id] - observed_states[protocol_id]
    }
    if missing_states:
        consolidated_partial.unlink(missing_ok=True)
        first_protocol = sorted(missing_states)[0]
        raise ValueError(
            f"Smina produced no pose for {sum(map(len, missing_states.values()))} scheduled "
            f"states; first={first_protocol}:{missing_states[first_protocol][0]}")
    if not rows:
        consolidated_partial.unlink(missing_ok=True)
        raise FileNotFoundError("No completed Smina output SDF files were found")
    os.replace(consolidated_partial, consolidated)
    output = stage / "docking_poses.csv"
    atomic_csv(output, rows, POSE_FIELDS)
    summary = {
        "stage": "collect_docking", "pose_count": len(rows),
        "completed_job_count": len(expected_jobs), "protocol_pose_counts": dict(counts),
        "scheduled_state_counts": {
            protocol_id: len(states) for protocol_id, states in expected_states.items()},
        "observed_state_counts": {
            protocol_id: len(states) for protocol_id, states in observed_states.items()},
        "all_scheduled_states_have_poses": True,
        "docking_poses": str(output),
        "consolidated_pose_sdf": str(consolidated),
        "affinity_handoff": "Run prepare-affinity after postdock3d to create the standalone receptor/ligand structure package and blank prediction table.",
    }
    atomic_json(done, summary)
    return summary
