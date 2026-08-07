from __future__ import annotations

import csv
import gzip
import json
import multiprocessing as mp
import os
import platform
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from rdkit import Chem, rdBase

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
from .chemistry import score_reference_self
from .config import resolve_path
from .distributed_runtime import RankContext
from .execution import run_smina_job, smina_binary
from .io_utils import atomic_csv, atomic_json, iter_csv, prepare_stage_dir
from .reference import PNU_SMILES, validate_and_prepare


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _distributed_metadata(config: dict, include_total: bool = False) -> dict:
    settings = config["distributed"]
    nodes = int(settings["nodes"])
    workers = int(settings["workers_per_node"])
    result = {"nodes": nodes, "workers_per_node": workers}
    if include_total:
        result["total_workers"] = nodes * workers
    return result


def _root_step(context: RankContext, name: str, function: Callable[[], object]) -> None:
    if context.is_root:
        context.log(f"start {name}")
        function()
        context.log(f"complete {name}")
    context.barrier(name)


def _prepare_distributed_stage(
    context: RankContext, stage_name: str, stage: Path, done: Path,
) -> bool:
    decision = context.root_value(f"{stage_name}_decision", lambda: {
        "skip": done.exists(), "stage": str(stage),
    })
    if bool(decision["skip"]):
        context.log(f"skip completed stage {stage_name}")
        context.barrier(f"{stage_name}_skipped")
        return False
    if context.is_root:
        prepare_stage_dir(stage, force=False)
        (stage / "_distributed").mkdir(parents=True, exist_ok=True)
    context.barrier(f"{stage_name}_prepared")
    return True


def _write_gzip_rows(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8", newline="", compresslevel=3) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _standardization_plan(config: dict, context: RankContext, stage: Path) -> dict:
    data_dir = resolve_path(config, config["data_dir"])
    active, candidates = standardize.discover_sources(data_dir)
    sample = config["sample"]
    tasks = []
    if sample["enabled"]:
        active_count = int(sample["active_records"])
        total_candidate = int(sample["candidate_records"])
        base, residue = divmod(total_candidate, len(candidates))
    else:
        active_count = 0
        base = residue = 0
    tasks.append({
        "source_index": 0, "path": str(active), "source_pool": "activity_recorded",
        "sample_count": active_count, "assigned_rank": context.size - 1,
        "raw_output": str(stage / "_distributed" / "standardized_source_000.csv.gz"),
    })
    for index, path in enumerate(candidates, start=1):
        tasks.append({
            "source_index": index, "candidate_index": index, "path": str(path),
            "source_pool": "unlabeled",
            "sample_count": base + int(index - 1 < residue) if sample["enabled"] else 0,
            "assigned_rank": (index - 1) % context.size,
            "raw_output": str(stage / "_distributed" / f"standardized_source_{index:03d}.csv.gz"),
        })
    return {
        "data_dir": str(data_dir), "activity_file": str(active),
        "candidate_files": [str(path) for path in candidates], "tasks": tasks,
        "sample": sample, "started": time.time(),
    }


def _standardize_worker(config: dict, context: RankContext, plan: dict) -> None:
    data_dir = Path(plan["data_dir"])
    for task in plan["tasks"]:
        if int(task["assigned_rank"]) != context.rank:
            continue
        path = Path(task["path"])
        records: Iterable[dict] = standardize.iter_source(path, data_dir, task["source_pool"])
        count = int(task.get("sample_count", 0))
        if bool(plan["sample"]["enabled"]):
            seed = int(config["seed"]) if task["source_pool"] == "activity_recorded" else (
                int(config["seed"]) + int(task["candidate_index"]))
            records = standardize.minhash_sample(records, count, seed)
        output = Path(task["raw_output"])
        partial = Path(str(output) + ".partial")
        processed = 0
        with gzip.open(partial, "wt", encoding="utf-8", newline="", compresslevel=3) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=standardize.MAPPING_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for result in standardize._process_records(
                    records, config["standardization"], int(config["workers"])):
                writer.writerow(result)
                processed += 1
                if processed % 100000 == 0:
                    context.log(f"standardized {processed:,} rows from {path.name}")
        os.replace(partial, output)
        context.log(f"standardization shard complete: {path.name} ({processed:,})")


def _merge_standardized(config: dict, plan: dict, stage: Path) -> None:
    mapping_partial, mapping_handle, mapping_writer = standardize._open_writer(
        stage / "parent_mapping.csv.gz", standardize.MAPPING_FIELDS)
    activity_partial, activity_handle, activity_writer = standardize._open_writer(
        stage / "activity_parents.csv.gz", standardize.PARENT_FIELDS)
    seen_exact: set[str] = set()
    active_exact: set[str] = set()
    active_connectivity: set[str] = set()
    counts: Counter[str] = Counter()
    try:
        for task in sorted(plan["tasks"], key=lambda item: int(item["source_index"])):
            pool_name = str(task["source_pool"])
            if pool_name == "activity_recorded":
                parent_writer = activity_writer
                output_handle = None
                output_partial = None
            else:
                output_path = stage / f"unlabeled_parents_{int(task['candidate_index']):02d}.csv.gz"
                output_partial, output_handle, parent_writer = standardize._open_writer(
                    output_path, standardize.PARENT_FIELDS)
            for result in iter_csv(Path(task["raw_output"])):
                counts["input_records"] += 1
                counts[f"status:{result['status']}"] += 1
                structure_key = result.get("structure_key", "")
                connectivity_key = result.get("connectivity_key", "")
                result["active_exact_overlap"] = int(bool(
                    structure_key and structure_key in active_exact))
                result["active_connectivity_overlap"] = int(bool(
                    connectivity_key and connectivity_key in active_connectivity))
                result["resolved_source_pool"] = pool_name
                if result["status"] != "valid_parent" or not _as_bool(result["model_eligible"]):
                    result["dedup_status"] = "not_in_parent_pool"
                elif structure_key in seen_exact:
                    result["dedup_status"] = "exact_duplicate"
                    if pool_name == "unlabeled" and structure_key in active_exact:
                        result["resolved_source_pool"] = "activity_recorded"
                        counts["candidate_exact_overlap_activity"] += 1
                else:
                    result["dedup_status"] = "unique_parent"
                    seen_exact.add(structure_key)
                    parent_writer.writerow(result)
                    counts[f"unique_parent:{pool_name}"] += 1
                    if pool_name == "activity_recorded":
                        active_exact.add(structure_key)
                        active_connectivity.add(connectivity_key)
                mapping_writer.writerow(result)
            if output_handle is not None and output_partial is not None:
                output_handle.close()
                os.replace(output_partial, stage / f"unlabeled_parents_{int(task['candidate_index']):02d}.csv.gz")
        activity_handle.close()
        mapping_handle.close()
        os.replace(activity_partial, stage / "activity_parents.csv.gz")
        os.replace(mapping_partial, stage / "parent_mapping.csv.gz")
    except BaseException:
        activity_handle.close()
        mapping_handle.close()
        raise
    summary = {
        "stage": "standardize", "data_dir": plan["data_dir"],
        "activity_file": plan["activity_file"], "candidate_files": plan["candidate_files"],
        "sample": plan["sample"], "counts": dict(counts),
        "exact_parent_count": len(seen_exact),
        "distributed": _distributed_metadata(config),
        "interpretation": {
            "activity_recorded_is_target_label": False,
            "component_policy": "Remove only explicitly recognized salt/solvent components when one unambiguous organic parent remains; quarantine ambiguous multi-organic and metal-containing records.",
            "dedup_policy": "Exact standardized isomeric structure only; stereoisomers remain distinct; activity-recorded copy wins exact cross-source overlap.",
        },
        "elapsed_seconds": time.time() - float(plan["started"]),
    }
    shutil.rmtree(stage / "_distributed")
    atomic_json(stage / "summary.json", summary)


def run_standardize(config: dict, context: RankContext) -> None:
    root = resolve_path(config, config["output_dir"])
    stage = root / "01_standardized"
    if not _prepare_distributed_stage(
            context, "standardize", stage, stage / "summary.json"):
        return
    plan = context.root_value(
        "standardization_plan", lambda: _standardization_plan(config, context, stage))
    _standardize_worker(config, context, plan)
    context.barrier("standardization_workers_complete")
    if context.is_root:
        _merge_standardized(config, plan, stage)
    context.barrier("standardization_merged")


def _score2d_plan(config: dict, stage: Path, context: RankContext) -> dict:
    root = resolve_path(config, config["output_dir"])
    source = root / "01_standardized"
    inputs = [source / "activity_parents.csv.gz", *sorted(source.glob("unlabeled_parents_*.csv.gz"))]
    if not inputs or not all(path.exists() for path in inputs):
        raise FileNotFoundError("Standardized parent shards are incomplete")
    tasks = []
    for index, path in enumerate(inputs):
        tasks.append({
            "input": str(path),
            "output": str(stage / path.name.replace("parents", "scored")),
            "assigned_rank": index % context.size,
        })
    return {"tasks": tasks, "started": time.time()}


def run_score2d(config: dict, context: RankContext) -> None:
    root = resolve_path(config, config["output_dir"])
    stage = root / "02_2d_scored"
    if not _prepare_distributed_stage(context, "score2d", stage, stage / "summary.json"):
        return
    plan = context.root_value("score2d_plan", lambda: _score2d_plan(config, stage, context))
    counters: Counter[str] = Counter()
    started = time.time()
    for task in plan["tasks"]:
        if int(task["assigned_rank"]) == context.rank:
            screen2d._score_file(
                Path(task["input"]), Path(task["output"]), config, counters, started)
    atomic_json(stage / "_distributed" / f"counts_rank_{context.rank:02d}.json", dict(counters))
    context.barrier("score2d_workers_complete")
    if context.is_root:
        totals: Counter[str] = Counter()
        for path in sorted((stage / "_distributed").glob("counts_rank_*.json")):
            totals.update(json.loads(path.read_text(encoding="utf-8")))
        self_score = score_reference_self(
            PNU_SMILES, config["fingerprint"], config["filters"],
            config["integrated_scoring"]["predocking"])
        for field in (
                "pnu_ecfp_similarity", "pnu_fcfp_similarity",
                "pharm2d_similarity", "pnu_consensus"):
            if abs(float(self_score[field]) - 1.0) > 1e-9:
                raise AssertionError(f"PNU self score failed for {field}: {self_score[field]}")
        summary = {
            "stage": "2d_scoring", "counts": dict(totals),
            "pnu_self_check": self_score,
            "formula": "0.55*ECFP4_Tanimoto + 0.20*FCFP4_Tanimoto + 0.25*Gobbi_Pharm2D_Tanimoto",
            "library_evidence_formula": "0.65*PNU_consensus + 0.35*quality_score; channel ranking only, not final scoring",
            "diversity_is_not_collapsed_into_scalar_score": True,
            "hard_filters_are_wide_gate": True, "pains_brenk_are_soft_flags": True,
            "distributed": _distributed_metadata(config),
            "elapsed_seconds": time.time() - float(plan["started"]),
        }
        shutil.rmtree(stage / "_distributed")
        atomic_json(stage / "summary.json", summary)
    context.barrier("score2d_merged")


def _three_d_reference(config: dict) -> tuple[Chem.Mol, list[list[float]], list[dict], str, tuple]:
    refs = config["references"]
    bound = Chem.SDMolSupplier(
        str(resolve_path(config, refs["bound_ligand_sdf"])), removeHs=False)[0]
    if bound is None:
        raise ValueError("Experimental PNU reference cannot be parsed")
    settings = config["structure_3d"]
    pocket = screen3d.parse_pocket_atoms(
        resolve_path(config, refs["pdb_file"]), refs["ligand_resname"], bound,
        float(settings["pocket_radius"]))
    features = screen3d.build_feature_model(bound, pocket, settings)
    reference_block = Chem.MolToMolBlock(bound)
    screen3d.init_worker(reference_block, pocket, features, settings)
    self_values = screen3d.score_conformer(Chem.Mol(bound), 0)
    if self_values[0] < 0.80 or self_values[1] < 0.95 or self_values[2] < 0.99:
        raise AssertionError(f"Experimental PNU 3D self-check failed: {self_values}")
    return bound, pocket, features, reference_block, self_values


def _score3d_plan(config: dict, stage: Path) -> dict:
    root = resolve_path(config, config["output_dir"])
    pool_size = int(config["prescreen_3d_pool"]["size"])
    input_path = root / "03_3d_pool" / f"pool_{pool_size}.csv.gz"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing 3D pool: {input_path}")
    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle:
        fields = list(csv.DictReader(handle).fieldnames or [])
    if not fields:
        raise ValueError(f"3D pool has no header: {input_path}")
    output_fields = list(fields)
    for field in screen3d.THREED_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    return {
        "input": str(input_path), "pool_size": pool_size,
        "input_fields": fields, "output_fields": output_fields,
        "started": time.time(),
    }


def _score3d_worker(config: dict, context: RankContext, stage: Path, plan: dict) -> None:
    _, pocket, features, reference_block, self_values = _three_d_reference(config)
    local_rows: dict[int, dict] = {}
    for index, row in enumerate(iter_csv(Path(plan["input"]))):
        if index % context.size == context.rank:
            local_rows[index] = dict(row)
    tasks = (
        (index, row["standardized_smiles"], row["structure_key"], row["prescreen_channel"])
        for index, row in local_rows.items()
    )
    statuses: Counter[str] = Counter()
    started = time.time()
    with mp.Pool(
            int(config["workers"]), initializer=screen3d.init_worker,
            initargs=(reference_block, pocket, features, config["structure_3d"])) as pool:
        for count, (index, result) in enumerate(
                pool.imap(screen3d.score_task, tasks, chunksize=8), start=1):
            local_rows[index].update(result)
            statuses[result["structure3d_status"]] += 1
            if count % 1000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                context.log(
                    f"3D scored {count:,}/{len(local_rows):,}; {count / elapsed:,.1f}/s")
    fields = ["_global_index", *plan["output_fields"]]
    rows = ({"_global_index": index, **local_rows[index]} for index in sorted(local_rows))
    _write_gzip_rows(
        stage / "_distributed" / f"scored_rank_{context.rank:02d}.csv.gz", rows, fields)
    atomic_json(stage / "_distributed" / f"summary_rank_{context.rank:02d}.json", {
        "rank": context.rank, "count": len(local_rows), "statuses": dict(statuses),
        "self_values": list(self_values), "elapsed_seconds": time.time() - started,
    })


def _merge_score3d(config: dict, stage: Path, plan: dict) -> None:
    pool_size = int(plan["pool_size"])
    rows: list[dict | None] = [None] * pool_size
    statuses: Counter[str] = Counter()
    rank_summaries = []
    for summary_path in sorted((stage / "_distributed").glob("summary_rank_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rank_summaries.append(summary)
        statuses.update(summary["statuses"])
    for shard in sorted((stage / "_distributed").glob("scored_rank_*.csv.gz")):
        for row in iter_csv(shard):
            index = int(row.pop("_global_index"))
            if index < 0 or index >= pool_size or rows[index] is not None:
                raise ValueError(f"Invalid or duplicate distributed 3D index {index}")
            rows[index] = row
    if any(row is None for row in rows):
        missing = [index for index, row in enumerate(rows) if row is None]
        raise ValueError(f"Distributed 3D scoring missed {len(missing)} rows; first={missing[0]}")
    output = stage / f"scored_{pool_size}.csv.gz"
    _write_gzip_rows(output, (row for row in rows if row is not None), plan["output_fields"])
    _, pocket, features, _, self_values = _three_d_reference(config)
    atomic_json(stage / "pharmacophore_features.json", {
        "features": features, "pocket_atom_count": len(pocket),
    })
    summary = {
        "stage": "structure_3d", "input_count": pool_size,
        "statuses": dict(statuses),
        "experimental_pnu_self_check": {
            "combined": self_values[0], "feature": self_values[1],
            "shape": self_values[2], "clash": self_values[3],
        },
        "method": "All channels receive ETKDGv3 feasibility checks. Only configured PNU/Pharm2D channels receive Crippen O3A plus experimental 7LUF YE4 feature/shape/clash scoring; diversity and exploration are not rejected for PNU mismatch.",
        "not_docking": True,
        "distributed": {
            **_distributed_metadata(config),
            "rank_counts": [item["count"] for item in rank_summaries],
        },
        "elapsed_seconds": time.time() - float(plan["started"]),
        "versions": {"python": platform.python_version(),
                     "rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
    }
    if statuses["ok"] == 0:
        raise AssertionError("No 3D candidate completed successfully")
    shutil.rmtree(stage / "_distributed")
    atomic_json(stage / "summary.json", summary)


def run_score3d(config: dict, context: RankContext) -> None:
    root = resolve_path(config, config["output_dir"])
    stage = root / "04_3d_scored"
    if not _prepare_distributed_stage(context, "score3d", stage, stage / "summary.json"):
        return
    plan = context.root_value("score3d_plan", lambda: _score3d_plan(config, stage))
    _score3d_worker(config, context, stage, plan)
    context.barrier("score3d_workers_complete")
    if context.is_root:
        _merge_score3d(config, stage, plan)
    context.barrier("score3d_merged")


def _ligand_plan(config: dict, stage: Path) -> dict:
    root = resolve_path(config, config["output_dir"])
    final_summary = json.loads(
        (root / "05_final_selection" / "summary.json").read_text(encoding="utf-8"))
    expected = int(final_summary["selected"])
    selected_path = root / "05_final_selection" / "selected.csv"
    with selected_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        selected_fields = list(reader.fieldnames or [])
        parents = list(reader)
    if len(parents) != expected:
        raise ValueError(f"Final parent file has {len(parents)}, expected {expected}")
    protonation = ligands.protonate_parents(parents, stage, config["ligand_states"])
    fields = list(selected_fields)
    if "ph_protonated_smiles" not in fields:
        fields.append("ph_protonated_smiles")
    parent_path = stage / "_distributed" / "protonated_parents.csv"
    partial = Path(str(parent_path) + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(parents)
    os.replace(partial, parent_path)
    return {
        "parents": str(parent_path), "parent_count": len(parents),
        "protonation": protonation, "started": time.time(),
    }


def _ligand_worker(config: dict, context: RankContext, stage: Path, plan: dict) -> None:
    with Path(plan["parents"]).open(encoding="utf-8", newline="") as handle:
        parents = [row for index, row in enumerate(csv.DictReader(handle))
                   if index % context.size == context.rank]
    tasks = ((row, config["ligand_states"]) for row in parents)
    csv_path = stage / "_distributed" / f"states_rank_{context.rank:02d}.csv"
    sdf_path = stage / "_distributed" / f"states_rank_{context.rank:02d}.sdf"
    csv_partial = Path(str(csv_path) + ".partial")
    sdf_partial = Path(str(sdf_path) + ".partial")
    fields = ["_global_index", *ligands.STATE_FIELDS]
    statuses: Counter[str] = Counter()
    proposals = 0
    valid = 0
    with csv_partial.open("w", encoding="utf-8", newline="") as csv_handle, \
            sdf_partial.open("w", encoding="utf-8") as sdf_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        with mp.Pool(int(config["workers"]), initializer=ligands._init_worker) as pool:
            for count, (parent_key, states) in enumerate(
                    pool.imap(ligands._generate_task, tasks, chunksize=8), start=1):
                global_index = context.rank + (count - 1) * context.size
                for metadata, mol_block in states:
                    writer.writerow({"_global_index": global_index, **metadata})
                    proposals += 1
                    statuses[metadata["embed_status"]] += 1
                    if metadata["embed_status"].startswith("ok"):
                        sdf_handle.write(mol_block)
                        for key, value in metadata.items():
                            sdf_handle.write(f">  <{key}>\n{value}\n\n")
                        sdf_handle.write(">  <_distributed_global_index>\n")
                        sdf_handle.write(f"{global_index}\n\n$$$$\n")
                        valid += 1
                if count % 1000 == 0:
                    context.log(f"generated {count:,}/{len(parents):,} ligand states")
    os.replace(csv_partial, csv_path)
    os.replace(sdf_partial, sdf_path)
    atomic_json(stage / "_distributed" / f"ligand_summary_rank_{context.rank:02d}.json", {
        "rank": context.rank, "parent_count": len(parents), "proposals": proposals,
        "valid": valid, "statuses": dict(statuses),
    })


def _merge_ligands(config: dict, stage: Path, plan: dict) -> None:
    parent_count = int(plan["parent_count"])
    rows: list[dict | None] = [None] * parent_count
    molecule_blocks: dict[int, str] = {}
    statuses: Counter[str] = Counter()
    proposal_count = 0
    state_count = 0
    for summary_path in sorted((stage / "_distributed").glob("ligand_summary_rank_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        statuses.update(summary["statuses"])
        proposal_count += int(summary["proposals"])
        state_count += int(summary["valid"])
    seen_state_ids: set[str] = set()
    for path in sorted((stage / "_distributed").glob("states_rank_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                index = int(row.pop("_global_index"))
                if index < 0 or index >= parent_count or rows[index] is not None:
                    raise ValueError(f"Invalid or duplicate distributed ligand index {index}")
                if row["state_id"] in seen_state_ids:
                    raise AssertionError(f"Duplicate state ID {row['state_id']}")
                seen_state_ids.add(row["state_id"])
                rows[index] = row
    for path in sorted((stage / "_distributed").glob("states_rank_*.sdf")):
        for mol in Chem.SDMolSupplier(str(path), removeHs=False):
            if mol is None or not mol.HasProp("_distributed_global_index"):
                raise ValueError(f"Invalid distributed ligand record in {path}")
            index = int(mol.GetProp("_distributed_global_index"))
            mol.ClearProp("_distributed_global_index")
            if index in molecule_blocks:
                raise ValueError(f"Duplicate distributed ligand structure index {index}")
            molecule_blocks[index] = Chem.MolToMolBlock(mol)
    missing_rows = [index for index, row in enumerate(rows) if row is None]
    missing_structures = [index for index, row in enumerate(rows)
                          if row is not None and row["embed_status"].startswith("ok")
                          and index not in molecule_blocks]
    failed = [index for index, row in enumerate(rows)
              if row is not None and not row["embed_status"].startswith("ok")]
    if missing_rows or missing_structures:
        raise AssertionError(
            "Distributed ligand generation produced incomplete output: "
            f"missing_rows={len(missing_rows)}, "
            f"missing_structures={len(missing_structures)}")
    valid = [index for index, row in enumerate(rows)
             if row is not None and row["embed_status"].startswith("ok")]
    failure_path = stage / "ligand_state_failures.csv"
    if failed:
        atomic_csv(
            failure_path,
            (rows[index] for index in failed if rows[index] is not None),
            ligands.STATE_FIELDS,
        )
    csv_path = stage / "ligand_states.csv"
    sdf_path = stage / "ligand_states.sdf"
    csv_partial = Path(str(csv_path) + ".partial")
    sdf_partial = Path(str(sdf_path) + ".partial")
    with csv_partial.open("w", encoding="utf-8", newline="") as csv_handle, \
            sdf_partial.open("w", encoding="utf-8") as sdf_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=ligands.STATE_FIELDS)
        writer.writeheader()
        for index in valid:
            row = rows[index]
            assert row is not None
            writer.writerow(row)
            sdf_handle.write(molecule_blocks[index])
            for key in ligands.STATE_FIELDS:
                sdf_handle.write(f">  <{key}>\n{row.get(key, '')}\n\n")
            sdf_handle.write("$$$$\n")
    os.replace(csv_partial, csv_path)
    os.replace(sdf_partial, sdf_path)
    if proposal_count != parent_count or state_count != len(valid):
        raise AssertionError(
            f"Expected {parent_count} single-state proposals, got proposals={proposal_count}, "
            f"embedded={state_count}")
    states_per_parent = {"1": state_count}
    if failed:
        states_per_parent["0"] = len(failed)
    summary = {
        "stage": "ligand_states", "parent_count": parent_count,
        "dockable_parent_count": state_count,
        "excluded_parent_count": len(failed),
        "state_count": state_count, "state_proposal_count": proposal_count,
        "failed_state_proposals": len(failed),
        "ligand_state_failure_report": str(failure_path) if failed else None,
        "embedding_failure_policy": config["ligand_states"]["embedding_failure_policy"],
        "protonation": plan["protonation"], "states_per_parent": states_per_parent,
        "embed_statuses": dict(statuses),
        "all_parents_represented": not failed,
        "all_dockable_parents_represented": True,
        "single_state_policy": True,
        "distributed": _distributed_metadata(config),
        "interpretation": "At most one state per selected parent: Open Babel pH adjustment, deterministic RDKit canonical tautomer selection, source stereochemistry preservation, and one ETKDGv3 starting conformer. Parents that fail verified 3D embedding are excluded with a traceable failure report. This is a reproducible screening state, not a microscopic pKa population model.",
        "elapsed_seconds": time.time() - float(plan["started"]),
    }
    shutil.rmtree(stage / "_distributed")
    atomic_json(stage / "summary.json", summary)


def run_ligands(config: dict, context: RankContext) -> None:
    root = resolve_path(config, config["output_dir"])
    stage = root / "06_ligand_states"
    if not _prepare_distributed_stage(context, "ligands", stage, stage / "summary.json"):
        return
    plan = context.root_value("ligand_plan", lambda: _ligand_plan(config, stage))
    _ligand_worker(config, context, stage, plan)
    context.barrier("ligand_workers_complete")
    if context.is_root:
        _merge_ligands(config, stage, plan)
    context.barrier("ligands_merged")


def run_smina_jobs(config: dict, context: RankContext) -> None:
    root = resolve_path(config, config["output_dir"])
    manifest_path = root / "07_smina" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Smina manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("production_ready", False):
        raise RuntimeError(f"8V1Q-WT production gate failed; inspect {manifest_path}")
    with Path(manifest["jobs"]).open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    if not jobs:
        raise ValueError("Smina job table is empty")
    binary = smina_binary()
    local_jobs = [row for index, row in enumerate(jobs) if index % context.size == context.rank]
    parallel = int(config["docking"]["parallel_jobs_per_node"])
    context.log(
        f"Smina assigned {len(local_jobs)}/{len(jobs)} chunks; "
        f"{parallel} concurrent one-CPU jobs")
    processed = 0
    outcomes: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=min(parallel, max(1, len(local_jobs)))) as executor:
        futures = [executor.submit(run_smina_job, binary, row) for row in local_jobs]
        for future in as_completed(futures):
            result = future.result()
            outcomes[result["status"]] += 1
            processed += 1
            if processed == len(local_jobs) or processed % 10 == 0:
                context.log(
                    f"Smina processed {processed}/{len(local_jobs)} local chunks "
                    f"(new={outcomes['completed']}, skipped={outcomes['skipped']}, "
                    f"recovered={outcomes['recovered']})")
    atomic_json(context.sync_dir / f"smina_rank_{context.rank:02d}.json", {
        "rank": context.rank, "hostname": context.hostname,
        "assigned_jobs": len(local_jobs), "processed_jobs": processed,
        "completed_jobs": outcomes["completed"],
        "skipped_jobs": outcomes["skipped"],
        "recovered_jobs": outcomes["recovered"],
    })
    context.barrier("smina_jobs_complete")


def _postdock_reference(config: dict) -> tuple[Chem.Mol, list[list[float]], list[dict], str, tuple]:
    refs = config["references"]
    reference = Chem.SDMolSupplier(
        str(resolve_path(config, refs["bound_ligand_sdf"])), removeHs=False)[0]
    if reference is None:
        raise ValueError("Experimental PNU reference cannot be parsed")
    settings = config["structure_3d"]
    pocket = screen3d.parse_pocket_atoms(
        resolve_path(config, refs["pdb_file"]), refs["ligand_resname"], reference,
        float(settings["pocket_radius"]))
    features = screen3d.build_feature_model(reference, pocket, settings)
    reference_block = Chem.MolToMolBlock(reference)
    screen3d.init_worker(reference_block, pocket, features, settings)
    self_score = screen3d.score_conformation_similarity(Chem.Mol(reference), 0)
    if self_score[0] < 0.99 or self_score[1] < 0.99 or self_score[2] < 0.99:
        raise AssertionError(f"Post-docking PNU 3D self-check failed: {self_score}")
    return reference, pocket, features, reference_block, self_score


def _postdock_plan(config: dict, context: RankContext, stage: Path) -> dict:
    root = resolve_path(config, config["output_dir"])
    poses_path = root / "08_docking_results" / "docked_poses.sdf"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing consolidated docked-pose SDF: {poses_path}")
    writers = [Chem.SDWriter(str(stage / "_distributed" / f"poses_rank_{rank:02d}.sdf"))
               for rank in range(context.size)]
    seen: set[str] = set()
    count = 0
    try:
        for index, mol in enumerate(Chem.SDMolSupplier(str(poses_path), removeHs=False)):
            if mol is None:
                raise ValueError(f"Invalid pose record {index + 1} in {poses_path}")
            pose_id = mol.GetProp("pose_id") if mol.HasProp("pose_id") else ""
            if not pose_id or pose_id in seen:
                raise ValueError(f"Missing or duplicate pose_id at record {index + 1}: {pose_id}")
            seen.add(pose_id)
            mol.SetIntProp("_distributed_global_index", index)
            writers[index % context.size].write(mol)
            count += 1
    finally:
        for writer in writers:
            writer.close()
    if count == 0:
        raise ValueError("No docked poses are available for post-docking 3D similarity")
    return {"input_pose_count": count, "poses": str(poses_path), "started": time.time()}


def _postdock_worker(config: dict, context: RankContext, stage: Path) -> None:
    _, pocket, features, reference_block, self_score = _postdock_reference(config)
    shard = stage / "_distributed" / f"poses_rank_{context.rank:02d}.sdf"
    tasks = []
    for mol in Chem.SDMolSupplier(str(shard), removeHs=False):
        if mol is None or not mol.HasProp("_distributed_global_index"):
            raise ValueError(f"Invalid distributed post-docking pose in {shard}")
        index = int(mol.GetProp("_distributed_global_index"))
        pose_id = mol.GetProp("pose_id") if mol.HasProp("pose_id") else ""
        tasks.append((index, pose_id, Chem.MolToMolBlock(mol)))
    statuses: Counter[str] = Counter()
    results = []
    with mp.Pool(
            int(config["workers"]), initializer=screen3d.init_worker,
            initargs=(reference_block, pocket, features, config["structure_3d"])) as pool:
        for index, result in pool.imap(postdock3d._score_pose, tasks, chunksize=32):
            statuses[result["postdock_3d_status"]] += 1
            results.append({"_global_index": index, **result})
    results.sort(key=lambda row: int(row["_global_index"]))
    output = stage / "_distributed" / f"postdock_rank_{context.rank:02d}.csv"
    partial = Path(str(output) + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["_global_index", *postdock3d.FIELDS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    os.replace(partial, output)
    atomic_json(stage / "_distributed" / f"postdock_summary_rank_{context.rank:02d}.json", {
        "rank": context.rank, "count": len(tasks), "statuses": dict(statuses),
        "self_score": list(self_score),
    })


def _merge_postdock(config: dict, stage: Path, plan: dict) -> None:
    total = int(plan["input_pose_count"])
    rows: list[dict | None] = [None] * total
    statuses: Counter[str] = Counter()
    for summary_path in sorted((stage / "_distributed").glob("postdock_summary_rank_*.json")):
        statuses.update(json.loads(summary_path.read_text(encoding="utf-8"))["statuses"])
    for path in sorted((stage / "_distributed").glob("postdock_rank_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                index = int(row.pop("_global_index"))
                if index < 0 or index >= total or rows[index] is not None:
                    raise ValueError(f"Invalid or duplicate post-docking index {index}")
                rows[index] = row
    if any(row is None for row in rows):
        missing = [index for index, row in enumerate(rows) if row is None]
        raise ValueError(
            f"Distributed post-docking scoring missed {len(missing)} poses; first={missing[0]}")
    success = statuses.get("ok", 0)
    coverage = success / total
    minimum = float(config["postdock_3d"]["minimum_pose_coverage"])
    if coverage < minimum:
        raise ValueError(
            f"Post-docking 3D similarity coverage {coverage:.3f} is below required {minimum:.3f}")
    output = stage / "pose_3d_similarity.csv"
    partial = Path(str(output) + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=postdock3d.FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows if row is not None)
    os.replace(partial, output)
    _, _, _, _, self_score = _postdock_reference(config)
    summary = {
        "stage": "postdock_3d_similarity", "input_pose_count": total,
        "statuses": dict(statuses), "pose_coverage": coverage,
        "reference": str(resolve_path(config, config["references"]["bound_ligand_sdf"])),
        "method": "Docked conformation is copied and Crippen-O3A aligned to experimental PNU for similarity scoring; the saved 8V1Q docking pose is not overwritten.",
        "pnu_self_check": {
            "combined": self_score[0], "feature": self_score[1],
            "shape": self_score[2], "o3a": self_score[3],
        },
        "distributed": _distributed_metadata(config),
        "output": str(output), "elapsed_seconds": time.time() - float(plan["started"]),
    }
    shutil.rmtree(stage / "_distributed")
    atomic_json(stage / "summary.json", summary)


def run_postdock(config: dict, context: RankContext) -> None:
    root = resolve_path(config, config["output_dir"])
    stage = root / "09_postdock_3d"
    if not _prepare_distributed_stage(
            context, "postdock3d", stage, stage / "summary.json"):
        return
    plan = context.root_value(
        "postdock_plan", lambda: _postdock_plan(config, context, stage))
    _postdock_worker(config, context, stage)
    context.barrier("postdock_workers_complete")
    if context.is_root:
        _merge_postdock(config, stage, plan)
    context.barrier("postdock_merged")


def _run_docking_and_after(config: dict, context: RankContext) -> dict | None:
    root = resolve_path(config, config["output_dir"])
    docking_done = root / "08_docking_results" / "summary.json"
    if not docking_done.exists():
        run_smina_jobs(config, context)
        if context.is_root:
            docking.collect(config, force=True)
        context.barrier("docking_collected")
    else:
        context.log("skip completed docking collection")
        context.barrier("docking_collection_skipped")
    run_postdock(config, context)
    _root_step(
        context, "prepare_affinity",
        lambda: affinity_input.run(config, force=not (
            root / "10_affinity_input" / "summary.json").exists()))
    if context.is_root:
        return {
            "status": "WAITING_FOR_EXTERNAL_AFFINITY",
            "distributed": _distributed_metadata(config, include_total=True),
            "preparation_qc": (
                json.loads((root / "QC_REPORT.json").read_text(encoding="utf-8"))
                if (root / "QC_REPORT.json").exists() else {}),
            "docking": json.loads(docking_done.read_text(encoding="utf-8")),
            "postdock_3d": json.loads(
                (root / "09_postdock_3d" / "summary.json").read_text(encoding="utf-8")),
            "affinity_input": json.loads(
                (root / "10_affinity_input" / "summary.json").read_text(encoding="utf-8")),
        }
    return None


def run(config: dict, context: RankContext, mode: str) -> dict | None:
    hosts = context.validate_unique_hosts(config)
    if context.is_root:
        context.log(
            f"validated {int(config['distributed']['nodes'])}-node layout: "
            f"{', '.join(hosts)}")
    if mode == "complete":
        _root_step(
            context, "references",
            lambda: validate_and_prepare(config, prepare_pdbqt=True))
        run_standardize(config, context)
        run_score2d(config, context)
        _root_step(context, "pool3d", lambda: prescreen_pool.run(config))
        run_score3d(config, context)
        _root_step(context, "final_selection", lambda: final_select.run(config))
        run_ligands(config, context)
        _root_step(context, "prepare_smina", lambda: docking.prepare(config))
        _root_step(context, "pre_docking_qc", lambda: qc.run(config))
    elif mode == "smina":
        manifest = resolve_path(config, config["output_dir"]) / "07_smina" / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Missing {manifest}; run the complete preparation stages first")
    else:
        raise ValueError(f"Unsupported distributed mode: {mode}")
    result = _run_docking_and_after(config, context)
    context.barrier("pipeline_complete")
    return result
