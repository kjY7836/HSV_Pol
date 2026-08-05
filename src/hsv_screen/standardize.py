from __future__ import annotations

import csv
import gzip
import heapq
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from rdkit import RDLogger

from .chemistry import classify_and_standardize
from .config import resolve_path
from .io_utils import atomic_json, iter_raw_dicts, prepare_stage_dir, raw_record, stable_hash


MAPPING_FIELDS = [
    "source_file", "source_row", "source_pool", "compound_id", "cas", "name",
    "original_mol_weight", "original_formula", "original_smiles", "status",
    "component_class", "original_fragment_count", "removed_component_smiles",
    "charged_parent_smiles", "standardized_smiles", "connectivity_smiles",
    "structure_key", "connectivity_key", "std_mol_weight", "heavy_atom_count",
    "formal_charge", "has_permanent_charge", "model_eligible", "dedup_status",
    "resolved_source_pool", "active_exact_overlap", "active_connectivity_overlap",
]

PARENT_FIELDS = [
    "compound_id", "standardized_smiles", "charged_parent_smiles", "structure_key",
    "connectivity_key", "source_pool", "source_file", "source_row", "component_class",
    "has_permanent_charge", "active_connectivity_overlap",
]


def discover_sources(data_dir: Path) -> tuple[Path, list[Path]]:
    files = sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".xlsx"})
    active = [path for path in files if "2w" in path.name.lower()]
    if len(active) != 1:
        xlsx = [path for path in files if path.suffix.lower() == ".xlsx"]
        active = xlsx if len(xlsx) == 1 else active
    if len(active) != 1:
        raise ValueError(f"Expected exactly one activity-record file, found: {active}")
    candidates = [path for path in files if path != active[0]]
    if not candidates:
        raise ValueError("No candidate source files found")
    return active[0], candidates


def iter_source(path: Path, data_dir: Path, source_pool: str) -> Iterator[dict[str, str]]:
    for row in iter_raw_dicts(path):
        yield raw_record(path, data_dir, row, source_pool)


def minhash_sample(records: Iterable[dict], count: int, seed: int) -> list[dict]:
    if count <= 0:
        return []
    heap: list[tuple[int, str, dict]] = []
    for record in records:
        identity = "|".join((record["source_file"], record["source_row"], record["original_smiles"]))
        value = stable_hash(identity, seed)
        tie = f"{record['source_file']}|{record['source_row']}"
        item = (-value, tie, record)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]


def _standardize_task(args: tuple[dict, dict]) -> dict:
    return classify_and_standardize(*args)


def _process_records(records: Iterable[dict], settings: dict, workers: int) -> Iterator[dict]:
    tasks = ((record, settings) for record in records)
    if workers <= 1:
        for task in tasks:
            yield _standardize_task(task)
        return
    with mp.Pool(workers, initializer=RDLogger.DisableLog, initargs=("rdApp.*",)) as pool:
        yield from pool.imap(_standardize_task, tasks, chunksize=256)


def _open_writer(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    handle = gzip.open(partial, "wt", encoding="utf-8", newline="", compresslevel=3)
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return partial, handle, writer


def run(config: dict, force: bool = False) -> dict:
    output_root = resolve_path(config, config["output_dir"])
    stage = output_root / "01_standardized"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    data_dir = resolve_path(config, config["data_dir"])
    active_path, candidate_paths = discover_sources(data_dir)
    sample = config["sample"]
    seed = int(config["seed"])
    started = time.time()

    if sample["enabled"]:
        active_records: Iterable[dict] = minhash_sample(
            iter_source(active_path, data_dir, "activity_recorded"), int(sample["active_records"]), seed)
        total_candidate = int(sample["candidate_records"])
        base, residue = divmod(total_candidate, len(candidate_paths))
        candidate_sources = []
        for index, path in enumerate(candidate_paths):
            n = base + int(index < residue)
            candidate_sources.append((path, minhash_sample(iter_source(path, data_dir, "unlabeled"), n, seed + index + 1)))
    else:
        active_records = iter_source(active_path, data_dir, "activity_recorded")
        candidate_sources = [(path, iter_source(path, data_dir, "unlabeled")) for path in candidate_paths]

    map_partial, map_handle, map_writer = _open_writer(stage / "parent_mapping.csv.gz", MAPPING_FIELDS)
    act_partial, act_handle, act_writer = _open_writer(stage / "activity_parents.csv.gz", PARENT_FIELDS)
    seen_exact: set[str] = set()
    active_exact: set[str] = set()
    active_connectivity: set[str] = set()
    counts = Counter()

    def consume(records: Iterable[dict], parent_writer, pool_name: str) -> None:
        for result in _process_records(records, config["standardization"], int(config["workers"])):
            counts["input_records"] += 1
            counts[f"status:{result['status']}"] += 1
            result["active_exact_overlap"] = int(bool(result["structure_key"] and result["structure_key"] in active_exact))
            result["active_connectivity_overlap"] = int(bool(result["connectivity_key"] and result["connectivity_key"] in active_connectivity))
            result["resolved_source_pool"] = pool_name
            if result["status"] != "valid_parent" or not result["model_eligible"]:
                result["dedup_status"] = "not_in_parent_pool"
            elif result["structure_key"] in seen_exact:
                result["dedup_status"] = "exact_duplicate"
                if pool_name == "unlabeled" and result["structure_key"] in active_exact:
                    result["resolved_source_pool"] = "activity_recorded"
                    counts["candidate_exact_overlap_activity"] += 1
            else:
                result["dedup_status"] = "unique_parent"
                seen_exact.add(result["structure_key"])
                parent_writer.writerow(result)
                counts[f"unique_parent:{pool_name}"] += 1
                if pool_name == "activity_recorded":
                    active_exact.add(result["structure_key"])
                    active_connectivity.add(result["connectivity_key"])
            map_writer.writerow(result)
            if counts["input_records"] % 100000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(f"[standardize] {counts['input_records']:,} records; {counts['input_records']/elapsed:,.0f}/s", flush=True)

    try:
        consume(active_records, act_writer, "activity_recorded")
        act_handle.close()
        os.replace(act_partial, stage / "activity_parents.csv.gz")
        for shard_index, (path, records) in enumerate(candidate_sources, start=1):
            output = stage / f"unlabeled_parents_{shard_index:02d}.csv.gz"
            partial, handle, writer = _open_writer(output, PARENT_FIELDS)
            consume(records, writer, "unlabeled")
            handle.close()
            os.replace(partial, output)
        map_handle.close()
        os.replace(map_partial, stage / "parent_mapping.csv.gz")
    except Exception:
        map_handle.close()
        act_handle.close()
        raise

    summary = {
        "stage": "standardize", "data_dir": str(data_dir), "activity_file": str(active_path),
        "candidate_files": [str(path) for path in candidate_paths], "sample": sample,
        "counts": dict(counts), "exact_parent_count": len(seen_exact),
        "interpretation": {
            "activity_recorded_is_target_label": False,
            "component_policy": "Remove only explicitly recognized salt/solvent components when one unambiguous organic parent remains; quarantine ambiguous multi-organic and metal-containing records.",
            "dedup_policy": "Exact standardized isomeric structure only; stereoisomers remain distinct; activity-recorded copy wins exact cross-source overlap."
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(done, summary)
    return summary
