from __future__ import annotations

import csv
import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from .config import resolve_path
from .io_utils import atomic_csv, atomic_json, iter_csv, prepare_stage_dir, stable_hash


FLOAT_FIELDS = {"quality_score", "qed", "pnu_consensus", "pharm2d_similarity", "structure3d_score",
                "structure3d_feature_score", "structure3d_shape_similarity", "structure3d_clash_fraction",
                "library_evidence_score"}
INT_FIELDS = {"soft_alert_count", "hard_filter_pass"}


def typed(row: dict[str, str]) -> dict:
    for field in FLOAT_FIELDS:
        row[field] = float(row.get(field, 0) or 0)
    for field in INT_FIELDS:
        row[field] = int(float(row.get(field, 0) or 0))
    return row


def maxmin_order(records: list[dict], indices: list[int], target: int, seed: int, oversample: int) -> list[int]:
    if not indices or target <= 0:
        return []
    # Prefer one high-quality representative per scaffold before fingerprint diversity.
    best: dict[str, tuple[tuple, int]] = {}
    for index in indices:
        record = records[index]
        key = record["murcko_scaffold"]
        rank = (int(record["soft_alert_count"] == 0), record["quality_score"], -stable_hash(record["structure_key"], seed))
        if key not in best or rank > best[key][0]:
            best[key] = (rank, index)
    reps = [value[1] for value in best.values()]
    reps.sort(key=lambda i: stable_hash(records[i]["structure_key"], seed + 1))
    pool_size = min(len(reps), max(target, min(target * max(1, oversample), 30000)))
    candidate = reps[:pool_size]
    if len(candidate) <= target:
        return candidate
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(records[i]["standardized_smiles"])) for i in candidate]
    picks = MaxMinPicker().LazyBitVectorPick(fps, len(fps), min(len(fps), max(target, int(target * 1.25))), seed=seed)
    return [candidate[int(local)] for local in picks]


def run(config: dict, force: bool = False) -> dict:
    started = time.time()
    root = resolve_path(config, config["output_dir"])
    stage = root / "05_final_selection"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    pool_size = int(config["prescreen_3d_pool"]["size"])
    input_path = root / "04_3d_scored" / f"scored_{pool_size}.csv.gz"
    records = [typed(dict(row)) for row in iter_csv(input_path)]
    selection = config["final_selection"]
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_channel: dict[int, str] = {}
    scaffold_counts: Counter[str] = Counter()
    seed = int(config["seed"])

    def accept(index: int, channel: str, enforce_scaffold_cap: bool = True) -> bool:
        if index in selected_set:
            return False
        scaffold = records[index]["murcko_scaffold"]
        if enforce_scaffold_cap:
            channel_cap = int(selection["targeted_scaffold_cap"] if channel in {
                "pnu_structure3d_enrichment", "pnu_2d_enrichment", "pnu_pharm2d_enrichment"
            } else selection["default_scaffold_cap"])
            cap = min(channel_cap, int(selection["global_scaffold_cap"]))
            if scaffold_counts[scaffold] >= cap:
                return False
            scaffold_counts[scaffold] += 1
        selected.append(index)
        selected_set.add(index)
        selected_channel[index] = channel
        return True

    diagnostics = {}
    channel_order = [
        "pnu_structure3d_enrichment", "pnu_2d_enrichment", "pnu_pharm2d_enrichment",
        "h9e937_pocket_discovery", "diversity", "exploration",
    ]
    activity_pool_indices = sorted(
        (i for i, record in enumerate(records) if record["source_pool"] == "activity_recorded"),
        key=lambda i: records[i]["structure_key"])
    activity_indices = sorted(
        (i for i, record in enumerate(records)
         if record["source_pool"] == "activity_recorded"
         and record.get("structure3d_status") in {"ok", "feasible_only"}),
        key=lambda i: records[i]["structure_key"])
    feasible_activity_set = set(activity_indices)
    activity_failures = [i for i in activity_pool_indices if i not in feasible_activity_set]
    if activity_failures and selection.get("require_all_activity_recorded_3d", False):
        failure_path = stage / "activity_3d_failures.csv"
        atomic_csv(failure_path, ({
            "compound_id": records[i].get("compound_id", ""),
            "structure_key": records[i].get("structure_key", ""),
            "standardized_smiles": records[i].get("standardized_smiles", ""),
            "structure3d_status": records[i].get("structure3d_status", ""),
            "structure3d_embedding_method": records[i].get("structure3d_embedding_method", ""),
        } for i in activity_failures), [
            "compound_id", "structure_key", "standardized_smiles", "structure3d_status",
            "structure3d_embedding_method",
        ])
        raise ValueError(
            f"{len(activity_failures)} activity-recorded parents lack a verified 3D conformation; "
            f"see {failure_path}. The all-activity docking policy will not silently drop them")
    activity_channel = selection["activity_selection_channel"]
    for index in activity_indices:
        if not accept(index, activity_channel, enforce_scaffold_cap=False):
            raise AssertionError(f"Could not include activity-recorded parent {records[index]['structure_key']}")
    diagnostics["activity_recorded_all"] = {
        "3d_feasible": len(activity_indices), "selected": len(activity_indices),
        "scaffold_cap_applied": False,
    }

    source_indices = [
        i for i, record in enumerate(records)
        if record["source_pool"] == "unlabeled"
        and record.get("structure3d_status") in {"ok", "feasible_only"}
    ]
    unlabeled_target = int(selection["unlabeled_count"])
    if len(source_indices) < unlabeled_target:
        raise ValueError(
            f"Unlabeled source has {len(source_indices)} 3D-feasible parents, needs {unlabeled_target}")
    pnu_values = [records[i]["pnu_consensus"] for i in source_indices]
    novelty_cutoff = float(np.quantile(pnu_values, 0.50)) if pnu_values else 0.0
    diagnostics["unlabeled_novelty_cutoff"] = novelty_cutoff
    for channel_offset, channel in enumerate(channel_order):
        target = int(selection["unlabeled_channel_counts"][channel])
        if channel == "pnu_structure3d_enrichment":
            order = sorted(
                (i for i in source_indices if records[i].get("structure3d_status") == "ok" and records[i]["structure3d_score"] >= float(selection["min_structure3d_score"])),
                key=lambda i: (records[i]["structure3d_score"], records[i]["quality_score"]), reverse=True)
        elif channel == "pnu_2d_enrichment":
            order = sorted(source_indices, key=lambda i: (
                records[i].get("library_evidence_score", records[i]["pnu_consensus"]),
                records[i]["quality_score"]), reverse=True)
        elif channel == "pnu_pharm2d_enrichment":
            order = sorted(source_indices, key=lambda i: (records[i]["pharm2d_similarity"], records[i]["quality_score"]), reverse=True)
        else:
            candidates = [i for i in source_indices if i not in selected_set]
            if channel == "exploration":
                candidates = [i for i in candidates if records[i]["pnu_consensus"] <= novelty_cutoff and records[i]["soft_alert_count"] == 0 and records[i]["qed"] >= 0.45]
            elif channel == "h9e937_pocket_discovery":
                # No H9E937 ligand pose exists yet.  This is an honest,
                # chemically diverse discovery reserve, not a fabricated
                # target-fit score.
                candidates = [i for i in candidates if records[i].get("structure3d_status") in {"ok", "feasible_only"}
                              and records[i]["soft_alert_count"] == 0
                              and records[i]["qed"] >= float(selection["min_diversity_qed"])]
            else:
                candidates = [i for i in candidates if records[i]["qed"] >= float(selection["min_diversity_qed"])]
            order = maxmin_order(records, candidates, target, seed + 100 + channel_offset,
                                 int(selection["diversity_oversample_factor"]))
        before = len(selected)
        for index in order:
            if len(selected) - before >= target:
                break
            accept(index, channel)
        if len(selected) - before < target:
            # Deterministic same-source/channel backfill; caps remain strict.
            fallback = sorted(
                (i for i in source_indices if i not in selected_set),
                key=lambda i: (records[i]["quality_score"], -stable_hash(records[i]["structure_key"], seed + 999)), reverse=True)
            for index in fallback:
                if len(selected) - before >= target:
                    break
                accept(index, channel)
        obtained = len(selected) - before
        diagnostics[f"unlabeled|{channel}"] = {
            "requested": target, "selected": obtained, "ranked_pool": len(order)}
        if obtained != target:
            raise RuntimeError(
                f"Could select only {obtained}/{target} for unlabeled|{channel} under scaffold caps")

    expected_size = len(activity_indices) + unlabeled_target
    if len(selected) != expected_size:
        raise AssertionError(f"Final selection size {len(selected)} != dynamic target {expected_size}")
    fields = list(records[0]) + ["selection_channel", "selection_rank"]
    output = stage / "selected.csv"
    partial = Path(str(output) + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, index in enumerate(selected, start=1):
            writer.writerow({**records[index], "selection_channel": selected_channel[index], "selection_rank": rank})
    os.replace(partial, output)
    smiles_path = stage / "selected.smi"
    partial_smi = Path(str(smiles_path) + ".partial")
    with partial_smi.open("w", encoding="utf-8") as handle:
        for index in selected:
            handle.write(f"{records[index]['standardized_smiles']} {records[index]['compound_id']}\n")
    os.replace(partial_smi, smiles_path)
    source_counts = Counter(records[i]["source_pool"] for i in selected)
    channel_counts = Counter(selected_channel.values())
    cross_counts = Counter((records[i]["source_pool"], selected_channel[i]) for i in selected)
    expected_sources = {"activity_recorded": len(activity_indices), "unlabeled": unlabeled_target}
    if dict(source_counts) != expected_sources:
        raise AssertionError(f"Source quotas failed: {source_counts}")
    expected_channels = {
        activity_channel: len(activity_indices),
        **{key: int(value) for key, value in selection["unlabeled_channel_counts"].items()},
    }
    if dict(channel_counts) != expected_channels:
        raise AssertionError(f"Channel quotas failed: {channel_counts}")
    if max(scaffold_counts.values()) > int(selection["global_scaffold_cap"]):
        raise AssertionError("Unlabeled scaffold cap violated")
    overall_scaffold_counts = Counter(records[i]["murcko_scaffold"] for i in selected)
    summary = {
        "stage": "final_selection", "selected": len(selected), "source_counts": dict(source_counts),
        "channel_counts": dict(channel_counts),
        "source_channel_counts": {f"{source}|{channel}": count for (source, channel), count in cross_counts.items()},
        "selection_policy": "all 3D-feasible activity-recorded parents plus a fixed unlabeled quota",
        "activity_3d_feasible": len(activity_indices), "activity_included": len(activity_indices),
        "activity_in_3d_pool": len(activity_pool_indices),
        "activity_3d_failure_count": len(activity_failures),
        "unlabeled_requested": unlabeled_target,
        "output": str(output),
        "unique_scaffolds": len(overall_scaffold_counts),
        "maximum_scaffold_multiplicity": max(overall_scaffold_counts.values()),
        "maximum_unlabeled_scaffold_multiplicity": max(scaffold_counts.values()),
        "diagnostics": diagnostics, "elapsed_seconds": time.time() - started,
    }
    atomic_json(done, summary)
    return summary
