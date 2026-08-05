from __future__ import annotations

import csv
import gzip
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from .config import resolve_path
from .io_utils import atomic_json, iter_csv, prepare_stage_dir, stable_hash
from .screen2d import SCORED_FIELDS


NUMERIC_FLOATS = {"mw", "clogp", "tpsa", "fraction_csp3", "qed", "pnu_ecfp_similarity",
                  "pnu_fcfp_similarity", "pharm2d_similarity", "pnu_consensus", "quality_score",
                  "library_evidence_score"}
NUMERIC_INTS = {"hard_filter_pass", "hbd", "hba", "rotatable_bonds", "ring_count", "heavy_atoms",
                "formal_charge", "soft_alert_count", "has_permanent_charge", "active_connectivity_overlap"}


def typed(row: dict[str, str]) -> dict:
    for key in NUMERIC_FLOATS:
        row[key] = float(row.get(key, 0) or 0)
    for key in NUMERIC_INTS:
        row[key] = int(float(row.get(key, 0) or 0))
    return row


def fraction_quotas(size: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {key: size * float(value) for key, value in fractions.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    missing = size - sum(result.values())
    order = sorted(raw, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:missing]:
        result[key] += 1
    return result


def select_source(records: list[dict], target: int, fractions: dict, seed: int) -> tuple[list[int], dict[int, str], dict]:
    target = min(target, len(records))
    quotas = fraction_quotas(target, fractions)
    selected: list[int] = []
    selected_set: set[int] = set()
    channels: dict[int, str] = {}
    channel_counts: Counter[str] = Counter()

    def add(index: int, channel: str) -> bool:
        if index in selected_set:
            return False
        selected.append(index)
        selected_set.add(index)
        channels[index] = channel
        channel_counts[channel] += 1
        return True

    def fill_ranked(channel: str, order: list[int]) -> None:
        for index in order:
            if channel_counts[channel] >= quotas.get(channel, 0):
                break
            add(index, channel)

    if "pnu_2d" in quotas:
        fill_ranked("pnu_2d", sorted(
            range(len(records)),
            key=lambda i: (records[i].get("library_evidence_score", records[i]["pnu_consensus"]),
                           records[i]["quality_score"]), reverse=True))
    if "pharm2d" in quotas:
        fill_ranked("pharm2d", sorted(
            range(len(records)),
            key=lambda i: (records[i]["pharm2d_similarity"], records[i]["quality_score"]), reverse=True))

    for channel, offset in (("scaffold_diversity", 1), ("exploration", 11)):
        if channel not in quotas:
            continue
        cutoff = statistics.median(record["pnu_consensus"] for record in records) if records else 0.0
        best_by_scaffold: dict[str, tuple[tuple, int]] = {}
        for index, record in enumerate(records):
            if index in selected_set:
                continue
            if channel == "exploration" and not (
                record["pnu_consensus"] <= cutoff
                and record["soft_alert_count"] == 0
                and record["quality_score"] >= 0.35
            ):
                continue
            scaffold = str(record["murcko_scaffold"])
            rank = (int(record["soft_alert_count"] == 0), record["quality_score"],
                    -stable_hash(record["structure_key"], seed + offset))
            if scaffold not in best_by_scaffold or rank > best_by_scaffold[scaffold][0]:
                best_by_scaffold[scaffold] = (rank, index)
        reps = [value[1] for value in best_by_scaffold.values()]
        reps.sort(key=lambda i: stable_hash(records[i]["structure_key"], seed + offset + 1))
        fill_ranked(channel, reps)

    # Each quota is backfilled explicitly under its original channel name.  This
    # keeps the audit table exact while making any shortage visible in summary.
    for offset, channel in enumerate(quotas, start=100):
        if channel_counts[channel] >= quotas[channel]:
            continue
        remaining = [i for i in range(len(records)) if i not in selected_set]
        remaining.sort(key=lambda i: (
            records[i]["quality_score"], -stable_hash(records[i]["structure_key"], seed + offset)),
            reverse=True)
        fill_ranked(channel, remaining)
    if len(selected) != target:
        raise RuntimeError(f"Selected {len(selected)} of requested {target}")
    return selected, channels, quotas


def run(config: dict, force: bool = False) -> dict:
    root = resolve_path(config, config["output_dir"])
    source = root / "02_2d_scored"
    stage = root / "03_3d_pool"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    started = time.time()
    by_source = {"activity_recorded": [], "unlabeled": []}
    paths = [source / "activity_scored.csv.gz", *sorted(source.glob("unlabeled_scored_*.csv.gz"))]
    for path in paths:
        for row in iter_csv(path):
            row = typed(row)
            if row["hard_filter_pass"]:
                by_source[row["source_pool"]].append(row)
    settings = config["prescreen_3d_pool"]
    total = int(settings["size"])
    available = sum(map(len, by_source.values()))
    if available < total:
        raise ValueError(f"Only {available:,} hard-filtered parents for 3D-pool target {total:,}")
    if settings.get("include_all_activity_recorded", False):
        activity_target = len(by_source["activity_recorded"])
        if activity_target > total:
            raise ValueError(
                f"All-activity policy needs {activity_target:,} activity-recorded parents, "
                f"but the 3D pool size is only {total:,}")
    else:
        raise ValueError("This workflow requires include_all_activity_recorded=true")
    unlabeled_target = total - activity_target
    if len(by_source["unlabeled"]) < unlabeled_target:
        raise ValueError(
            f"Only {len(by_source['unlabeled']):,} unlabeled parents are available for the "
            f"remaining {unlabeled_target:,} 3D-pool positions")
    targets = {"activity_recorded": activity_target, "unlabeled": unlabeled_target}

    selected_rows = []
    planned_channels = {}
    for offset, source_pool in enumerate(("activity_recorded", "unlabeled")):
        records = by_source[source_pool]
        indices, channels, quotas = select_source(
            records, targets[source_pool], settings["channel_fractions"], int(config["seed"]) + 100 * offset)
        planned_channels[source_pool] = quotas
        for rank, index in enumerate(indices, start=1):
            selected_rows.append({**records[index], "prescreen_channel": channels[index], "prescreen_source_rank": rank})
    if len(selected_rows) != total:
        raise AssertionError(f"3D pool size {len(selected_rows)} != {total}")
    fields = SCORED_FIELDS + ["prescreen_channel", "prescreen_source_rank"]
    output = stage / f"pool_{total}.csv.gz"
    partial = Path(str(output) + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8", newline="", compresslevel=3) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)
    os.replace(partial, output)
    actual = Counter((row["source_pool"], row["prescreen_channel"]) for row in selected_rows)
    summary = {
        "stage": "build_3d_pool", "target_size": total,
        "hard_filter_available": {key: len(value) for key, value in by_source.items()},
        "source_targets": targets, "planned_channel_quotas": planned_channels,
        "all_hard_filtered_activity_included": activity_target == len(by_source["activity_recorded"]),
        "actual_source_channels": {f"{source}|{channel}": count for (source, channel), count in actual.items()},
        "unique_structures": len({row["structure_key"] for row in selected_rows}),
        "unique_scaffolds": len({row["murcko_scaffold"] for row in selected_rows}),
        "elapsed_seconds": time.time() - started,
    }
    if summary["unique_structures"] != total:
        raise AssertionError("Duplicate exact structures entered the 3D pool")
    atomic_json(done, summary)
    return summary
