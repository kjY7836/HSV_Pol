from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .config import resolve_path
from .io_utils import atomic_csv, atomic_json, prepare_stage_dir


OUTPUT_FIELDS = [
    "rank", "parent_structure_key", "compound_id", "source_pool", "selection_channel",
    "integrated_score", "wt_affinity", "postdock_3d_similarity",
    "docking_consensus", "qed", "wt_protocol_count", "affinity_pose_count",
]


def _float(row: dict, key: str, default: float | None = None) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        if default is None:
            raise ValueError(f"Missing numeric field {key}")
        return float(default)
    return float(value)


def _unit(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0,1], found {value}")
    return value


def scaled_affinity(predicted_paffinity: float, settings: dict) -> float:
    """Map an external PDBbind-model pAffinity prediction to a 0..1 score."""
    midpoint = float(settings["midpoint_paffinity"])
    scale = float(settings["sigmoid_scale"])
    if scale <= 0:
        raise ValueError("Affinity sigmoid_scale must be positive")
    return 1.0 / (1.0 + math.exp(-(float(predicted_paffinity) - midpoint) / scale))


def _rank_scores(rows: list[dict]) -> dict[str, float]:
    """Within-protocol percentile scores; a more negative Smina score is better."""
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["protocol_id"]].append((row["pose_id"], _float(row, "smina_score")))
    result: dict[str, float] = {}
    for values in grouped.values():
        ordered = sorted(values, key=lambda item: item[1])
        size = len(ordered)
        positions: dict[float, list[int]] = defaultdict(list)
        for position, (_, value) in enumerate(ordered):
            positions[value].append(position)
        for pose_id, value in ordered:
            average_position = sum(positions[value]) / len(positions[value])
            result[pose_id] = 1.0 if size == 1 else 1.0 - average_position / (size - 1)
    return result


def _or_aggregate(values: list[float]) -> float:
    """Conformation-selective OR: favor the best state without ignoring all others."""
    if not values:
        return 0.0
    return 0.70 * max(values) + 0.30 * (sum(values) / len(values))


def aggregate_scores(parents: list[dict], docking_rows: list[dict], affinity_rows: list[dict],
                     settings: dict, profile_name: str,
                     similarity_rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    if profile_name not in settings["profiles"]:
        raise ValueError(f"Unknown integrated-scoring profile {profile_name}")
    profile = settings["profiles"][profile_name]
    affinity_settings = settings["affinity"]
    parent_by_key = {row["structure_key"]: row for row in parents}
    if len(parent_by_key) != len(parents):
        raise ValueError("Final-selection parent_structure_key values are not unique")
    all_docking_pose_ids: set[str] = set()
    excluded_protocols = set(settings.get("excluded_protocols", []))
    scoring_docking_rows: list[dict] = []
    docked_parent_keys: set[str] = set()
    for row in docking_rows:
        pose_id = row.get("pose_id", "")
        parent_key = row.get("parent_structure_key", "")
        if not pose_id or not parent_key:
            raise ValueError("Docking row lacks pose_id or parent_structure_key")
        if pose_id in all_docking_pose_ids:
            raise ValueError(f"Duplicate docking pose_id {pose_id}")
        if parent_key not in parent_by_key:
            raise ValueError(f"Docking pose {pose_id} refers to unknown parent {parent_key}")
        if row.get("mutant_id", "WT") != "WT":
            raise ValueError(
                f"Protocol {row.get('protocol_id', '')} contains non-WT pose {pose_id}; "
                "this workflow is WT-only")
        all_docking_pose_ids.add(pose_id)
        if row.get("protocol_id", "") not in excluded_protocols:
            scoring_docking_rows.append(row)
            docked_parent_keys.add(parent_key)
    if not scoring_docking_rows:
        raise ValueError("No scoring-eligible docking poses remain after protocol exclusions")
    states_by_parent: dict[str, set[str]] = defaultdict(set)
    for row in scoring_docking_rows:
        states_by_parent[row["parent_structure_key"]].add(row["state_id"])
    multiple_states = {
        parent: sorted(states) for parent, states in states_by_parent.items() if len(states) != 1}
    if multiple_states:
        example = next(iter(multiple_states.items()))
        raise ValueError(
            f"Single-state policy violated for {len(multiple_states)} parents; example: {example}")

    affinity_by_pose: dict[str, dict] = {}
    for row in affinity_rows:
        pose_id = row.get("pose_id", "")
        if not pose_id:
            raise ValueError("Affinity row lacks pose_id")
        if pose_id in affinity_by_pose:
            raise ValueError(f"Duplicate affinity prediction for {pose_id}")
        affinity_by_pose[pose_id] = row
    unknown_affinity_poses = set(affinity_by_pose) - all_docking_pose_ids
    if unknown_affinity_poses:
        example = sorted(unknown_affinity_poses)[0]
        raise ValueError(
            f"Affinity file contains {len(unknown_affinity_poses)} unknown pose_id values; "
            f"example: {example}")

    similarity_by_pose: dict[str, dict] = {}
    for row in similarity_rows or []:
        pose_id = row.get("pose_id", "")
        if not pose_id:
            raise ValueError("Post-docking 3D similarity row lacks pose_id")
        if pose_id in similarity_by_pose:
            raise ValueError(f"Duplicate post-docking 3D similarity for {pose_id}")
        if pose_id not in all_docking_pose_ids:
            raise ValueError(f"Post-docking 3D similarity contains unknown pose_id {pose_id}")
        similarity_by_pose[pose_id] = row
    similarity_matched = sum(
        row["pose_id"] in similarity_by_pose for row in scoring_docking_rows)
    similarity_usable = sum(
        similarity_by_pose.get(row["pose_id"], {}).get("postdock_3d_status", "") == "ok"
        for row in scoring_docking_rows)
    similarity_coverage = similarity_usable / len(scoring_docking_rows)
    similarity_required = bool(settings.get("postdock_3d_required", False))
    minimum_similarity = float(settings.get("minimum_postdock_3d_pose_coverage", 0.0))
    if similarity_required and similarity_coverage < minimum_similarity:
        raise ValueError(
            f"Post-docking 3D pose coverage {similarity_coverage:.3f} is below required "
            f"{minimum_similarity:.3f}")

    docking_rank = _rank_scores(scoring_docking_rows)
    merged_by_parent: dict[str, list[dict]] = defaultdict(list)
    matched = 0
    for docking in scoring_docking_rows:
        affinity = affinity_by_pose.get(docking["pose_id"])
        if affinity is None:
            continue
        matched += 1
        prediction = _float(affinity, affinity_settings["prediction_column"])
        similarity = similarity_by_pose.get(docking["pose_id"])
        similarity_value = 0.0
        if similarity is not None and similarity.get("postdock_3d_status", "") == "ok":
            similarity_value = _unit(
                _float(similarity, "postdock_3d_similarity"), "postdock_3d_similarity")
        merged_by_parent[docking["parent_structure_key"]].append({
            **docking, "predicted_paffinity": prediction,
            "affinity_evidence": scaled_affinity(prediction, affinity_settings),
            "docking_percentile": docking_rank[docking["pose_id"]],
            "postdock_3d_value": similarity_value,
        })
    coverage = matched / len(scoring_docking_rows)
    if coverage < float(affinity_settings["minimum_pose_coverage"]):
        raise ValueError(
            f"Affinity pose coverage {coverage:.3f} is below required "
            f"{float(affinity_settings['minimum_pose_coverage']):.3f}")
    matched_parent_keys = set(merged_by_parent)
    parent_coverage = (
        len(matched_parent_keys) / len(docked_parent_keys) if docked_parent_keys else 0.0
    )
    if parent_coverage < float(affinity_settings["minimum_parent_coverage"]):
        raise ValueError(
            f"Affinity parent coverage {parent_coverage:.3f} is below required "
            f"{float(affinity_settings['minimum_parent_coverage']):.3f}")

    output = []
    required_protocols = set(settings.get("required_protocols", []))
    for parent_key, parent in parent_by_key.items():
        poses = merged_by_parent.get(parent_key, [])
        if not poses:
            continue
        # Exactly one chemical state exists per parent. Select the pose with the
        # strongest external affinity evidence within each docking protocol.
        best_protocol: dict[str, dict] = {}
        for pose in poses:
            key = pose["protocol_id"]
            if key not in best_protocol or pose["affinity_evidence"] > best_protocol[key]["affinity_evidence"]:
                best_protocol[key] = pose

        missing_protocols = required_protocols - set(best_protocol)
        if missing_protocols:
            raise ValueError(
                f"Parent {parent_key} lacks required WT protocols: {sorted(missing_protocols)}")

        wt = list(best_protocol.values())
        if not wt:
            continue

        wt_affinity = _or_aggregate([pose["affinity_evidence"] for pose in wt])
        docking_consensus = _or_aggregate([pose["docking_percentile"] for pose in wt])
        postdock_3d_similarity = _or_aggregate([pose["postdock_3d_value"] for pose in wt])
        components = {
            "wt_affinity": wt_affinity,
            "postdock_3d_similarity": postdock_3d_similarity,
            "docking_consensus": docking_consensus,
            "qed": _unit(float(parent.get("qed", 0.0) or 0.0), "qed"),
        }
        base = sum(float(weight) * components[name] for name, weight in profile["weights"].items())
        integrated = 100.0 * max(0.0, min(1.0, base))
        output.append({
            "parent_structure_key": parent_key, "compound_id": parent.get("compound_id", ""),
            "source_pool": parent.get("source_pool", ""),
            "selection_channel": parent.get("selection_channel", ""),
            "integrated_score": integrated, **components,
            "wt_protocol_count": len(wt), "affinity_pose_count": len(poses),
        })
    output.sort(key=lambda row: (row["integrated_score"], row["wt_affinity"]), reverse=True)
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    diagnostics = {
        "profile": profile_name, "parent_input_count": len(parents),
        "ranked_parent_count": len(output), "docking_pose_count": len(docking_rows),
        "scoring_docking_pose_count": len(scoring_docking_rows),
        "excluded_protocols": sorted(excluded_protocols),
        "required_protocols": sorted(required_protocols),
        "matched_affinity_pose_count": matched, "affinity_pose_coverage": coverage,
        "docked_parent_count": len(docked_parent_keys),
        "matched_affinity_parent_count": len(matched_parent_keys),
        "affinity_parent_coverage": parent_coverage,
        "matched_postdock_3d_pose_count": similarity_matched,
        "usable_postdock_3d_pose_count": similarity_usable,
        "postdock_3d_pose_coverage": similarity_coverage,
        "formula_weights": profile["weights"],
        "affinity_is_external": True,
    }
    return output, diagnostics


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(config: dict, force: bool = False, profile: str = "wt") -> dict:
    root = resolve_path(config, config["output_dir"])
    stage = root / "11_integrated_scoring" / profile
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    parents_path = root / "05_final_selection" / "selected.csv"
    docking_path = root / "08_docking_results" / "docking_poses.csv"
    affinity_path = root / "10_affinity_input" / "affinity_predictions.csv"
    similarity_path = root / "09_postdock_3d" / "pose_3d_similarity.csv"
    if not affinity_path.exists():
        raise FileNotFoundError(
            f"External PDBbind-model predictions are required at {affinity_path}; "
            "this repository does not train or fabricate that model")
    if config["integrated_scoring"].get("postdock_3d_required", False) and not similarity_path.exists():
        raise FileNotFoundError(
            f"Post-docking 3D similarity is required at {similarity_path}; run postdock3d first")
    affinity_rows = _read_csv(affinity_path)
    prediction_column = config["integrated_scoring"]["affinity"]["prediction_column"]
    blank_predictions = sum(row.get(prediction_column, "") in (None, "") for row in affinity_rows)
    if blank_predictions:
        raise ValueError(
            f"{blank_predictions}/{len(affinity_rows)} affinity predictions are blank in "
            f"{affinity_path}; fill {prediction_column} before integrated scoring")
    rows, diagnostics = aggregate_scores(
        _read_csv(parents_path), _read_csv(docking_path), affinity_rows,
        config["integrated_scoring"], profile,
        _read_csv(similarity_path) if similarity_path.exists() else None)
    output = stage / "ranked_parents.csv"
    atomic_csv(output, rows, OUTPUT_FIELDS)
    summary = {"stage": "integrated_scoring", **diagnostics, "output": str(output)}
    atomic_json(done, summary)
    return summary
