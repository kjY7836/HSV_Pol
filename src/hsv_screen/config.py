from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in raw:
        parent = (path.parent / raw["extends"]).resolve()
        base, _ = load_config(parent)
        raw = _merge(base, raw)
    root = path.parent.parent.resolve()
    raw["_config_path"] = str(path)
    raw["_project_dir"] = str(root)
    validate_config(raw)
    return raw, root


def resolve_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(config["_project_dir"]) / path).resolve()


def validate_config(config: dict) -> None:
    pre = config["prescreen_3d_pool"]
    if abs(sum(pre["channel_fractions"].values()) - 1.0) > 1e-9:
        raise ValueError("prescreen_3d_pool.channel_fractions must sum to 1")
    if not pre.get("include_all_activity_recorded", False):
        raise ValueError("The docking policy requires all hard-filtered activity-recorded parents in the 3D pool")
    final = config["final_selection"]
    if final.get("activity_recorded_policy") != "all_3d_feasible":
        raise ValueError("final_selection.activity_recorded_policy must be all_3d_feasible")
    if not final.get("require_all_activity_recorded_3d", False):
        raise ValueError("final_selection.require_all_activity_recorded_3d must be true")
    unlabeled_count = int(final["unlabeled_count"])
    if unlabeled_count <= 0:
        raise ValueError("final_selection.unlabeled_count must be positive")
    if sum(int(value) for value in final["unlabeled_channel_counts"].values()) != unlabeled_count:
        raise ValueError("final_selection.unlabeled_channel_counts must sum to unlabeled_count")
    if int(pre["size"]) <= unlabeled_count:
        raise ValueError("The 3D pool must leave room for all activity-recorded parents plus unlabeled_count")
    ligand_states = config["ligand_states"]
    if int(ligand_states.get("states_per_parent", 0)) != 1:
        raise ValueError("ligand_states.states_per_parent must be exactly 1")
    if int(ligand_states.get("conformers_per_state", 0)) != 1:
        raise ValueError("ligand_states.conformers_per_state must be exactly 1")
    if ligand_states.get("tautomer_method") != "rdkit_canonical":
        raise ValueError("ligand_states.tautomer_method must be rdkit_canonical")
    scoring = config["integrated_scoring"]
    if set(scoring["profiles"]) != {"wt"}:
        raise ValueError("This workflow supports only the wt integrated-scoring profile")
    if abs(sum(float(value) for value in scoring["predocking"]["weights"].values()) - 1.0) > 1e-9:
        raise ValueError("integrated_scoring.predocking weights must sum to 1")
    for profile, settings in scoring["profiles"].items():
        if abs(sum(float(value) for value in settings["weights"].values()) - 1.0) > 1e-9:
            raise ValueError(f"integrated_scoring profile {profile} weights must sum to 1")
        expected_components = {
            "wt_affinity", "postdock_3d_similarity", "docking_consensus", "qed"}
        if set(settings["weights"]) != expected_components:
            raise ValueError(
                f"integrated_scoring profile {profile} must contain exactly "
                f"{sorted(expected_components)}")
    affinity = scoring["affinity"]
    for name in ("minimum_pose_coverage", "minimum_parent_coverage"):
        value = float(affinity[name])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"integrated_scoring.affinity.{name} must be in (0,1]")
    postdock_coverage = float(scoring["minimum_postdock_3d_pose_coverage"])
    if not 0.0 < postdock_coverage <= 1.0:
        raise ValueError("minimum_postdock_3d_pose_coverage must be in (0,1]")
    configured_postdock_coverage = float(config["postdock_3d"]["minimum_pose_coverage"])
    if abs(postdock_coverage - configured_postdock_coverage) > 1e-9:
        raise ValueError("Post-docking 3D coverage thresholds must agree")
    protocols = config["docking"]["protocols"]
    identifiers = [protocol["protocol_id"] for protocol in protocols]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("docking protocol_id values must be unique")
    enabled_identifiers = {protocol["protocol_id"] for protocol in protocols if protocol.get("enabled")}
    if enabled_identifiers != {"8v1q_open_wt"}:
        raise ValueError("Only the 8v1q_open_wt docking protocol may be enabled")
    for protocol in protocols:
        if protocol.get("mutant_id", "WT") != "WT":
            raise ValueError(f"Docking protocol {protocol['protocol_id']} is not WT")
        center = protocol["box"]["center"]
        size = protocol["box"]["size"]
        if len(center) != 3 or len(size) != 3 or any(float(value) <= 0 for value in size):
            raise ValueError(f"Invalid explicit box for docking protocol {protocol['protocol_id']}")
        if not protocol.get("scope", {}).get("selection_channels"):
            raise ValueError(f"Docking protocol {protocol['protocol_id']} has no selection scope")
    missing_required = set(config["docking"].get("production_required_protocols", [])) - set(identifiers)
    if missing_required:
        raise ValueError(f"Unknown production-required docking protocols: {sorted(missing_required)}")
    unknown_excluded = set(scoring.get("excluded_protocols", [])) - set(identifiers)
    if unknown_excluded:
        raise ValueError(f"Unknown integrated-scoring excluded protocols: {sorted(unknown_excluded)}")
    required_scoring = set(scoring.get("required_protocols", []))
    unknown_scoring = required_scoring - set(identifiers)
    if unknown_scoring:
        raise ValueError(f"Unknown integrated-scoring required protocols: {sorted(unknown_scoring)}")
    if required_scoring & set(scoring.get("excluded_protocols", [])):
        raise ValueError("A docking protocol cannot be both required and excluded from scoring")
