from __future__ import annotations

import csv
import json
from pathlib import Path

from rdkit import Chem

from .config import resolve_path
from .io_utils import atomic_json, iter_csv
from .reference import validate_and_prepare


def run(config: dict) -> dict:
    root = resolve_path(config, config["output_dir"])
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"QC failed: {name}: {detail}")

    reference = validate_and_prepare(config, prepare_pdbqt=False)
    check("reference_bound_instance", reference["ligand_instances"].get("C:1201") == 30, reference["ligand_instances"])
    check("target_structures_are_h9e937", all(
        item["uniprot_id"] == "H9E937" for item in reference["target_structures"]),
        reference["target_structures"])
    receptor_paths = sorted({
        resolve_path(config, item["receptor_pdbqt"])
        for item in config["docking"]["protocols"] if item.get("enabled")
    })
    receptor_details = []
    receptors_rigid = True
    for receptor_pdbqt in receptor_paths:
        pdbqt_text = receptor_pdbqt.read_text(encoding="utf-8", errors="replace")
        rigid = not any(token in pdbqt_text for token in ("ROOT", "BRANCH", "TORSDOF"))
        atom_lines = sum(line.startswith(("ATOM", "HETATM")) for line in pdbqt_text.splitlines())
        receptors_rigid = receptors_rigid and rigid and atom_lines > 0
        receptor_details.append({"path": str(receptor_pdbqt), "atom_lines": atom_lines, "rigid": rigid})
    check("single_8v1q_receptor_is_rigid", receptors_rigid and len(receptor_details) == 1
          and "8V1Q" in receptor_details[0]["path"],
          receptor_details)
    standardized = json.loads((root / "01_standardized/summary.json").read_text(encoding="utf-8"))
    expected_sample = int(config["sample"]["active_records"]) + int(config["sample"]["candidate_records"])
    if config["sample"]["enabled"]:
        check("sample_input_count", standardized["counts"]["input_records"] == expected_sample, standardized["counts"]["input_records"])
    mapping_count = sum(1 for _ in iter_csv(root / "01_standardized/parent_mapping.csv.gz"))
    check("mapping_traceability", mapping_count == standardized["counts"]["input_records"], mapping_count)
    scored = json.loads((root / "02_2d_scored/summary.json").read_text(encoding="utf-8"))
    check("pnu_2d_self_score", abs(float(scored["pnu_self_check"]["pnu_consensus"]) - 1.0) < 1e-9,
          scored["pnu_self_check"]["pnu_consensus"])
    pool = json.loads((root / "03_3d_pool/summary.json").read_text(encoding="utf-8"))
    check("3d_pool_size", pool["target_size"] == int(config["prescreen_3d_pool"]["size"]), pool["target_size"])
    check("3d_pool_exact_unique", pool["unique_structures"] == pool["target_size"], pool["unique_structures"])
    check("all_hard_filtered_activity_in_3d_pool",
          pool.get("all_hard_filtered_activity_included", False)
          and pool["source_targets"]["activity_recorded"] == pool["hard_filter_available"]["activity_recorded"],
          {"available": pool["hard_filter_available"]["activity_recorded"],
           "selected": pool["source_targets"]["activity_recorded"]})
    scored3d = json.loads((root / "04_3d_scored/summary.json").read_text(encoding="utf-8"))
    check("3d_has_success", scored3d["statuses"].get("ok", 0) > 0, scored3d["statuses"])
    check("3d_reference_self", scored3d["experimental_pnu_self_check"]["combined"] >= 0.80,
          scored3d["experimental_pnu_self_check"])
    final = json.loads((root / "05_final_selection/summary.json").read_text(encoding="utf-8"))
    expected_final_size = int(final["activity_3d_feasible"]) + int(config["final_selection"]["unlabeled_count"])
    check("dynamic_final_size", final["selected"] == expected_final_size, final["selected"])
    check("all_3d_feasible_activity_included",
          final["activity_included"] == final["activity_3d_feasible"]
          and final["activity_3d_failure_count"] == 0
          and final["activity_in_3d_pool"] == final["activity_included"]
          and final["source_counts"].get("activity_recorded") == final["activity_3d_feasible"],
          {"in_3d_pool": final["activity_in_3d_pool"],
           "feasible": final["activity_3d_feasible"], "included": final["activity_included"],
           "failures": final["activity_3d_failure_count"]})
    check("fixed_unlabeled_quota",
          final["source_counts"].get("unlabeled") == int(config["final_selection"]["unlabeled_count"]),
          final["source_counts"])
    expected_unlabeled_channels = {
        key: int(value) for key, value in config["final_selection"]["unlabeled_channel_counts"].items()}
    actual_unlabeled_channels = {
        key.split("|", 1)[1]: value for key, value in final["source_channel_counts"].items()
        if key.startswith("unlabeled|")}
    check("unlabeled_channel_quotas", actual_unlabeled_channels == expected_unlabeled_channels,
          actual_unlabeled_channels)
    check("unlabeled_scaffold_cap",
          final["maximum_unlabeled_scaffold_multiplicity"] <= int(config["final_selection"]["global_scaffold_cap"]),
          final["maximum_unlabeled_scaffold_multiplicity"])
    ligands = json.loads((root / "06_ligand_states/summary.json").read_text(encoding="utf-8"))
    check("all_parents_have_states", ligands["all_parents_represented"] and ligands["parent_count"] == final["selected"], ligands)
    check("exactly_one_state_per_parent",
          ligands.get("single_state_policy") is True
          and ligands["state_count"] == ligands["parent_count"]
          and ligands["state_proposal_count"] == ligands["parent_count"]
          and ligands["states_per_parent"] == {"1": ligands["parent_count"]},
          {"parent_count": ligands["parent_count"], "state_count": ligands["state_count"],
           "state_proposal_count": ligands["state_proposal_count"],
           "states_per_parent": ligands["states_per_parent"]})
    embedded_count = sum(count for status, count in ligands["embed_statuses"].items() if status.startswith("ok"))
    check("only_embedded_states_enter_smina", embedded_count == ligands["state_count"], ligands["embed_statuses"])
    sdf_count = sum(1 for mol in Chem.SDMolSupplier(str(root / "06_ligand_states/ligand_states.sdf"), removeHs=False) if mol is not None)
    check("ligand_sdf_count", sdf_count == ligands["state_count"], sdf_count)
    docking = json.loads((root / "07_smina/manifest.json").read_text(encoding="utf-8"))
    check("smina_input_state_count", docking["input_state_count"] == ligands["state_count"], docking)
    check("smina_has_scheduled_protocols", docking["job_count"] > 0 and docking["scheduled_state_count"] > 0, docking)
    enabled_protocols = {item["protocol_id"] for item in docking["protocols"]}
    expected_protocols = {"8v1q_open_wt"}
    check("only_8v1q_wt_protocol", enabled_protocols == expected_protocols and all(
        protocol.get("mutant_id", "WT") == "WT"
        for protocol in config["docking"]["protocols"] if protocol.get("enabled")),
        sorted(enabled_protocols))
    with Path(docking["jobs"]).open(encoding="utf-8", newline="") as handle:
        job_rows = list(csv.DictReader(handle, delimiter="\t"))
    seeds = [row.get("seed", "") for row in job_rows]
    check("reproducible_smina_seeds", len(seeds) == docking["job_count"]
          and all(seed.isdigit() and 0 < int(seed) < 2147483647 for seed in seeds)
          and len(seeds) == len(set(seeds)), seeds[:5])
    check("smina_protocol_scope_bounded", all(
        item["state_count"] <= docking["input_state_count"] for item in docking["protocols"]), docking["protocols"])
    status = "PASS" if docking["production_ready"] else "PRE_DOCKING_PASS_PRODUCTION_BLOCKED"
    report = {"status": status, "checks": checks, "passed": len(checks), "failed": 0,
              "production_ready": docking["production_ready"],
              "missing_required_protocols": docking["missing_required_protocols"]}
    atomic_json(root / "QC_REPORT.json", report)
    markdown = ["# QC report", "", f"Status: **{status}**", ""]
    markdown.extend(f"- PASS — {item['name']}: `{item['detail']}`" for item in checks)
    (root / "QC_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report
