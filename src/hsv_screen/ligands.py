from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .config import resolve_path
from .io_utils import atomic_json, prepare_stage_dir


STATE_FIELDS = ["parent_structure_key", "compound_id", "source_pool", "selection_channel",
                "state_id", "state_rank", "state_smiles", "state_origin", "formal_charge",
                "embed_status", "embedding_method"]


def _generate_task(task: tuple[dict, dict]) -> tuple[str, list[tuple[dict, str]]]:
    row, settings = task
    parent_key = row["structure_key"]
    standardized = Chem.MolFromSmiles(row["standardized_smiles"])
    if standardized is None:
        return parent_key, []

    # One parent must contribute exactly one chemical state to docking.  Prefer
    # the explicitly generated pH state; falling back to the standardized
    # parent is only for callers that intentionally disable protonation.  RDKit
    # canonical tautomer selection makes the one-state policy deterministic.
    # Source-defined stereochemistry is preserved and undefined stereochemistry
    # is not expanded into arbitrary stereoisomers.
    selected = Chem.MolFromSmiles(row.get("ph_protonated_smiles", ""))
    if selected is None:
        selected = Chem.Mol(standardized)
        origin = "standardized_parent"
    else:
        origin = f"openbabel_pH_{settings['ph']}"
    enumerator = rdMolStandardize.TautomerEnumerator()
    try:
        selected = Chem.Mol(enumerator.Canonicalize(selected))
        origin += ";rdkit_canonical_tautomer"
    except Exception:
        origin += ";tautomer_canonicalization_failed"

    smiles = Chem.MolToSmiles(selected, canonical=True, isomericSmiles=True)
    rank = 1
    state_id = f"{parent_key[:16]}_s01"
    embedded = Chem.AddHs(Chem.Mol(selected))
    status = "not_embedded"
    embedding_method = "not_embedded"
    if settings.get("embed_3d", True):
        params = AllChem.ETKDGv3()
        params.randomSeed = (int(parent_key[:8], 16) + rank) & 0x7FFFFFFF
        params.numThreads = 1
        code = AllChem.EmbedMolecule(embedded, params)
        embedding_method = "ETKDGv3"
        if code != 0:
            base_seed = (int(parent_key[:8], 16) + rank) & 0x7FFFFFFF
            expected = smiles
            code = -1
            for attempt in range(8):
                fallback = AllChem.ETKDGv3()
                fallback.randomSeed = (base_seed + 104729 * attempt) & 0x7FFFFFFF
                fallback.numThreads = 1
                fallback.useRandomCoords = True
                fallback.enforceChirality = False
                fallback.maxIterations = 100
                trial = Chem.AddHs(Chem.Mol(selected))
                try:
                    trial_code = AllChem.EmbedMolecule(trial, fallback)
                except Exception:
                    trial_code = -1
                if trial_code != 0:
                    continue
                checked = Chem.Mol(trial)
                Chem.AssignAtomChiralTagsFromStructure(
                    checked, confId=0, replaceExistingTags=True)
                observed = Chem.MolToSmiles(
                    Chem.RemoveHs(checked), canonical=True, isomericSmiles=True)
                if "@" in expected and observed != expected:
                    continue
                embedded = trial
                code = 0
                embedding_method = (
                    f"ETKDGv3_random_coords_stereo_verified_attempt_{attempt + 1}")
                break
            if code != 0:
                embedding_method = "ETKDGv3_and_verified_random_coords_failed"
        if code == 0:
            status = "ok"
            try:
                if AllChem.MMFFHasAllMoleculeParams(embedded):
                    AllChem.MMFFOptimizeMolecule(embedded, maxIters=200)
                else:
                    AllChem.UFFOptimizeMolecule(embedded, maxIters=200)
            except Exception:
                status = "ok_optimization_failed"
        else:
            status = "embed_failed"
    metadata = {
        "parent_structure_key": parent_key, "compound_id": row["compound_id"],
        "source_pool": row["source_pool"], "selection_channel": row["selection_channel"],
        "state_id": state_id, "state_rank": rank, "state_smiles": smiles,
        "state_origin": origin + ";source_stereochemistry_preserved",
        "formal_charge": sum(atom.GetFormalCharge() for atom in selected.GetAtoms()),
        "embed_status": status,
        "embedding_method": embedding_method,
    }
    for key, value in metadata.items():
        embedded.SetProp(str(key), str(value))
    embedded.SetProp("_Name", state_id)
    return parent_key, [(metadata, Chem.MolToMolBlock(embedded))]


def _init_worker() -> None:
    RDLogger.DisableLog("rdApp.*")


def protonate_parents(parents: list[dict], stage: Path, settings: dict) -> dict:
    if settings.get("protonation_method") != "openbabel":
        return {"method": "none", "converted": 0}
    obabel = shutil.which("obabel")
    if not obabel:
        raise FileNotFoundError("ligand_states.protonation_method=openbabel but obabel is unavailable")
    input_path = stage / "parents_for_protonation.smi"
    output_path = stage / "parents_pH_protonated.smi"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in parents:
            handle.write(f"{row['standardized_smiles']} {row['structure_key']}\n")
    process = subprocess.run([
        obabel, "-ismi", str(input_path), "-osmi", "-O", str(output_path), "-p", str(settings["ph"])
    ], text=True, capture_output=True)
    if process.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"Open Babel protonation failed: {process.stderr[-1000:]}")
    converted = {}
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) >= 2:
                converted[fields[1]] = fields[0]
    for row in parents:
        row["ph_protonated_smiles"] = converted.get(row["structure_key"], "")
    if len(converted) != len(parents) or any(not row["ph_protonated_smiles"] for row in parents):
        raise AssertionError(f"Open Babel protonated {len(converted)}/{len(parents)} parents")
    return {"method": "Open Babel -p", "ph": float(settings["ph"]), "converted": len(converted),
            "stderr_tail": process.stderr[-500:]}


def run(config: dict, force: bool = False) -> dict:
    started = time.time()
    root = resolve_path(config, config["output_dir"])
    stage = root / "06_ligand_states"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    final_summary = json.loads(
        (root / "05_final_selection" / "summary.json").read_text(encoding="utf-8"))
    final_size = int(final_summary["selected"])
    input_path = root / "05_final_selection" / "selected.csv"
    with input_path.open(encoding="utf-8", newline="") as handle:
        parents = list(csv.DictReader(handle))
    if len(parents) != final_size:
        raise ValueError(f"Final parent file has {len(parents)}, expected {final_size}")
    protonation = protonate_parents(parents, stage, config["ligand_states"])
    csv_path = stage / "ligand_states.csv"
    sdf_path = stage / "ligand_states.sdf"
    csv_partial, sdf_partial = Path(str(csv_path) + ".partial"), Path(str(sdf_path) + ".partial")
    state_count = 0
    proposal_count = 0
    parent_states = Counter()
    state_ids = set()
    statuses = Counter()
    with csv_partial.open("w", encoding="utf-8", newline="") as csv_handle, sdf_partial.open("w", encoding="utf-8") as sdf_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=STATE_FIELDS)
        writer.writeheader()
        tasks = ((row, config["ligand_states"]) for row in parents)
        with mp.Pool(int(config["workers"]), initializer=_init_worker) as pool:
            for count, (parent_key, states) in enumerate(pool.imap(_generate_task, tasks, chunksize=8), start=1):
                for metadata, mol_block in states:
                    if metadata["state_id"] in state_ids:
                        raise AssertionError(f"Duplicate state ID {metadata['state_id']}")
                    state_ids.add(metadata["state_id"])
                    writer.writerow(metadata)
                    proposal_count += 1
                    statuses[metadata["embed_status"]] += 1
                    if metadata["embed_status"].startswith("ok"):
                        sdf_handle.write(mol_block)
                        for key, value in metadata.items():
                            sdf_handle.write(f">  <{key}>\n{value}\n\n")
                        sdf_handle.write("$$$$\n")
                        state_count += 1
                        parent_states[parent_key] += 1
                if count % 1000 == 0:
                    print(f"[ligands] {count:,}/{len(parents):,} parents; {state_count:,} states", flush=True)
    os.replace(csv_partial, csv_path)
    os.replace(sdf_partial, sdf_path)
    missing = [row["structure_key"] for row in parents if parent_states[row["structure_key"]] < 1]
    if missing:
        raise AssertionError(f"{len(missing)} parents have no generated ligand state")
    summary = {
        "stage": "ligand_states", "parent_count": len(parents), "state_count": state_count,
        "state_proposal_count": proposal_count, "failed_state_proposals": proposal_count - state_count,
        "protonation": protonation,
        "states_per_parent": dict(Counter(parent_states.values())), "embed_statuses": dict(statuses),
        "all_parents_represented": True,
        "single_state_policy": True,
        "interpretation": "Exactly one state per parent: Open Babel pH adjustment, deterministic RDKit canonical tautomer selection, source stereochemistry preservation, and one ETKDGv3 starting conformer. This is a reproducible screening state, not a microscopic pKa population model.",
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(done, summary)
    return summary
