from __future__ import annotations

import csv
import gzip
import json
import math
import multiprocessing as mp
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, RDLogger, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures, rdShapeHelpers

from .config import resolve_path
from .io_utils import atomic_json, iter_csv, prepare_stage_dir
from .screen2d import SCORED_FIELDS


THREED_FIELDS = [
    "structure3d_status", "structure3d_score", "structure3d_feature_score",
    "structure3d_shape_similarity", "structure3d_clash_fraction",
    "structure3d_best_conformer", "structure3d_conformer_count", "structure3d_embedding_method",
]
STATE: dict[str, object] = {}


def parse_pocket_atoms(pdb_path: Path, excluded_resname: str, reference: Chem.Mol, radius: float) -> list[list[float]]:
    ref_positions = np.array(reference.GetConformer().GetPositions(), dtype=float)
    positions = []
    with pdb_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[17:20].strip() in {excluded_resname, "HOH", "DOD"}:
                continue
            if line[76:78].strip().upper() in {"H", "D"}:
                continue
            try:
                point = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if float(np.min(np.linalg.norm(ref_positions - point, axis=1))) <= radius:
                positions.append(point.tolist())
    if not positions:
        raise ValueError("No receptor/DNA pocket atoms found around PNU")
    return positions


def valid_features(factory, mol: Chem.Mol, conf_id: int) -> list[dict]:
    allowed = {"Donor", "Acceptor", "Aromatic", "PosIonizable", "NegIonizable"}
    result = []
    for feature in factory.GetFeaturesForMol(mol, confId=conf_id):
        family = feature.GetFamily()
        if family not in allowed:
            continue
        if family == "Donor" and not any(
            mol.GetAtomWithIdx(index).GetTotalNumHs(includeNeighbors=True) > 0 for index in feature.GetAtomIds()
        ):
            continue
        result.append({"family": family, "position": list(feature.GetPos()), "atom_ids": list(feature.GetAtomIds())})
    return result


def build_feature_model(reference: Chem.Mol, pocket_positions: list[list[float]], settings: dict) -> list[dict]:
    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    pocket = np.array(pocket_positions)
    features = []
    for feature in valid_features(factory, reference, 0):
        distance = float(np.min(np.linalg.norm(pocket - np.array(feature["position"]), axis=1)))
        cutoff = float(settings["feature_contact_cutoffs"][feature["family"]])
        if distance <= cutoff:
            features.append({
                **feature, "nearest_contact_distance": distance,
                "tolerance": float(settings["feature_tolerances"][feature["family"]]),
                "weight": float(settings["feature_weights"][feature["family"]]),
            })
    if not features:
        raise ValueError("No contact-filtered PNU pharmacophore features were generated")
    return features


def init_worker(reference_block: str, pocket_positions: list[list[float]], features: list[dict], settings: dict) -> None:
    RDLogger.DisableLog("rdApp.*")
    reference = Chem.MolFromMolBlock(reference_block, sanitize=True, removeHs=False)
    STATE.clear()
    STATE.update({
        "reference": reference, "pocket": np.array(pocket_positions, dtype=float),
        "features": features, "settings": settings,
        "factory": ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")),
    })


def feature_score(candidate_features: list[dict]) -> float:
    refs = STATE["features"]
    total = sum(float(feature["weight"]) for feature in refs)
    score = 0.0
    for family in {feature["family"] for feature in refs}:
        pairs = []
        for ri, ref in enumerate(refs):
            if ref["family"] != family:
                continue
            for ci, candidate in enumerate(candidate_features):
                if candidate["family"] != family:
                    continue
                distance = float(np.linalg.norm(np.array(ref["position"]) - np.array(candidate["position"])))
                similarity = math.exp(-0.5 * (distance / float(ref["tolerance"])) ** 2)
                pairs.append((similarity, ri, ci))
        used_ref, used_candidate = set(), set()
        for similarity, ri, ci in sorted(pairs, reverse=True):
            if ri in used_ref or ci in used_candidate:
                continue
            used_ref.add(ri)
            used_candidate.add(ci)
            score += similarity * float(refs[ri]["weight"])
    return score / total if total else 0.0


def score_conformation_similarity(mol: Chem.Mol, conf_id: int) -> tuple[float, float, float, float]:
    """Align one existing conformation to PNU and return combined/feature/shape/O3A scores."""
    reference = STATE["reference"]
    try:
        alignment = AllChem.GetCrippenO3A(mol, reference, prbCid=conf_id, refCid=0)
        o3a_score = float(alignment.Score())
        alignment.Align()
    except Exception:
        return -1.0, 0.0, 0.0, 0.0
    shape = 1.0 - float(rdShapeHelpers.ShapeTanimotoDist(reference, mol, confId1=0, confId2=conf_id, ignoreHs=True))
    features = feature_score(valid_features(STATE["factory"], mol, conf_id))
    combined = 0.60 * features + 0.40 * shape
    return combined, features, shape, o3a_score


def score_conformer(mol: Chem.Mol, conf_id: int) -> tuple[float, float, float, float]:
    combined, features, shape, _ = score_conformation_similarity(mol, conf_id)
    if combined < 0:
        return -1.0, 0.0, 0.0, 1.0
    conformer = mol.GetConformer(conf_id)
    positions = np.array([list(conformer.GetAtomPosition(atom.GetIdx())) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
    pocket = STATE["pocket"]
    distances = np.min(np.linalg.norm(positions[:, None, :] - pocket[None, :, :], axis=2), axis=1)
    clash = float(np.mean(distances < float(STATE["settings"]["clash_distance"])))
    combined *= max(0.0, 1.0 - 0.80 * clash)
    return combined, features, shape, clash


def score_task(task: tuple[int, str, str, str]) -> tuple[int, dict]:
    index, smiles, structure_key, prescreen_channel = task
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return index, {"structure3d_status": "invalid_smiles"}
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(structure_key[:8], 16) & 0x7FFFFFFF
    params.pruneRmsThresh = float(STATE["settings"]["prune_rms_threshold"])
    params.numThreads = 1
    try:
        conformers = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(STATE["settings"]["num_conformers"]), params=params))
    except Exception as exc:
        return index, {"structure3d_status": f"embed_error:{type(exc).__name__}"}
    embedding_method = "ETKDGv3"
    if not conformers:
        # Highly bridged, stereo-rich structures can fail ETKDG's constrained
        # bounds. Try a small bounded set of random-coordinate embeddings and accept one
        # only when coordinates reproduce the specified stereochemistry.
        base_seed = int(structure_key[:8], 16) & 0x7FFFFFFF
        expected = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
        for attempt in range(8):
            fallback = AllChem.ETKDGv3()
            fallback.randomSeed = (base_seed + 104729 * attempt) & 0x7FFFFFFF
            fallback.numThreads = 1
            fallback.useRandomCoords = True
            fallback.enforceChirality = False
            fallback.maxIterations = 100
            trial = Chem.Mol(mol)
            try:
                code = AllChem.EmbedMolecule(trial, fallback)
            except Exception:
                code = -1
            if code != 0:
                continue
            checked = Chem.Mol(trial)
            Chem.AssignAtomChiralTagsFromStructure(checked, confId=0, replaceExistingTags=True)
            observed = Chem.MolToSmiles(Chem.RemoveHs(checked), canonical=True, isomericSmiles=True)
            if expected == observed:
                mol = trial
                conformers = [0]
                embedding_method = f"ETKDGv3_random_coords_stereo_verified_attempt_{attempt + 1}"
                break
        if not conformers:
            return index, {
                "structure3d_status": "embed_failed",
                "structure3d_embedding_method": "ETKDGv3_and_verified_random_coords_failed",
            }
    if prescreen_channel not in set(STATE["settings"].get(
            "pnu_alignment_channels", ["pnu_2d", "pharm2d"])):
        return index, {
            "structure3d_status": "feasible_only", "structure3d_score": 0.0,
            "structure3d_feature_score": 0.0, "structure3d_shape_similarity": 0.0,
            "structure3d_clash_fraction": 0.0, "structure3d_best_conformer": int(conformers[0]),
            "structure3d_conformer_count": len(conformers), "structure3d_embedding_method": embedding_method,
        }
    values = [(score_conformer(mol, int(conf_id)), int(conf_id)) for conf_id in conformers]
    (score, feature, shape, clash), best = max(values, key=lambda value: value[0][0])
    return index, {
        "structure3d_status": "ok" if score >= 0 else "alignment_failed",
        "structure3d_score": max(score, 0.0), "structure3d_feature_score": feature,
        "structure3d_shape_similarity": shape, "structure3d_clash_fraction": clash,
        "structure3d_best_conformer": best, "structure3d_conformer_count": len(conformers),
        "structure3d_embedding_method": embedding_method,
    }


def run(config: dict, force: bool = False) -> dict:
    root = resolve_path(config, config["output_dir"])
    source = root / "03_3d_pool"
    stage = root / "04_3d_scored"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    pool_size = int(config["prescreen_3d_pool"]["size"])
    input_path = source / f"pool_{pool_size}.csv.gz"
    rows = [dict(row) for row in iter_csv(input_path)]
    if len(rows) != pool_size:
        raise ValueError(f"3D pool contains {len(rows)}, expected {pool_size}")
    refs = config["references"]
    bound = Chem.SDMolSupplier(str(resolve_path(config, refs["bound_ligand_sdf"])), removeHs=False)[0]
    settings = config["structure_3d"]
    pocket = parse_pocket_atoms(resolve_path(config, refs["pdb_file"]), refs["ligand_resname"], bound, float(settings["pocket_radius"]))
    features = build_feature_model(bound, pocket, settings)
    reference_block = Chem.MolToMolBlock(bound)
    init_worker(reference_block, pocket, features, settings)
    self_values = score_conformer(Chem.Mol(bound), 0)
    if self_values[0] < 0.80 or self_values[1] < 0.95 or self_values[2] < 0.99:
        raise AssertionError(f"Experimental PNU 3D self-check failed: {self_values}")
    started = time.time()
    status_counts = Counter()
    tasks = ((index, row["standardized_smiles"], row["structure_key"], row["prescreen_channel"])
             for index, row in enumerate(rows))
    with mp.Pool(int(config["workers"]), initializer=init_worker, initargs=(reference_block, pocket, features, settings)) as pool:
        for count, (index, result) in enumerate(pool.imap(score_task, tasks, chunksize=8), start=1):
            rows[index].update(result)
            status_counts[result["structure3d_status"]] += 1
            if count % 1000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(f"[3d] {count:,}/{len(rows):,}; {count/elapsed:,.1f}/s", flush=True)
    fields = list(rows[0])
    for field in THREED_FIELDS:
        if field not in fields:
            fields.append(field)
    output = stage / f"scored_{pool_size}.csv.gz"
    partial = Path(str(output) + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8", newline="", compresslevel=3) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, output)
    atomic_json(stage / "pharmacophore_features.json", {"features": features, "pocket_atom_count": len(pocket)})
    summary = {
        "stage": "structure_3d", "input_count": len(rows), "statuses": dict(status_counts),
        "experimental_pnu_self_check": {"combined": self_values[0], "feature": self_values[1], "shape": self_values[2], "clash": self_values[3]},
        "method": "All channels receive ETKDGv3 feasibility checks. Only configured PNU/Pharm2D channels receive Crippen O3A plus experimental 7LUF YE4 feature/shape/clash scoring; diversity and exploration are not rejected for PNU mismatch.",
        "not_docking": True, "elapsed_seconds": time.time() - started,
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
    }
    if status_counts["ok"] == 0:
        raise AssertionError("No 3D candidate completed successfully")
    atomic_json(done, summary)
    return summary
