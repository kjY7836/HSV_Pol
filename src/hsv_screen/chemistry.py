from __future__ import annotations

import hashlib
import math
import os
from collections import Counter

from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import (
    ChemicalFeatures,
    Crippen,
    Descriptors,
    FilterCatalog,
    Lipinski,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.Scaffolds import MurckoScaffold


# These are removable only when exactly one plausible organic parent remains.
# Multi-organic systems outside this explicit list are quarantined, never merged.
KNOWN_COUNTERION_SMILES = {
    "[Cl-]", "[Br-]", "[I-]", "[F-]", "Cl", "Br", "I", "O", "N",
    "[Na+]", "[K+]", "[Li+]", "[Ca+2]", "[Mg+2]", "[Zn+2]",
    "O=S(=O)(O)O", "O=P(O)(O)O", "O=C(O)C", "CS(=O)(=O)O",
    "O=C(O)C(=O)O", "O=C(O)/C=C/C(=O)O", "O=C(O)/C=C\\C(=O)O",
    "O=C(O)CCC(=O)O", "O=C(O)C(O)C(O)C(=O)O",
    "O=C(O)C(O)(CC(=O)O)CC(=O)O", "O=C(O)c1ccccc1",
    "Cc1ccc(S(=O)(=O)O)cc1", "CCO", "CO", "CC(C)O", "CC(=O)C",
    "CCOC(C)=O", "CS(C)=O", "CN(C)C=O", "C1CCOC1", "C1COCCO1",
}
METAL_ATOMIC_NUMBERS = {3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                        28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                        46, 47, 48, 49, 50, 55, 56, 57, 78, 79, 80, 81, 82}


def _canonical_set(values: set[str]) -> set[str]:
    output = set()
    for value in values:
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            output.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return output


KNOWN_COUNTERIONS = _canonical_set(KNOWN_COUNTERION_SMILES)


def structure_hash(smiles: str) -> str:
    return hashlib.sha1(smiles.encode("utf-8")).hexdigest()


def _fragment_info(mol: Chem.Mol) -> list[dict]:
    result = []
    for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
        smiles = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        atoms = [atom.GetAtomicNum() for atom in fragment.GetAtoms()]
        result.append({
            "mol": fragment,
            "smiles": smiles,
            "heavy_atoms": fragment.GetNumHeavyAtoms(),
            "has_carbon": 6 in atoms,
            "has_metal": bool(set(atoms) & METAL_ATOMIC_NUMBERS),
            "known_counterion": smiles in KNOWN_COUNTERIONS,
        })
    return result


def classify_and_standardize(record: dict, settings: dict) -> dict:
    result = dict(record)
    result.update({
        "status": "", "component_class": "", "original_fragment_count": 0,
        "removed_component_smiles": "", "charged_parent_smiles": "",
        "standardized_smiles": "", "connectivity_smiles": "", "structure_key": "",
        "connectivity_key": "", "std_mol_weight": "", "heavy_atom_count": "",
        "formal_charge": "", "has_permanent_charge": 0, "model_eligible": 0,
    })
    raw = record.get("original_smiles", "").strip()
    if not raw:
        result["status"] = "empty_smiles"
        return result
    try:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            result["status"] = "invalid_smiles"
            return result
        infos = _fragment_info(mol)
        result["original_fragment_count"] = len(infos)
        if any(item["has_metal"] and not item["known_counterion"] for item in infos):
            result.update(status="quarantine_special_element", component_class="metal_or_special")
            return result

        if len(infos) == 1:
            parent_info = infos[0]
            component_class = "single_component"
        else:
            plausible = [item for item in infos if item["has_carbon"] and not item["known_counterion"]]
            removable = [item for item in infos if item["known_counterion"]]
            if len(plausible) != 1 or len(removable) != len(infos) - 1:
                result.update(status="quarantine_ambiguous_components", component_class="ambiguous_multi_component")
                return result
            parent_info = plausible[0]
            organic_removed = any(item["has_carbon"] for item in removable)
            component_class = "organic_counterion_salt" if organic_removed else "simple_salt_or_solvate"
            result["removed_component_smiles"] = ".".join(sorted(item["smiles"] for item in removable))

        if not parent_info["has_carbon"]:
            result.update(status="quarantine_no_organic_parent", component_class="no_organic_parent")
            return result
        cleaned = rdMolStandardize.Cleanup(parent_info["mol"])
        charged_smiles = Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)
        uncharged = rdMolStandardize.Uncharger().uncharge(Chem.Mol(cleaned))
        Chem.SanitizeMol(uncharged)
        Chem.AssignStereochemistry(uncharged, cleanIt=True, force=True)
        smiles = Chem.MolToSmiles(uncharged, canonical=True, isomericSmiles=True)
        connectivity = Chem.MolToSmiles(uncharged, canonical=True, isomericSmiles=False)
        atoms = {atom.GetAtomicNum() for atom in uncharged.GetAtoms()}
        allowed = set(settings["allowed_parent_atomic_numbers"])
        heavy = uncharged.GetNumHeavyAtoms()
        charge = sum(atom.GetFormalCharge() for atom in uncharged.GetAtoms())
        result.update({
            "status": "valid_parent" if atoms.issubset(allowed) else "quarantine_special_element",
            "component_class": component_class,
            "charged_parent_smiles": charged_smiles,
            "standardized_smiles": smiles,
            "connectivity_smiles": connectivity,
            "structure_key": structure_hash(smiles),
            "connectivity_key": structure_hash(connectivity),
            "std_mol_weight": f"{Descriptors.MolWt(uncharged):.6f}",
            "heavy_atom_count": heavy,
            "formal_charge": charge,
            "has_permanent_charge": int(charge != 0),
            "model_eligible": int(atoms.issubset(allowed)),
        })
        return result
    except Exception as exc:
        result["status"] = f"standardization_error:{type(exc).__name__}"
        return result


def make_filter_catalog(kind):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(kind)
    return FilterCatalog.FilterCatalog(params)


def alert_description(catalog, mol: Chem.Mol) -> str:
    return "|".join(sorted({match.GetDescription() for match in catalog.GetMatches(mol)}))


def hard_filter_reasons(props: dict, atomic_numbers: set[int], filters: dict) -> list[str]:
    reasons = []
    if not atomic_numbers.issubset(set(filters["allowed_atomic_numbers"])):
        reasons.append("disallowed_element")
    checks = [
        (props["heavy_atoms"] < filters["min_heavy_atoms"], "heavy_atoms_low"),
        (props["heavy_atoms"] > filters["max_heavy_atoms"], "heavy_atoms_high"),
        (props["mw"] < filters["min_mw"], "mw_low"),
        (props["mw"] > filters["max_mw"], "mw_high"),
        (props["clogp"] < filters["min_clogp"], "clogp_low"),
        (props["clogp"] > filters["max_clogp"], "clogp_high"),
        (props["tpsa"] < filters["min_tpsa"], "tpsa_low"),
        (props["tpsa"] > filters["max_tpsa"], "tpsa_high"),
        (props["hbd"] > filters["max_hbd"], "hbd_high"),
        (props["hba"] > filters["max_hba"], "hba_high"),
        (props["rotatable_bonds"] > filters["max_rotatable_bonds"], "rotatable_bonds_high"),
        (props["ring_count"] > filters["max_rings"], "ring_count_high"),
        (abs(props["formal_charge"]) > filters["max_abs_formal_charge"], "formal_charge_extreme"),
    ]
    reasons.extend(label for failed, label in checks if failed)
    return reasons


SCORE_STATE: dict[str, object] = {}


def init_score_worker(ref_smiles: str, fp_settings: dict, filters: dict,
                      evidence_settings: dict | None = None) -> None:
    RDLogger.DisableLog("rdApp.*")
    ref = Chem.MolFromSmiles(ref_smiles)
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=int(fp_settings["radius"]), fpSize=int(fp_settings["n_bits"]))
    feature = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fp_settings["radius"]), fpSize=int(fp_settings["n_bits"]),
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    SCORE_STATE.clear()
    SCORE_STATE.update({
        "filters": filters, "morgan": morgan, "feature": feature,
        "ref_ecfp": morgan.GetFingerprint(ref), "ref_fcfp": feature.GetFingerprint(ref),
        "ref_pharm": Generate.Gen2DFingerprint(ref, Gobbi_Pharm2D.factory),
        "pains": make_filter_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS),
        "brenk": make_filter_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK),
        "evidence_weights": (evidence_settings or {}).get(
            "weights", {"pnu_consensus": 0.65, "quality_score": 0.35}),
    })


def scaffold_for(mol: Chem.Mol, structure_key: str) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or f"ACYCLIC:{structure_key}"


def score_parent(source: dict) -> dict | None:
    mol = Chem.MolFromSmiles(source["standardized_smiles"])
    if mol is None:
        return None
    props = {
        "mw": float(Descriptors.MolWt(mol)), "clogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)), "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)), "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "ring_count": int(Lipinski.RingCount(mol)), "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)), "qed": float(QED.qed(mol)),
    }
    reasons = hard_filter_reasons(props, {a.GetAtomicNum() for a in mol.GetAtoms()}, SCORE_STATE["filters"])
    pains = alert_description(SCORE_STATE["pains"], mol)
    brenk = alert_description(SCORE_STATE["brenk"], mol)
    ecfp = SCORE_STATE["morgan"].GetFingerprint(mol)
    fcfp = SCORE_STATE["feature"].GetFingerprint(mol)
    pharm = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    ecfp_sim = float(DataStructs.TanimotoSimilarity(SCORE_STATE["ref_ecfp"], ecfp))
    fcfp_sim = float(DataStructs.TanimotoSimilarity(SCORE_STATE["ref_fcfp"], fcfp))
    pharm_sim = float(DataStructs.TanimotoSimilarity(SCORE_STATE["ref_pharm"], pharm))
    alerts = int(bool(pains)) + int(bool(brenk))
    pnu_consensus = 0.55 * ecfp_sim + 0.20 * fcfp_sim + 0.25 * pharm_sim
    quality_score = max(0.0, props["qed"] - 0.08 * alerts)
    evidence_weights = SCORE_STATE["evidence_weights"]
    library_evidence_score = (
        float(evidence_weights.get("pnu_consensus", 0.0)) * pnu_consensus
        + float(evidence_weights.get("quality_score", 0.0)) * quality_score
    )
    return {
        **source, **props, "hard_filter_pass": int(not reasons), "hard_filter_reasons": "|".join(reasons),
        "pains_alert": pains, "brenk_alert": brenk, "soft_alert_count": alerts,
        "murcko_scaffold": scaffold_for(mol, source["structure_key"]),
        "pnu_ecfp_similarity": ecfp_sim, "pnu_fcfp_similarity": fcfp_sim,
        "pharm2d_similarity": pharm_sim,
        "pnu_consensus": pnu_consensus, "quality_score": quality_score,
        "library_evidence_score": library_evidence_score,
    }


def score_reference_self(ref_smiles: str, fp_settings: dict, filters: dict,
                         evidence_settings: dict | None = None) -> dict:
    init_score_worker(ref_smiles, fp_settings, filters, evidence_settings)
    record = {"compound_id": "PNU-183792", "standardized_smiles": ref_smiles,
              "structure_key": structure_hash(ref_smiles), "source_pool": "reference"}
    return score_parent(record)
