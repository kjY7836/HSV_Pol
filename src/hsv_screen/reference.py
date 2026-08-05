from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

from .config import resolve_path
from .io_utils import atomic_json


PNU_SMILES = "CN1C=C(C(=O)C2=C1C=CC(=C2)CN3CCOCC3)C(=O)NCC4=CC=C(C=C4)Cl"
EXPECTED_SHA256 = {
    "7LUF.pdb": "a62ce9ea817ccc3c7519df99dffcddfd86b73589899eaba2198d3f07f3928aed",
    "7LUF.cif": "c82d09d8fb7ef87d93feab8caf1c90b51d76a723a04fd324e58480a9eb1ca8f3",
    "YE4.cif": "dadc3d881dcf4c9d6e862e88ef07debcd74ce080755863300de1ff4f0ebf957a",
    "YE4_ideal.sdf": "e1b55fa888b903e6dea4940b358b8e8138d8b594746cc15f9345b93e895307b4",
    "PNU-183792_7LUF_bound.sdf": "2db65afe0ec219e4a7ac1bb2d35a7d0d86071ea8b272903d6ff16e65ba2d7c34",
    "8V1Q.pdb": "f80e211f7d45fd30cb31e10d03ca6c76ed1a110e38500640d37a411ab995a7fc",
    "8V1Q.cif": "338c306634aba6df76838bc47b2d32f7a05f2bddaa254100f52dc225dbd3e467",
    "8V1R.pdb": "5fb5d0881ae2a4446b568f107ad7b604c2ca760dec972950093bf3705017ced7",
    "8V1R.cif": "4436e12a0fd434346f0142d2f462971ae02b09d7d93c5628b615fac426350313",
    "dTTP_8V1R_bound.sdf": "8a83b622ed41beb22911423a5c643b4b985f1b9548501fac6b3ad885bb59e102",
}
SOURCE_URLS = {
    "7LUF.pdb": "https://files.rcsb.org/download/7LUF.pdb",
    "7LUF.cif": "https://files.rcsb.org/download/7LUF.cif",
    "YE4.cif": "https://files.rcsb.org/ligands/download/YE4.cif",
    "YE4_ideal.sdf": "https://files.rcsb.org/ligands/download/YE4_ideal.sdf",
    "PNU-183792_7LUF_bound.sdf": "https://models.rcsb.org/v1/7luf/ligand?auth_asym_id=C&auth_seq_id=1201&encoding=sdf",
    "8V1Q.pdb": "https://files.rcsb.org/download/8V1Q.pdb",
    "8V1Q.cif": "https://files.rcsb.org/download/8V1Q.cif",
    "8V1R.pdb": "https://files.rcsb.org/download/8V1R.pdb",
    "8V1R.cif": "https://files.rcsb.org/download/8V1R.cif",
    "dTTP_8V1R_bound.sdf": "https://models.rcsb.org/v1/8v1r/ligand?auth_asym_id=A&auth_seq_id=1306&encoding=sdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ligand_instances(pdb_path: Path, resname: str) -> dict[str, int]:
    result: dict[str, int] = {}
    with pdb_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("HETATM") and line[17:20].strip() == resname and line[76:78].strip().upper() != "H":
                key = f"{line[21].strip()}:{int(line[22:26])}"
                result[key] = result.get(key, 0) + 1
    return result


def _write_target_receptor(pdb_path: Path, output: Path, receptor_chains: set[str],
                           remove_resnames: set[str]) -> dict:
    lines: list[str] = []
    removed: dict[str, int] = {}
    excluded_chains: dict[str, int] = {}
    included: dict[str, int] = {}
    with pdb_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip()
            residue = line[17:20].strip()
            if chain not in receptor_chains:
                excluded_chains[chain or "NONE"] = excluded_chains.get(chain or "NONE", 0) + 1
                continue
            if residue in remove_resnames:
                removed[residue or "UNKNOWN"] = removed.get(residue or "UNKNOWN", 0) + 1
                continue
            lines.append(line)
            included[chain or "NONE"] = included.get(chain or "NONE", 0) + 1
    if not lines:
        raise ValueError(f"Target receptor selection produced no atoms for {pdb_path}")
    lines.append("END\n")
    partial = Path(str(output) + ".partial")
    partial.write_text("".join(lines), encoding="utf-8")
    partial.replace(output)
    return {
        "included_chains": sorted(receptor_chains), "included_atom_counts": included,
        "excluded_chain_atom_counts": excluded_chains,
        "removed_resnames": sorted(remove_resnames), "removed_atom_counts": removed,
    }


def _environment_executable(name: str) -> str | None:
    adjacent = Path(sys.executable).resolve().parent / name
    return str(adjacent) if adjacent.exists() else shutil.which(name)


def _prepare_rigid_pdbqt(receptor_path: Path, pdbqt_path: Path) -> str:
    obabel = _environment_executable("obabel")
    if not obabel:
        return "obabel_missing"
    if pdbqt_path.exists():
        pdbqt_path.unlink()
    process = subprocess.run(
        [obabel, "-ipdb", str(receptor_path), "-opdbqt", "-O", str(pdbqt_path), "-h", "-xrcn"],
        text=True, capture_output=True)
    if process.returncode != 0 or not pdbqt_path.exists():
        return f"failed:{process.stderr[-500:]}"
    pdbqt_text = pdbqt_path.read_text(encoding="utf-8", errors="replace")
    if any(token in pdbqt_text for token in ("ROOT", "BRANCH", "TORSDOF")):
        raise ValueError(f"{pdbqt_path} was written as a flexible ligand instead of a rigid receptor")
    atom_count = sum(line.startswith(("ATOM", "HETATM")) for line in pdbqt_text.splitlines())
    if atom_count == 0:
        raise ValueError(f"No receptor atoms were written to {pdbqt_path}")
    return "ok"


def _has_uniprot_dbref(pdb_path: Path, chain: str, uniprot_id: str) -> bool:
    with pdb_path.open(encoding="utf-8") as handle:
        return any(
            line.startswith(("DBREF ", "DBREF1"))
            and len(line) > 12 and line[12].strip() == chain
            and uniprot_id in line
            for line in handle
        )


def _residue_name(pdb_path: Path, chain: str, residue_number: int) -> str | None:
    with pdb_path.open(encoding="utf-8") as handle:
        for line in handle:
            if (line.startswith("ATOM  ") and line[21].strip() == chain
                    and int(line[22:26]) == residue_number):
                return line[17:20].strip()
    return None


def validate_and_prepare(config: dict, prepare_pdbqt: bool = True) -> dict:
    refs = config["references"]
    ref_dir = resolve_path(config, "references")
    checksums = {}
    for name, expected in EXPECTED_SHA256.items():
        path = ref_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing reference {path}; run bash scripts/download_references.sh")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}: {actual} != {expected}")
        checksums[name] = actual

    bound_path = resolve_path(config, refs["bound_ligand_sdf"])
    bound = Chem.SDMolSupplier(str(bound_path), removeHs=False)[0]
    if bound is None or bound.GetNumHeavyAtoms() != 30 or bound.GetNumConformers() != 1:
        raise ValueError("Experimental PNU SDF must contain one valid 30-heavy-atom 3D structure")
    ccd_smiles = Chem.MolToSmiles(Chem.RemoveHs(bound), canonical=True, isomericSmiles=True)
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(PNU_SMILES), canonical=True, isomericSmiles=True)
    if ccd_smiles != expected:
        raise ValueError(f"Experimental YE4 connectivity does not match PNU-183792: {ccd_smiles}")
    dttp_path = ref_dir / "dTTP_8V1R_bound.sdf"
    dttp = Chem.SDMolSupplier(str(dttp_path), removeHs=False)[0]
    if dttp is None or dttp.GetNumHeavyAtoms() != 29 or dttp.GetNumConformers() != 1:
        raise ValueError("Experimental 8V1R dTTP SDF must contain one valid 29-heavy-atom 3D structure")

    pdb_path = resolve_path(config, refs["pdb_file"])
    instances = _ligand_instances(pdb_path, refs["ligand_resname"])
    requested = f"{refs['ligand_chain']}:{int(refs['ligand_resid'])}"
    if instances.get(requested) != 30:
        raise ValueError(f"Expected 30 heavy atoms for bound ligand {requested}, found {instances.get(requested)}")
    target_structures = []
    for item in config.get("target_structures", []):
        target_pdb = resolve_path(config, item["pdb_file"])
        if not _has_uniprot_dbref(target_pdb, "A", item["uniprot_id"]):
            raise ValueError(
                f"{item['pdb_id']} chain A does not declare UniProt {item['uniprot_id']}")
        wt_checks = {}
        for residue_number, expected_name in item.get("wt_residue_checks", {}).items():
            observed_name = _residue_name(target_pdb, "A", int(residue_number))
            if observed_name != expected_name:
                raise ValueError(
                    f"{item['pdb_id']} chain A residue {residue_number} is {observed_name}; "
                    f"WT workflow requires {expected_name}")
            wt_checks[str(residue_number)] = observed_name
        target_record = {
            "pdb_id": item["pdb_id"], "uniprot_id": item["uniprot_id"],
            "state": item["state"], "pdb_file": str(target_pdb), "use": item["use"],
            "docking_receptor": bool(item.get("docking_receptor", False)),
            "box": item["box"], "wt_residue_checks": wt_checks,
        }
        if item.get("docking_receptor", False):
            target_receptor = resolve_path(config, item["receptor_pdb"])
            target_stats = _write_target_receptor(
                target_pdb, target_receptor, set(item["receptor_chains"]),
                set(item["remove_resnames"]))
            target_pdbqt = resolve_path(config, item["receptor_pdbqt"])
            target_pdbqt_status = (
                _prepare_rigid_pdbqt(target_receptor, target_pdbqt)
                if prepare_pdbqt else (
                    "existing_not_rebuilt" if target_pdbqt.exists() else "not_requested"))
            target_record.update({
                "receptor": str(target_receptor), "receptor_pdbqt": str(target_pdbqt),
                "receptor_preparation": target_stats, "pdbqt_status": target_pdbqt_status,
            })
        target_structures.append(target_record)

    manifest = {
        "pdb_id": refs["pdb_id"], "ligand": refs["ligand_name"], "ccd_id": refs["ligand_resname"],
        "bound_instance": requested, "pnu_canonical_smiles": expected,
        "ligand_instances": instances, "checksums": checksums, "source_urls": SOURCE_URLS,
        "experimental_bound_conformation": str(bound_path),
        "experimental_dttp_conformation": str(dttp_path),
        "target_structures": target_structures,
        "warning": "8V1Q Open Babel receptor PDBQT is a baseline only; inspect protonation, metal, and DNA treatment before production docking."
    }
    manifest_path = ref_dir / "manifest.json"
    if prepare_pdbqt or not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    return manifest
