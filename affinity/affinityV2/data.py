from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch, Data

from utils.molecular_utils import ATOM_TYPE_DICT, NUM_ATOM_TYPES, get_atom_features, get_bond_features


STRICT_AFFINITY_RE = re.compile(
    r"\b(Ki|Kd|IC50)\s*=\s*([0-9.]+|\.[0-9]+)\s*(F?M|fM|pM|nM|uM|mM|MM)\b",
    re.IGNORECASE,
)
# Backward-compatible name for callers that imported the original parser regex.
STRICT_KD_KI_RE = STRICT_AFFINITY_RE

STANDARD_RESIDUES: Tuple[str, ...] = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
)
RESIDUE_TYPE_TO_ID: Dict[str, int] = {r: i for i, r in enumerate(STANDARD_RESIDUES)}
NUM_RESIDUE_TYPES = len(STANDARD_RESIDUES)


def suppress_rdkit_logs() -> None:
    """Silence noisy RDKit parser warnings while keeping Python exceptions intact."""
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")


@dataclass(frozen=True)
class AffinityRow:
    complex_id: str
    complex_dir: str
    affinity_pk: float
    affinity_type: str = ""


AFFINITY_TYPE_TO_ID: Dict[str, int] = {"UNK": 0, "KI": 1, "KD": 2, "IC50": 3}
AFFINITY_ID_TO_TYPE: Dict[int, str] = {v: k for k, v in AFFINITY_TYPE_TO_ID.items()}
KIKD_AFFINITY_TYPE_IDS = frozenset({AFFINITY_TYPE_TO_ID["KI"], AFFINITY_TYPE_TO_ID["KD"]})


def molar_value(value: float, unit: str) -> float:
    u = unit.replace("μ", "u").lower()
    if u == "m" and len(unit) == 1:
        return float(value)
    scale = {
        "fm": 1e-15,
        "pm": 1e-12,
        "nm": 1e-9,
        "um": 1e-6,
        "mm": 1e-3,
    }
    return float(value) * scale.get(u, 1.0)


def parse_strict_affinity_record(line_tail: str) -> Optional[Tuple[float, str]]:
    text = line_tail.split("//", 1)[0].strip()
    m = STRICT_AFFINITY_RE.search(text)
    if m is None:
        return None
    affinity_type = m.group(1).upper()
    value = float(m.group(2))
    concentration = molar_value(value, m.group(3))
    if concentration <= 0.0 or not np.isfinite(concentration):
        return None
    return float(-np.log10(concentration)), affinity_type


def parse_strict_affinity_to_pk(line_tail: str) -> Optional[float]:
    parsed = parse_strict_affinity_record(line_tail)
    if parsed is None:
        return None
    return parsed[0]


def parse_strict_kd_ki_to_pk(line_tail: str) -> Optional[float]:
    """Backward-compatible alias; strict IC50 labels are accepted as well."""
    return parse_strict_affinity_to_pk(line_tail)


def parse_index_file(
    index_path: str,
    *,
    allowed_affinity_types: Optional[Iterable[str]] = None,
) -> List[Tuple[str, float, str]]:
    allowed = {x.strip().upper() for x in allowed_affinity_types or [] if x.strip()}
    rows: List[Tuple[str, float, str]] = []
    with open(index_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            complex_id = parts[0].lower()
            parsed = parse_strict_affinity_record(" ".join(parts[3:]))
            if parsed is None:
                continue
            pk, affinity_type = parsed
            if allowed and affinity_type not in allowed:
                continue
            if not np.isfinite(pk):
                continue
            rows.append((complex_id, pk, affinity_type))
    return rows


def build_pdb_to_dir_map(pl_root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.isdir(pl_root):
        return mapping
    for root, dirs, _ in os.walk(pl_root):
        for d in dirs:
            if len(d) == 4:
                mapping.setdefault(d.lower(), os.path.join(root, d))
    return mapping


def read_casf_core_ids(casf_root: Optional[str]) -> set[str]:
    ids: set[str] = set()
    if not casf_root or not os.path.isdir(casf_root):
        return ids

    coreset_dir = os.path.join(casf_root, "coreset")
    if os.path.isdir(coreset_dir):
        for name in os.listdir(coreset_dir):
            p = os.path.join(coreset_dir, name)
            if os.path.isdir(p) and len(name) == 4:
                ids.add(name.lower())

    core_file = os.path.join(casf_root, "power_scoring", "CoreSet.dat")
    if os.path.isfile(core_file):
        with open(core_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                token = s.split()[0].lower()
                if len(token) >= 4:
                    ids.add(token[:4])
    return ids


def make_native_split(
    rows: Sequence[AffinityRow],
    casf_ids: Iterable[str],
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    internal_test_fraction: float = 0.0,
) -> Dict[str, List[int]]:
    casf = {x.lower() for x in casf_ids}
    by_code: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        by_code.setdefault(row.complex_id.lower(), []).append(i)

    test_codes = sorted(code for code in by_code if code in casf)
    pool_codes = sorted(code for code in by_code if code not in casf)
    rng = random.Random(seed)
    rng.shuffle(pool_codes)

    if internal_test_fraction > 0.0:
        n_internal = int(round(len(pool_codes) * internal_test_fraction))
        internal_test_codes = pool_codes[:n_internal]
        pool_codes = pool_codes[n_internal:]
    else:
        internal_test_codes = []

    n_val = int(round(len(pool_codes) * val_fraction))
    split_codes = {
        "train": pool_codes[n_val:],
        "val": pool_codes[:n_val],
        "test": test_codes,
        "internal_test": internal_test_codes,
    }
    return {name: [i for code in codes for i in by_code[code]] for name, codes in split_codes.items()}


def assert_disjoint_split_codes(rows: Sequence[AffinityRow], splits: Dict[str, Sequence[int]]) -> None:
    codes = {
        name: {rows[int(i)].complex_id.lower() for i in indices}
        for name, indices in splits.items()
    }
    names = list(codes)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = codes[left] & codes[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"Affinity split leakage: {left}/{right} share {len(overlap)} codes, e.g. {examples}")


def _candidate_ligand_paths(complex_dir: str, code: str) -> List[str]:
    lo = code.lower()
    up = code.upper()
    return [
        os.path.join(complex_dir, f"{lo}_ligand.sdf"),
        os.path.join(complex_dir, f"{up}_ligand.sdf"),
        os.path.join(complex_dir, f"{lo}_ligand.mol2"),
        os.path.join(complex_dir, f"{up}_ligand.mol2"),
    ]


def _candidate_pocket_paths(complex_dir: str, code: str) -> List[str]:
    lo = code.lower()
    up = code.upper()
    return [
        os.path.join(complex_dir, f"{lo}_pocket.pdb"),
        os.path.join(complex_dir, f"{up}_pocket.pdb"),
    ]


def load_ligand_native_graph(complex_dir: str, code: str) -> Optional[Data]:
    path = next((p for p in _candidate_ligand_paths(complex_dir, code) if os.path.isfile(p)), None)
    if path is None:
        return None
    mol = None
    if path.lower().endswith(".sdf"):
        for sanitize in (True, False):
            supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=sanitize)
            mol = supplier[0] if supplier and len(supplier) else None
            if mol is not None and mol.GetNumConformers() > 0:
                break
    else:
        for sanitize in (True, False):
            mol = Chem.MolFromMol2File(path, removeHs=False, sanitize=sanitize)
            if mol is not None and mol.GetNumConformers() > 0:
                break
    if mol is None or mol.GetNumConformers() < 1:
        return None
    if mol.GetNumAtoms() < 1 or mol.GetNumConformers() < 1:
        return None
    conf = mol.GetConformer()
    atom_ids: List[int] = []
    rdkit_to_heavy: Dict[int, int] = {}
    pos: List[Tuple[float, float, float]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (1,):
            continue
        idx = atom.GetIdx()
        rdkit_to_heavy[idx] = len(atom_ids)
        atom_ids.append(int(get_atom_features(atom)))
        p = conf.GetAtomPosition(idx)
        pos.append((p.x, p.y, p.z))
    if not atom_ids:
        return None
    edge_pairs: List[Tuple[int, int]] = []
    edge_types: List[int] = []
    for bond in mol.GetBonds():
        begin = rdkit_to_heavy.get(bond.GetBeginAtomIdx())
        end = rdkit_to_heavy.get(bond.GetEndAtomIdx())
        if begin is None or end is None:
            continue
        bond_type = int(get_bond_features(bond))
        edge_pairs.extend(((begin, end), (end, begin)))
        edge_types.extend((bond_type, bond_type))
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.tensor(atom_ids, dtype=torch.long),
        pos=torch.tensor(np.asarray(pos, dtype=np.float32), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_types, dtype=torch.long),
    )


def _normalize_pdb_element(raw: str) -> Optional[str]:
    elem = raw.strip()
    if not elem:
        return None
    return elem[0].upper() + elem[1:].lower()


def _infer_pdb_element(line: str) -> Optional[str]:
    elem = _normalize_pdb_element(line[76:78] if len(line) >= 78 else "")
    if elem:
        return elem
    atom_name = line[12:16].strip()
    letters = "".join(ch for ch in atom_name.lstrip("0123456789") if ch.isalpha())
    if not letters:
        return None
    first = letters[0].upper()
    if line.startswith("HETATM") and len(letters) >= 2:
        candidate = first + letters[1].lower()
        if candidate in ATOM_TYPE_DICT:
            return candidate
    return first


def _residue_type_id(resname: str, is_hetatm: bool) -> int:
    if is_hetatm:
        return RESIDUE_TYPE_TO_ID["UNK"]
    return RESIDUE_TYPE_TO_ID.get(resname.strip().upper(), RESIDUE_TYPE_TO_ID["UNK"])


def load_pocket_heavy_atom_graph(complex_dir: str, code: str) -> Optional[Data]:
    path = next((p for p in _candidate_pocket_paths(complex_dir, code) if os.path.isfile(p)), None)
    if path is None:
        return None

    atom_ids: List[int] = []
    residue_types: List[int] = []
    residue_ids: List[int] = []
    pos: List[Tuple[float, float, float]] = []
    residue_key_to_id: Dict[Tuple[str, str, str, str], int] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            altloc = line[16:17] if len(line) > 16 else " "
            if altloc not in (" ", "A"):
                continue
            elem = _infer_pdb_element(line)
            if elem is None or elem in ("H", "D"):
                continue
            try:
                xyz = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            resname = line[17:20].strip().upper() if len(line) >= 20 else "UNK"
            chain = line[21:22].strip() if len(line) >= 22 else ""
            resseq = line[22:26].strip() if len(line) >= 26 else ""
            icode = line[26:27].strip() if len(line) >= 27 else ""
            key = (chain, resseq, icode, resname)
            if key not in residue_key_to_id:
                residue_key_to_id[key] = len(residue_key_to_id)
            atom_ids.append(ATOM_TYPE_DICT.get(elem, ATOM_TYPE_DICT["OTHER"]))
            residue_types.append(_residue_type_id(resname, line.startswith("HETATM")))
            residue_ids.append(residue_key_to_id[key])
            pos.append(xyz)

    if not atom_ids:
        return None
    return Data(
        x=torch.tensor(atom_ids, dtype=torch.long),
        pos=torch.tensor(np.asarray(pos, dtype=np.float32), dtype=torch.float32),
        residue_type=torch.tensor(residue_types, dtype=torch.long),
        residue_id=torch.tensor(residue_ids, dtype=torch.long),
    )


class NativePdbbindAffinityDataset(Dataset):
    """PDBbind native-only affinity dataset."""

    def __init__(
        self,
        pl_root: str,
        index_path: str,
        *,
        allowed_affinity_types: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
        skip_invalid: bool = True,
    ) -> None:
        if not os.path.isdir(pl_root):
            raise FileNotFoundError(f"P-L root not found: {pl_root}")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"INDEX file not found: {index_path}")
        pdb_to_dir = build_pdb_to_dir_map(pl_root)
        rows: List[AffinityRow] = []
        for code, pk, affinity_type in parse_index_file(
            index_path,
            allowed_affinity_types=allowed_affinity_types,
        ):
            d = pdb_to_dir.get(code.lower())
            if d is None:
                continue
            if skip_invalid and self._load_graphs(d, code) is None:
                continue
            rows.append(AffinityRow(code.lower(), d, float(pk), affinity_type))
            if max_samples is not None and len(rows) >= max_samples:
                break
        if not rows:
            raise ValueError("No valid native PDBbind affinity samples were found.")
        self.rows = rows

    @staticmethod
    def _load_graphs(complex_dir: str, code: str) -> Optional[Tuple[Data, Data]]:
        lig = load_ligand_native_graph(complex_dir, code)
        pocket = load_pocket_heavy_atom_graph(complex_dir, code)
        if lig is None or pocket is None:
            return None
        if lig.x.numel() < 1 or pocket.x.numel() < 1:
            return None
        return lig, pocket

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, Tensor, Tensor]:
        row = self.rows[idx]
        graphs = self._load_graphs(row.complex_dir, row.complex_id)
        if graphs is None:
            raise RuntimeError(f"Failed to load native complex {row.complex_id}")
        lig, pocket = graphs
        lig.complex_id = row.complex_id
        pocket.complex_id = row.complex_id
        y = torch.tensor(float(row.affinity_pk), dtype=torch.float32)
        affinity_type_id = AFFINITY_TYPE_TO_ID.get(row.affinity_type.upper(), AFFINITY_TYPE_TO_ID["UNK"])
        return lig, pocket, y, torch.tensor(affinity_type_id, dtype=torch.long)


def collate_native_affinity_batch(
    batch: Sequence[Tuple[Data, Data, Tensor, Tensor]]
) -> Tuple[Batch, Batch, Tensor, Tensor]:
    ligs, pockets, ys, affinity_type_ids = zip(*batch)
    return (
        Batch.from_data_list(list(ligs)),
        Batch.from_data_list(list(pockets)),
        torch.stack([y.float() for y in ys], dim=0),
        torch.stack([t.long() for t in affinity_type_ids], dim=0),
    )


def build_native_datasets(
    pl_root: str,
    index_path: str,
    casf_root: Optional[str],
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    internal_test_fraction: float = 0.0,
    allowed_affinity_types: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
    skip_invalid: bool = True,
) -> Tuple[NativePdbbindAffinityDataset, Dict[str, Subset], Dict[str, List[int]]]:
    dataset = NativePdbbindAffinityDataset(
        pl_root,
        index_path,
        allowed_affinity_types=allowed_affinity_types,
        max_samples=max_samples,
        skip_invalid=skip_invalid,
    )
    splits = make_native_split(
        dataset.rows,
        read_casf_core_ids(casf_root),
        seed=seed,
        val_fraction=val_fraction,
        internal_test_fraction=internal_test_fraction,
    )
    assert_disjoint_split_codes(dataset.rows, splits)
    subsets = {name: Subset(dataset, idxs) for name, idxs in splits.items() if idxs}
    return dataset, subsets, splits


def target_mean_std(dataset: NativePdbbindAffinityDataset, indices: Sequence[int]) -> Tuple[float, float]:
    values = np.asarray([dataset.rows[i].affinity_pk for i in indices], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


def clamp_atom_types(x: Tensor) -> Tensor:
    return torch.where(
        (x >= 0) & (x < NUM_ATOM_TYPES),
        x,
        torch.full_like(x, ATOM_TYPE_DICT["OTHER"]),
    )
