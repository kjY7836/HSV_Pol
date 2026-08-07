#!/usr/bin/env python3
"""Run the bundled 3DMPG affinity model on docked ligand poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from rdkit import Chem, rdBase
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinityV2.data import (  # noqa: E402
    NUM_RESIDUE_TYPES,
    RESIDUE_TYPE_TO_ID,
    STANDARD_RESIDUES,
)
from affinityV2.model import AffinityScorerModel  # noqa: E402
from utils.molecular_utils import (  # noqa: E402
    ATOM_TYPE_DICT,
    get_atom_features,
    get_bond_features,
)


DEFAULT_CHECKPOINT = (
    ROOT / "weights" / "affinity_v2_r2020_semantic_epoch3encoder_e90_1_rerun.pt"
)
MODEL_KWARGS = {
    "hidden_dim",
    "num_layers",
    "heads",
    "ligand_r_cut",
    "pocket_r_cut",
    "interface_cutoff",
    "max_neighbors_per_lig",
    "max_pairs_per_graph",
    "pair_dim",
    "dropout",
}
SMINA_SCORE_PROPERTIES = ("minimizedAffinity", "smina_affinity", "affinity")
PREDICTION_FIELDS = [
    "ligand_record_index",
    "pose_id",
    "ligand_id",
    "smina_score",
    "ligand_heavy_atoms",
    "pocket_heavy_atoms",
    "pocket_residues",
    "nearest_receptor_distance",
    "predicted_paffinity",
    "approx_concentration_nM",
]


@dataclass(frozen=True)
class ReceptorAtom:
    element: str
    position: tuple[float, float, float]
    residue_key: tuple[str, str, str, str]
    residue_type: int


def _pdb_element(line: str) -> str | None:
    raw = line[76:78].strip() if len(line) >= 78 else ""
    if raw:
        return raw[0].upper() + raw[1:].lower()
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


def read_receptor_atoms(path: Path) -> list[ReceptorAtom]:
    atoms: list[ReceptorAtom] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            altloc = line[16:17] if len(line) > 16 else " "
            if altloc not in (" ", "A"):
                continue
            element = _pdb_element(line)
            if element is None or element in {"H", "D"}:
                continue
            try:
                position = (
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
            is_hetatm = line.startswith("HETATM")
            residue_type = (
                RESIDUE_TYPE_TO_ID["UNK"]
                if is_hetatm
                else RESIDUE_TYPE_TO_ID.get(resname, RESIDUE_TYPE_TO_ID["UNK"])
            )
            atoms.append(
                ReceptorAtom(
                    element=element,
                    position=position,
                    residue_key=(chain, resseq, icode, resname),
                    residue_type=residue_type,
                )
            )
    if not atoms:
        raise ValueError(f"No receptor heavy atoms could be parsed from {path}")
    return atoms


def ligand_graph(mol: Chem.Mol) -> Data:
    if mol.GetNumConformers() != 1:
        raise ValueError("Ligand record must contain exactly one 3D conformer")
    conf = mol.GetConformer()
    atom_types: list[int] = []
    positions: list[tuple[float, float, float]] = []
    rdkit_to_heavy: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        rdkit_to_heavy[atom.GetIdx()] = len(atom_types)
        atom_types.append(int(get_atom_features(atom)))
        p = conf.GetAtomPosition(atom.GetIdx())
        positions.append((p.x, p.y, p.z))
    if not atom_types:
        raise ValueError("Ligand contains no heavy atoms")

    edge_pairs: list[tuple[int, int]] = []
    edge_types: list[int] = []
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
        x=torch.tensor(atom_types, dtype=torch.long),
        pos=torch.tensor(np.asarray(positions, dtype=np.float32), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_types, dtype=torch.long),
    )


def pocket_graph(
    receptor_atoms: Sequence[ReceptorAtom],
    ligand: Data,
    radius: float,
) -> tuple[Data, int, float]:
    receptor_xyz = np.asarray([atom.position for atom in receptor_atoms], dtype=np.float32)
    ligand_xyz = ligand.pos.detach().cpu().numpy()
    distance_sq = np.sum(
        (receptor_xyz[:, None, :] - ligand_xyz[None, :, :]) ** 2,
        axis=2,
    )
    nearest = float(np.sqrt(distance_sq.min()))
    near_atom = np.any(distance_sq <= float(radius) ** 2, axis=1)
    selected_residues = {
        atom.residue_key for atom, selected in zip(receptor_atoms, near_atom) if selected
    }
    if not selected_residues:
        raise ValueError(
            f"No receptor residue lies within {radius:.1f} A of the ligand "
            f"(nearest heavy-atom distance: {nearest:.2f} A)"
        )

    selected = [atom for atom in receptor_atoms if atom.residue_key in selected_residues]
    residue_ids: dict[tuple[str, str, str, str], int] = OrderedDict()
    for atom in selected:
        residue_ids.setdefault(atom.residue_key, len(residue_ids))
    graph = Data(
        x=torch.tensor(
            [ATOM_TYPE_DICT.get(atom.element, ATOM_TYPE_DICT["OTHER"]) for atom in selected],
            dtype=torch.long,
        ),
        pos=torch.tensor(
            np.asarray([atom.position for atom in selected], dtype=np.float32),
            dtype=torch.float32,
        ),
        residue_type=torch.tensor(
            [atom.residue_type for atom in selected], dtype=torch.long
        ).clamp(min=0, max=NUM_RESIDUE_TYPES - 1),
        residue_id=torch.tensor(
            [residue_ids[atom.residue_key] for atom in selected], dtype=torch.long
        ),
    )
    return graph, len(residue_ids), nearest


def _model_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    saved_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    kwargs = {key: saved_args[key] for key in MODEL_KWARGS if key in saved_args}
    if "y_mean" in payload:
        kwargs["y_mean"] = float(payload["y_mean"])
    if "y_std" in payload:
        kwargs["y_std"] = float(payload["y_std"])
    return kwargs


def load_model(checkpoint: Path, device: torch.device) -> tuple[AffinityScorerModel, dict]:
    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        state = dict(payload["model_state_dict"])
        metadata = {
            "checkpoint_kind": payload.get("checkpoint_kind", ""),
            "epoch": payload.get("epoch"),
            "best_val_rmse": payload.get("best_val_rmse"),
            "y_mean": payload.get("y_mean"),
            "y_std": payload.get("y_std"),
            "model_kwargs": _model_kwargs(payload),
        }
        model = AffinityScorerModel(**_model_kwargs(payload))
    elif isinstance(payload, dict):
        state = dict(payload)
        metadata = {"checkpoint_kind": "raw_state_dict"}
        model = AffinityScorerModel()
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

    state = {key: value for key, value in state.items() if not key.startswith("pose_head.")}
    info = model.load_state_dict(state, strict=False)
    missing = [key for key in info.missing_keys if not key.startswith("contrastive_head.")]
    if missing or info.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing}, "
            f"unexpected={list(info.unexpected_keys)}"
        )
    model.to(device)
    model.eval()
    return model, metadata


def iter_ligands(path: Path) -> Iterable[tuple[int, Chem.Mol]]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    with rdBase.BlockLogs():
        for index, mol in enumerate(supplier, start=1):
            if mol is None:
                raise ValueError(f"Invalid SDF record {index} in {path}")
            if mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D():
                raise ValueError(f"SDF record {index} does not contain one 3D conformer")
            yield index, mol


def _property(mol: Chem.Mol, names: Iterable[str]) -> str:
    for name in names:
        if mol.HasProp(name):
            return mol.GetProp(name)
    return ""


def _smina_score(mol: Chem.Mol) -> float | None:
    raw = _property(mol, SMINA_SCORE_PROPERTIES)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested, but CUDA is unavailable")
    return torch.device(requested)


def predict(
    receptor_pdb: Path,
    ligand_sdf: Path,
    checkpoint: Path,
    *,
    device_name: str = "auto",
    pocket_radius: float = 10.0,
    batch_size: int = 1,
    threads: int = 0,
    progress_every: int = 0,
    expected_pose_count: int | None = None,
    completed_predictions: Sequence[dict[str, Any]] = (),
    row_sink: Callable[[Sequence[dict[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if threads > 0:
        torch.set_num_threads(threads)
    device = resolve_device(device_name)
    receptor_atoms = read_receptor_atoms(receptor_pdb)
    started = time.time()
    model, checkpoint_metadata = load_model(checkpoint, device)
    rows: list[dict[str, Any]] = []
    resumed_pose_count = len(completed_predictions)
    last_reported = resumed_pose_count

    def infer_chunk(chunk: Sequence[tuple[Data, Data, dict[str, Any]]]) -> None:
        nonlocal last_reported
        ligand_batch = Batch.from_data_list([item[0] for item in chunk]).to(device)
        pocket_batch = Batch.from_data_list([item[1] for item in chunk]).to(device)
        output = model(ligand_batch, pocket_batch)
        predictions = output["affinity_pred"].detach().float().cpu().tolist()
        chunk_rows: list[dict[str, Any]] = []
        for (_, _, metadata), predicted in zip(chunk, predictions):
            value = float(predicted)
            if not math.isfinite(value):
                raise ValueError(
                    f"Model returned non-finite pAffinity for {metadata['pose_id']}: {value}"
                )
            chunk_rows.append(
                {
                    **metadata,
                    "predicted_paffinity": value,
                    "approx_concentration_nM": 1.0e9 * math.pow(10.0, -value),
                }
            )
        rows.extend(chunk_rows)
        if row_sink is not None:
            row_sink(chunk_rows)
        processed = resumed_pose_count + len(rows)
        if progress_every > 0 and processed - last_reported >= progress_every:
            elapsed = max(time.time() - started, 1.0e-9)
            rate = len(rows) / elapsed
            progress = f"[affinity] processed={processed:,} rate={rate:.2f} poses/s"
            if expected_pose_count:
                remaining = max(expected_pose_count - processed, 0)
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                progress += (
                    f" total={expected_pose_count:,}"
                    f" progress={100.0 * processed / expected_pose_count:.2f}%"
                    f" eta={eta_seconds / 3600.0:.2f}h"
                )
            print(progress, flush=True)
            last_reported = processed

    prepared: list[tuple[Data, Data, dict[str, Any]]] = []
    record_count = 0
    with torch.inference_mode():
        for record_index, mol in iter_ligands(ligand_sdf):
            record_count += 1
            if record_index <= resumed_pose_count:
                expected_pose_id = str(
                    completed_predictions[record_index - 1].get("pose_id", "")
                )
                actual_pose_id = _property(mol, ("pose_id",)) or _property(
                    mol, ("_Name",)
                )
                if expected_pose_id and actual_pose_id != expected_pose_id:
                    raise ValueError(
                        f"Resume pose_id mismatch at record {record_index}: "
                        f"{expected_pose_id!r} != {actual_pose_id!r}"
                    )
                continue
            lig = ligand_graph(mol)
            pocket, residue_count, nearest = pocket_graph(
                receptor_atoms, lig, pocket_radius
            )
            pose_id = _property(mol, ("pose_id",)) or _property(mol, ("_Name",))
            ligand_id = _property(
                mol,
                ("compound_id", "state_id", "parent_structure_key", "_Name"),
            ) or f"record_{record_index}"
            prepared.append(
                (
                    lig,
                    pocket,
                    {
                        "ligand_record_index": record_index,
                        "pose_id": pose_id or f"record_{record_index}",
                        "ligand_id": ligand_id,
                        "smina_score": _smina_score(mol),
                        "ligand_heavy_atoms": int(lig.x.numel()),
                        "pocket_heavy_atoms": int(pocket.x.numel()),
                        "pocket_residues": residue_count,
                        "nearest_receptor_distance": nearest,
                    },
                )
            )
            if len(prepared) >= batch_size:
                infer_chunk(prepared)
                prepared.clear()
        if prepared:
            infer_chunk(prepared)
            prepared.clear()
    if record_count == 0:
        raise ValueError(f"No ligand records found in {ligand_sdf}")
    if resumed_pose_count > record_count:
        raise ValueError(
            f"Resume checkpoint has {resumed_pose_count} rows but the ligand SDF "
            f"contains only {record_count} records"
        )

    summary = {
        "receptor_pdb": str(receptor_pdb.resolve()),
        "ligand_sdf": str(ligand_sdf.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "device": str(device),
        "pocket_radius_A": float(pocket_radius),
        "pose_count": resumed_pose_count + len(rows),
        "resumed_pose_count": resumed_pose_count,
        "expected_pose_count": expected_pose_count,
        "elapsed_seconds": time.time() - started,
        "checkpoint_metadata": checkpoint_metadata,
        "residue_vocabulary": list(STANDARD_RESIDUES),
    }
    return rows, summary


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    partial = Path(str(path) + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def read_prediction_checkpoint(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if fields != PREDICTION_FIELDS:
            raise ValueError(
                f"Invalid prediction checkpoint columns in {path}: {fields}"
            )
        rows = list(reader)
    for expected_index, row in enumerate(rows, start=1):
        try:
            record_index = int(row["ligand_record_index"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid ligand_record_index in prediction checkpoint row {expected_index}"
            ) from exc
        if record_index != expected_index:
            raise ValueError(
                f"Prediction checkpoint is not a contiguous 1-based prefix: "
                f"row {expected_index} contains index {record_index}"
            )
        if not row.get("pose_id", ""):
            raise ValueError(f"Prediction checkpoint row {expected_index} lacks pose_id")
    return rows


def fill_prediction_table(
    template: Path,
    destination: Path,
    predictions: Sequence[dict[str, Any]],
) -> None:
    if template.resolve() == destination.resolve():
        raise ValueError("Refusing to overwrite the input affinity_predictions.csv")
    with template.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "ligand_record_index" not in fields or "predicted_paffinity" not in fields:
        raise ValueError(
            "Template must contain ligand_record_index and predicted_paffinity columns"
        )
    by_index = {int(row["ligand_record_index"]): row for row in predictions}
    if len(by_index) != len(predictions):
        raise ValueError("Predictions contain duplicate ligand_record_index values")
    template_indices = [int(row["ligand_record_index"]) for row in rows]
    if set(template_indices) != set(by_index):
        missing = sorted(set(template_indices) - set(by_index))
        extra = sorted(set(by_index) - set(template_indices))
        raise ValueError(
            f"Prediction/template record mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for row in rows:
        prediction = by_index[int(row["ligand_record_index"])]
        template_pose = row.get("pose_id", "")
        predicted_pose = str(prediction.get("pose_id", ""))
        if template_pose and predicted_pose and template_pose != predicted_pose:
            raise ValueError(
                f"pose_id mismatch at ligand_record_index={row['ligand_record_index']}: "
                f"{template_pose!r} != {predicted_pose!r}"
            )
        row["predicted_paffinity"] = f"{float(prediction['predicted_paffinity']):.8f}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, destination)


def install_prediction_table(
    template: Path,
    predictions: Sequence[dict[str, Any]],
) -> Path:
    """Atomically install local predictions while preserving the original table."""
    backup = template.with_name(
        f"{template.stem}.before_local_affinity{template.suffix}"
    )
    if not backup.exists():
        backup_partial = Path(str(backup) + ".partial")
        shutil.copy2(template, backup_partial)
        os.replace(backup_partial, backup)

    staged = template.with_name(f"{template.stem}.local_filled{template.suffix}")
    fill_prediction_table(template, staged, predictions)
    os.replace(staged, template)
    return backup


def count_prediction_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _package_inputs(package: Path) -> tuple[Path, Path, Path]:
    receptors = sorted((package / "receptor").glob("*.pdb"))
    if len(receptors) != 1:
        raise ValueError(f"Expected exactly one receptor/*.pdb in {package}, found {len(receptors)}")
    ligand = package / "ligands" / "docked_poses.sdf"
    template = package / "affinity_predictions.csv"
    if not ligand.is_file() or not template.is_file():
        raise FileNotFoundError(
            f"Affinity package lacks {ligand.relative_to(package)} or "
            f"{template.relative_to(package)}"
        )
    return receptors[0], ligand, template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict pAffinity for Smina-generated SDF poses with the bundled 3DMPG model."
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "--affinity-input",
        type=Path,
        help="A full_library_screening 10_affinity_input directory.",
    )
    inputs.add_argument("--receptor-pdb", type=Path)
    inputs.add_argument("--ligand-sdf", type=Path)
    inputs.add_argument("--template-csv", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--filled-template-csv",
        type=Path,
        help="Optional copy of the handoff table with predicted_paffinity filled.",
    )
    parser.add_argument(
        "--install-filled-template",
        action="store_true",
        help=(
            "Atomically replace the package affinity_predictions.csv with locally "
            "predicted values, preserving a before_local_affinity backup."
        ),
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--threads", type=int, default=0, help="Torch CPU threads; 0 keeps the environment default.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print throughput and ETA after this many additional poses; 0 disables progress.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the contiguous output-csv.partial prediction checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receptor = args.receptor_pdb
    ligand = args.ligand_sdf
    template = args.template_csv
    if args.affinity_input:
        package_receptor, package_ligand, package_template = _package_inputs(args.affinity_input)
        receptor = receptor or package_receptor
        ligand = ligand or package_ligand
        template = template or package_template
    if receptor is None or ligand is None:
        raise SystemExit(
            "Provide --affinity-input, or provide both --receptor-pdb and --ligand-sdf."
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    if args.install_filled_template and args.filled_template_csv:
        raise SystemExit(
            "Use either --install-filled-template or --filled-template-csv, not both."
        )
    if args.install_filled_template and template is None:
        raise SystemExit("--install-filled-template requires --affinity-input or --template-csv")
    for path, label in (
        (receptor, "receptor PDB"),
        (ligand, "ligand SDF"),
        (args.checkpoint, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    expected_pose_count = count_prediction_rows(template) if template else None
    checkpoint_path = Path(str(args.output_csv) + ".partial")
    completed_predictions: list[dict[str, Any]] = (
        list(read_prediction_checkpoint(checkpoint_path)) if args.resume else []
    )
    checkpoint_handle = None
    checkpoint_writer = None
    if args.resume:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            with checkpoint_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
        checkpoint_handle = checkpoint_path.open(
            "a", encoding="utf-8", newline=""
        )
        checkpoint_writer = csv.DictWriter(
            checkpoint_handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n"
        )

    def checkpoint_rows(rows: Sequence[dict[str, Any]]) -> None:
        assert checkpoint_writer is not None and checkpoint_handle is not None
        checkpoint_writer.writerows(rows)
        checkpoint_handle.flush()

    try:
        new_predictions, summary = predict(
            receptor,
            ligand,
            args.checkpoint,
            device_name=args.device,
            pocket_radius=args.pocket_radius,
            batch_size=args.batch_size,
            threads=args.threads,
            progress_every=args.progress_every,
            expected_pose_count=expected_pose_count,
            completed_predictions=completed_predictions,
            row_sink=checkpoint_rows if args.resume else None,
        )
    finally:
        if checkpoint_handle is not None:
            checkpoint_handle.close()
    predictions = [*completed_predictions, *new_predictions]
    if expected_pose_count is not None and len(predictions) != expected_pose_count:
        raise ValueError(
            f"Predicted {len(predictions)} poses but the template contains "
            f"{expected_pose_count} rows"
        )
    if args.resume:
        os.replace(checkpoint_path, args.output_csv)
    else:
        write_rows(args.output_csv, predictions)
    if args.filled_template_csv:
        if template is None:
            raise ValueError("--filled-template-csv requires a template or --affinity-input")
        fill_prediction_table(template, args.filled_template_csv, predictions)
        summary["filled_template_csv"] = str(args.filled_template_csv.resolve())
    if args.install_filled_template:
        assert template is not None
        backup = install_prediction_table(template, predictions)
        summary["installed_template_csv"] = str(template.resolve())
        summary["original_template_backup"] = str(backup.resolve())
    summary["predictions_csv"] = str(args.output_csv.resolve())
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
