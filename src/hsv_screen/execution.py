from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

from .io_utils import atomic_json


AFFINITY_PROPERTIES = ("minimizedAffinity", "smina_affinity", "affinity")
JOB_SIGNATURE_VERSION = 1


def smina_binary() -> Path:
    adjacent = Path(sys.executable).resolve().parent / "smina"
    candidate = adjacent if adjacent.is_file() else Path(shutil.which("smina") or "")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"smina was not found next to the current Python ({adjacent}) or on PATH")
    return candidate.resolve()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _job_signature(binary: Path, row: dict[str, str]) -> str:
    excluded = {"output", "log"}
    payload = {
        "version": JOB_SIGNATURE_VERSION,
        "job": {key: str(value) for key, value in sorted(row.items()) if key not in excluded},
        "binary": _file_identity(binary),
        "ligands": _file_identity(Path(row["ligands"])),
        "receptor": _file_identity(Path(row["receptor"])),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _molecule_id(mol: Chem.Mol) -> str:
    if mol.HasProp("state_id"):
        return mol.GetProp("state_id").strip()
    if mol.HasProp("_Name"):
        return mol.GetProp("_Name").strip()
    return ""


def _supplier(path: Path) -> Chem.SDMolSupplier:
    return Chem.SDMolSupplier(
        str(path), sanitize=False, removeHs=False, strictParsing=True)


def validate_smina_output(row: dict[str, str], output: Path | None = None) -> dict:
    """Validate job completeness without rejecting Smina bond-order perception bugs.

    Sanitization deliberately remains disabled here.  Collection performs the strict
    chemistry check (and narrowly scoped topology normalization); resume validation
    only proves that the output is a complete, traceable result for this exact chunk.
    """
    ligand_path = Path(row["ligands"])
    output_path = output or Path(row["output"])
    if not ligand_path.is_file():
        raise FileNotFoundError(f"Missing Smina ligand chunk: {ligand_path}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty Smina output: {output_path}")

    expected: set[str] = set()
    with rdBase.BlockLogs():
        for record_index, mol in enumerate(_supplier(ligand_path), start=1):
            if mol is None:
                raise ValueError(
                    f"Invalid Smina input record {record_index} in {ligand_path}")
            state_id = _molecule_id(mol)
            if not state_id:
                raise ValueError(
                    f"Untraceable Smina input record {record_index} in {ligand_path}")
            if state_id in expected:
                raise ValueError(f"Duplicate Smina input state {state_id} in {ligand_path}")
            expected.add(state_id)

    observed: Counter[str] = Counter()
    with rdBase.BlockLogs():
        for record_index, mol in enumerate(_supplier(output_path), start=1):
            if mol is None:
                raise ValueError(
                    f"Invalid raw Smina output record {record_index} in {output_path}")
            state_id = _molecule_id(mol)
            if not state_id or state_id not in expected:
                raise ValueError(
                    f"Untraceable or unexpected Smina output state {state_id!r} "
                    f"at record {record_index} in {output_path}")
            affinity_name = next(
                (name for name in AFFINITY_PROPERTIES if mol.HasProp(name)), None)
            if affinity_name is None:
                raise ValueError(
                    f"Smina output state {state_id} lacks an affinity property in {output_path}")
            try:
                float(mol.GetProp(affinity_name))
            except ValueError as exc:
                raise ValueError(
                    f"Smina output state {state_id} has invalid {affinity_name} in "
                    f"{output_path}") from exc
            observed[state_id] += 1

    missing = expected - set(observed)
    if missing:
        raise ValueError(
            f"Smina output missed {len(missing)} input states in {output_path}; "
            f"first={sorted(missing)[0]}")
    num_modes = int(row["num_modes"])
    excessive = sorted(state_id for state_id, count in observed.items() if count > num_modes)
    if excessive:
        first = excessive[0]
        raise ValueError(
            f"Smina output has {observed[first]} poses for {first}, above num_modes={num_modes}")
    return {
        "input_state_count": len(expected),
        "output_pose_count": sum(observed.values()),
        "observed_state_count": len(observed),
    }


def _completion_marker(output: Path) -> Path:
    return Path(str(output) + ".done.json")


def _read_marker(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"invalid": True}
    return value if isinstance(value, dict) else {"invalid": True}


def _write_marker(
    marker: Path, row: dict[str, str], signature: str, validation: dict, recovered: bool,
) -> None:
    atomic_json(marker, {
        "version": JOB_SIGNATURE_VERSION,
        "job_signature": signature,
        "protocol_id": row["protocol_id"],
        "chunk": row["chunk"],
        "output": row["output"],
        "recovered_existing_output": recovered,
        "validation": validation,
    })


def run_smina_job(binary: Path, row: dict[str, str]) -> dict[str, str]:
    output = Path(row["output"])
    log = Path(row["log"])
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    marker = _completion_marker(output)
    signature = _job_signature(binary, row)
    existing_marker = _read_marker(marker)

    # A matching marker plus a complete raw SDF is the normal resume path.
    if existing_marker and existing_marker.get("job_signature") == signature:
        try:
            validation = validate_smina_output(row, output)
        except (OSError, ValueError):
            pass
        else:
            return {"chunk": row["chunk"], "status": "skipped"}

    # Promote legacy outputs created before per-job markers existed.  A present but
    # invalid/mismatched marker is never promoted because the job inputs may differ.
    if existing_marker is None and output.is_file():
        try:
            validation = validate_smina_output(row, output)
        except (OSError, ValueError):
            pass
        else:
            _write_marker(marker, row, signature, validation, recovered=True)
            return {"chunk": row["chunk"], "status": "recovered"}

    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temporary_output = output.with_name(
        f"{output.stem}.{token}.partial{output.suffix}")
    temporary_log = log.with_name(f"{log.name}.{token}.partial")
    command = [
        str(binary), "--receptor", row["receptor"], "--ligand", row["ligands"],
        "--center_x", row["center_x"], "--center_y", row["center_y"],
        "--center_z", row["center_z"], "--size_x", row["size_x"],
        "--size_y", row["size_y"], "--size_z", row["size_z"],
        "--cpu", row["cpu"], "--exhaustiveness", row["exhaustiveness"],
        "--num_modes", row["num_modes"], "--seed", row["seed"],
        "--out", str(temporary_output), "--log", str(temporary_log),
    ]
    process = subprocess.run(
        command, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if process.returncode != 0:
        temporary_output.unlink(missing_ok=True)
        temporary_log.unlink(missing_ok=True)
        raise RuntimeError(
            f"Smina failed for protocol={row['protocol_id']} chunk={row['chunk']} "
            f"without replacing existing output (log: {row['log']}):\n"
            f"{process.stderr[-2000:]}")

    try:
        validation = validate_smina_output(row, temporary_output)
    except Exception as exc:
        raise RuntimeError(
            f"Smina produced an incomplete output for protocol={row['protocol_id']} "
            f"chunk={row['chunk']}; existing output was preserved and the invalid "
            f"temporary file is {temporary_output}: {exc}") from exc

    if temporary_log.is_file():
        os.replace(temporary_log, log)
    os.replace(temporary_output, output)
    _write_marker(marker, row, signature, validation, recovered=False)
    return {"chunk": row["chunk"], "status": "completed"}
