from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence


XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def prepare_stage_dir(stage: Path, force: bool) -> None:
    """Create an empty stage or explicitly replace it when --force is used."""
    if stage.exists() and any(stage.iterdir()):
        if not force:
            raise FileExistsError(f"Refusing to overwrite partial stage without --force: {stage}")
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def atomic_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    opener = gzip.open if path.suffix == ".gz" else open
    kwargs = {"compresslevel": 3} if path.suffix == ".gz" else {}
    with opener(partial, "wt", encoding="utf-8", newline="", **kwargs) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def stable_hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "big")


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    values = []
    with archive.open("xl/sharedStrings.xml") as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag == f"{{{XLSX_NS}}}si":
                values.append("".join(node.text or "" for node in elem.iter(f"{{{XLSX_NS}}}t")))
                elem.clear()
    return values


def _xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    kind = cell.attrib.get("t", "")
    value = cell.find(f"{{{XLSX_NS}}}v")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{XLSX_NS}}}t"))
    if value is None or value.text is None:
        return ""
    return shared_strings[int(value.text)] if kind == "s" else value.text


def iter_xlsx_dicts(path: Path) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive)
        sheets = sorted(name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
        if not sheets:
            raise ValueError(f"No worksheet found in {path}")
        headers: dict[str, str] = {}
        with archive.open(sheets[0]) as handle:
            for _, row in ET.iterparse(handle, events=("end",)):
                if row.tag != f"{{{XLSX_NS}}}row":
                    continue
                cells = {}
                for cell in row.findall(f"{{{XLSX_NS}}}c"):
                    match = re.match(r"([A-Z]+)", cell.attrib.get("r", ""))
                    if match:
                        cells[match.group(1)] = _xlsx_cell_value(cell, shared)
                if not headers:
                    headers = {col: value for col, value in cells.items()}
                else:
                    record = {headers[col]: value for col, value in cells.items() if col in headers}
                    record["__source_row__"] = row.attrib.get("r", "")
                    yield record
                row.clear()


def iter_raw_dicts(path: Path) -> Iterator[dict[str, str]]:
    with path.open("rb") as handle:
        xlsx = handle.read(2) == b"PK"
    if xlsx:
        yield from iter_xlsx_dicts(path)
        return
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header in {path}")
        for row_number, row in enumerate(reader, start=2):
            row["__source_row__"] = str(row_number)
            yield row


def normalized_field(row: dict, *aliases: str) -> str:
    normalized = {str(key).strip().lower().replace(" ", ""): str(value or "").strip() for key, value in row.items()}
    for alias in aliases:
        value = normalized.get(alias.lower().replace(" ", ""))
        if value is not None:
            return value
    return ""


def raw_record(path: Path, data_root: Path, row: dict, source_pool: str) -> dict[str, str]:
    return {
        "source_file": str(path.relative_to(data_root)),
        "source_row": normalized_field(row, "__source_row__"),
        "source_pool": source_pool,
        "compound_id": normalized_field(row, "TSID", "HITId", "HitId", "ID"),
        "cas": normalized_field(row, "CAS", "cas"),
        "name": normalized_field(row, "name", "Name"),
        "original_mol_weight": normalized_field(row, "Mol Weight", "MolWeight", "MW"),
        "original_formula": normalized_field(row, "Formula"),
        "original_smiles": normalized_field(row, "SMILES", "Smiles", "Smile", "smiles"),
    }
