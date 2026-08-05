from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_DIR / "references"

FILES = {
    "7LUF.pdb": (
        "https://files.rcsb.org/download/7LUF.pdb",
        "a62ce9ea817ccc3c7519df99dffcddfd86b73589899eaba2198d3f07f3928aed",
    ),
    "7LUF.cif": (
        "https://files.rcsb.org/download/7LUF.cif",
        "c82d09d8fb7ef87d93feab8caf1c90b51d76a723a04fd324e58480a9eb1ca8f3",
    ),
    "YE4.cif": (
        "https://files.rcsb.org/ligands/download/YE4.cif",
        "dadc3d881dcf4c9d6e862e88ef07debcd74ce080755863300de1ff4f0ebf957a",
    ),
    "YE4_ideal.sdf": (
        "https://files.rcsb.org/ligands/download/YE4_ideal.sdf",
        "e1b55fa888b903e6dea4940b358b8e8138d8b594746cc15f9345b93e895307b4",
    ),
    "PNU-183792_7LUF_bound.sdf": (
        "https://models.rcsb.org/v1/7luf/ligand?auth_asym_id=C&auth_seq_id=1201&encoding=sdf",
        "2db65afe0ec219e4a7ac1bb2d35a7d0d86071ea8b272903d6ff16e65ba2d7c34",
    ),
    "8V1Q.pdb": (
        "https://files.rcsb.org/download/8V1Q.pdb",
        "f80e211f7d45fd30cb31e10d03ca6c76ed1a110e38500640d37a411ab995a7fc",
    ),
    "8V1Q.cif": (
        "https://files.rcsb.org/download/8V1Q.cif",
        "338c306634aba6df76838bc47b2d32f7a05f2bddaa254100f52dc225dbd3e467",
    ),
    "8V1R.pdb": (
        "https://files.rcsb.org/download/8V1R.pdb",
        "5fb5d0881ae2a4446b568f107ad7b604c2ca760dec972950093bf3705017ced7",
    ),
    "8V1R.cif": (
        "https://files.rcsb.org/download/8V1R.cif",
        "4436e12a0fd434346f0142d2f462971ae02b09d7d93c5628b615fac426350313",
    ),
    "dTTP_8V1R_bound.sdf": (
        "https://models.rcsb.org/v1/8v1r/ligand?auth_asym_id=A&auth_seq_id=1306&encoding=sdf",
        "8a83b622ed41beb22911423a5c643b4b985f1b9548501fac6b3ad885bb59e102",
    ),
}


def download(url: str, destination: Path) -> None:
    partial = Path(str(destination) + ".partial")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                partial.write_bytes(response.read())
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt)


def strip_dynamic_modelserver_properties(path: Path) -> None:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if (line.startswith(">") and "<" in line) or line == "$$$$":
            break
        lines.append(line)
    path.write_text("\n".join(lines) + "\n$$$$\n", encoding="utf-8")


def main() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in FILES.items():
        path = REFERENCE_DIR / filename
        print(f"Downloading {filename}", flush=True)
        download(url, path)
        if filename == "PNU-183792_7LUF_bound.sdf":
            strip_dynamic_modelserver_properties(path)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected}, found {observed}")
    print("All reference checksums passed.")


if __name__ == "__main__":
    main()
