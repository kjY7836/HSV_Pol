from __future__ import annotations

import csv
import gzip
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

from .chemistry import init_score_worker, score_parent, score_reference_self
from .config import resolve_path
from .io_utils import atomic_json, iter_csv, prepare_stage_dir
from .reference import PNU_SMILES


SCORED_FIELDS = [
    "compound_id", "standardized_smiles", "charged_parent_smiles", "structure_key",
    "connectivity_key", "source_pool", "source_file", "source_row", "component_class",
    "has_permanent_charge", "active_connectivity_overlap", "hard_filter_pass",
    "hard_filter_reasons", "mw", "clogp", "tpsa", "hbd", "hba",
    "rotatable_bonds", "ring_count", "heavy_atoms", "formal_charge", "fraction_csp3",
    "qed", "pains_alert", "brenk_alert", "soft_alert_count", "murcko_scaffold",
    "pnu_ecfp_similarity", "pnu_fcfp_similarity", "pharm2d_similarity",
    "pnu_consensus", "quality_score", "library_evidence_score",
]


def _score_file(input_path: Path, output_path: Path, config: dict, counters: Counter, started: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output_path) + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8", newline="", compresslevel=3) as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        worker_count = int(config["workers"])

        def consume(results) -> None:
            for result in results:
                counters["input_parents"] += 1
                if result is None:
                    counters["invalid"] += 1
                    continue
                writer.writerow(result)
                counters["scored"] += 1
                counters["hard_filter_pass"] += int(result["hard_filter_pass"])
                for reason in str(result["hard_filter_reasons"]).split("|"):
                    if reason:
                        counters[f"reject:{reason}"] += 1
                if counters["input_parents"] % 100000 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(f"[2d] {counters['input_parents']:,} parents; {counters['input_parents']/elapsed:,.0f}/s", flush=True)
        init_args = (
            PNU_SMILES, config["fingerprint"], config["filters"],
            config["integrated_scoring"]["predocking"],
        )
        if worker_count <= 1:
            init_score_worker(*init_args)
            consume(map(score_parent, iter_csv(input_path)))
        else:
            with mp.Pool(
                    worker_count, initializer=init_score_worker,
                    initargs=init_args) as pool:
                consume(pool.imap(score_parent, iter_csv(input_path), chunksize=128))
    os.replace(partial, output_path)


def run(config: dict, force: bool = False) -> dict:
    root = resolve_path(config, config["output_dir"])
    source = root / "01_standardized"
    stage = root / "02_2d_scored"
    done = stage / "summary.json"
    if done.exists() and not force:
        return json.loads(done.read_text(encoding="utf-8"))
    prepare_stage_dir(stage, force)
    inputs = [source / "activity_parents.csv.gz", *sorted(source.glob("unlabeled_parents_*.csv.gz"))]
    if not all(path.exists() for path in inputs):
        raise FileNotFoundError("Standardized parent shards are incomplete")
    counters: Counter = Counter()
    started = time.time()
    for input_path in inputs:
        output_path = stage / input_path.name.replace("parents", "scored")
        _score_file(input_path, output_path, config, counters, started)
    self_score = score_reference_self(
        PNU_SMILES, config["fingerprint"], config["filters"],
        config["integrated_scoring"]["predocking"])
    for field in ("pnu_ecfp_similarity", "pnu_fcfp_similarity", "pharm2d_similarity", "pnu_consensus"):
        if abs(float(self_score[field]) - 1.0) > 1e-9:
            raise AssertionError(f"PNU self score failed for {field}: {self_score[field]}")
    summary = {
        "stage": "2d_scoring", "counts": dict(counters), "pnu_self_check": self_score,
        "formula": "0.55*ECFP4_Tanimoto + 0.20*FCFP4_Tanimoto + 0.25*Gobbi_Pharm2D_Tanimoto",
        "library_evidence_formula": "0.65*PNU_consensus + 0.35*quality_score; channel ranking only, not final scoring",
        "diversity_is_not_collapsed_into_scalar_score": True,
        "hard_filters_are_wide_gate": True, "pains_brenk_are_soft_flags": True,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(done, summary)
    return summary
