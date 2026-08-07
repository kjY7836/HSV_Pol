from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_affinity import (  # noqa: E402
    fill_prediction_table,
    install_prediction_table,
    ligand_graph,
    pocket_graph,
    read_prediction_checkpoint,
    read_receptor_atoms,
)


class AffinityInferenceTests(unittest.TestCase):
    def test_ligand_and_local_pocket_graphs(self) -> None:
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        lig = ligand_graph(mol)
        self.assertEqual(lig.x.numel(), 3)
        self.assertEqual(tuple(lig.pos.shape), (3, 3))
        self.assertEqual(tuple(lig.edge_index.shape), (2, 4))

        receptor = read_receptor_atoms(
            ROOT / "receptors" / "8V1Q_WT_UL30_DNA_no_water.pdb"
        )
        # Move the test ligand into the production docking box.
        lig.pos = lig.pos - lig.pos.mean(dim=0) + torch.tensor([147.660, 145.083, 124.388])
        pocket, residue_count, nearest = pocket_graph(receptor, lig, radius=10.0)
        self.assertGreater(pocket.x.numel(), 0)
        self.assertEqual(pocket.residue_type.shape, pocket.x.shape)
        self.assertGreater(residue_count, 0)
        self.assertLess(nearest, 8.0)

    def test_fill_prediction_table_preserves_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            template = tmp_path / "affinity_predictions.csv"
            destination = tmp_path / "filled.csv"
            with template.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["pose_id", "ligand_record_index", "predicted_paffinity", "keep"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"pose_id": "p1", "ligand_record_index": 1, "predicted_paffinity": "", "keep": "a"},
                        {"pose_id": "p2", "ligand_record_index": 2, "predicted_paffinity": "", "keep": "b"},
                    ]
                )
            fill_prediction_table(
                template,
                destination,
                [
                    {"pose_id": "p1", "ligand_record_index": 1, "predicted_paffinity": 5.25},
                    {"pose_id": "p2", "ligand_record_index": 2, "predicted_paffinity": 6.5},
                ],
            )
            with destination.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["pose_id"] for row in rows], ["p1", "p2"])
            self.assertEqual([row["keep"] for row in rows], ["a", "b"])
            self.assertEqual(
                [row["predicted_paffinity"] for row in rows],
                ["5.25000000", "6.50000000"],
            )

    def test_fill_prediction_table_rejects_pose_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            template = tmp_path / "affinity_predictions.csv"
            template.write_text(
                "pose_id,ligand_record_index,predicted_paffinity\np1,1,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pose_id mismatch"):
                fill_prediction_table(
                    template,
                    tmp_path / "filled.csv",
                    [{"pose_id": "wrong", "ligand_record_index": 1, "predicted_paffinity": 5.0}],
                )

    def test_install_prediction_table_is_atomic_and_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            template = tmp_path / "affinity_predictions.csv"
            template.write_text(
                "pose_id,ligand_record_index,predicted_paffinity\np1,1,\n",
                encoding="utf-8",
            )

            backup = install_prediction_table(
                template,
                [{"pose_id": "p1", "ligand_record_index": 1, "predicted_paffinity": 7.25}],
            )

            self.assertEqual(
                backup.read_text(encoding="utf-8"),
                "pose_id,ligand_record_index,predicted_paffinity\np1,1,\n",
            )
            self.assertEqual(
                template.read_text(encoding="utf-8"),
                "pose_id,ligand_record_index,predicted_paffinity\np1,1,7.25000000\n",
            )

    def test_prediction_checkpoint_requires_contiguous_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "predictions.csv.partial"
            checkpoint.write_text(
                "ligand_record_index,pose_id,ligand_id,smina_score,ligand_heavy_atoms,"
                "pocket_heavy_atoms,pocket_residues,nearest_receptor_distance,"
                "predicted_paffinity,approx_concentration_nM\n"
                "2,p2,l2,-7,10,100,5,2.0,6.0,1000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contiguous 1-based prefix"):
                read_prediction_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
