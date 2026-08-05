from __future__ import annotations

import csv
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from hsv_screen.affinity_input import run as run_affinity_input
from hsv_screen.chemistry import classify_and_standardize, hard_filter_reasons, score_reference_self
from hsv_screen.config import load_config
from hsv_screen.docking import collect
from hsv_screen.ligands import _generate_task
from hsv_screen.postdock3d import run as run_postdock3d
from hsv_screen.prescreen_pool import fraction_quotas, select_source
from hsv_screen.reference import PNU_SMILES, validate_and_prepare
from hsv_screen.scoring import aggregate_scores, scaled_affinity
from hsv_screen.screen3d import (
    build_feature_model, init_worker, parse_pocket_atoms, score_conformation_similarity,
    score_conformer, score_task,
)


class ChemistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config(Path(__file__).parents[1] / "config/sample_10000.json")
        cls.settings = cls.config["standardization"]

    def record(self, smiles: str) -> dict:
        return {"original_smiles": smiles, "compound_id": "x", "source_pool": "unlabeled"}

    def test_unambiguous_salt_is_separated(self):
        result = classify_and_standardize(self.record("CC(=O)Oc1ccccc1C(=O)O.[Cl-]"), self.settings)
        self.assertEqual(result["status"], "valid_parent")
        self.assertEqual(result["component_class"], "simple_salt_or_solvate")
        self.assertNotIn(".", result["standardized_smiles"])

    def test_two_unrecognized_organic_components_are_quarantined(self):
        result = classify_and_standardize(self.record("c1ccccc1.Cc1ccccc1"), self.settings)
        self.assertEqual(result["status"], "quarantine_ambiguous_components")
        self.assertEqual(result["standardized_smiles"], "")

    def test_simple_sodium_salt_is_separated(self):
        result = classify_and_standardize(self.record("CC(=O)[O-].[Na+]"), self.settings)
        self.assertEqual(result["status"], "valid_parent")
        self.assertEqual(result["component_class"], "simple_salt_or_solvate")

    def test_organometal_record_is_quarantined(self):
        result = classify_and_standardize(self.record("C[Mg]Br"), self.settings)
        self.assertEqual(result["status"], "quarantine_special_element")

    def test_permanent_charge_is_preserved(self):
        result = classify_and_standardize(self.record("C[N+](C)(C)CCc1ccccc1.[Cl-]"), self.settings)
        self.assertEqual(result["status"], "valid_parent")
        self.assertNotEqual(int(result["formal_charge"]), 0)

    def test_pnu_2d_self_scores_are_one(self):
        result = score_reference_self(PNU_SMILES, self.config["fingerprint"], self.config["filters"])
        for field in ("pnu_ecfp_similarity", "pnu_fcfp_similarity", "pharm2d_similarity", "pnu_consensus"):
            self.assertAlmostEqual(result[field], 1.0, places=12)

    def test_wide_filter_boundaries(self):
        props = {"heavy_atoms": 9, "mw": 179, "clogp": 0, "tpsa": 30, "hbd": 1, "hba": 2,
                 "rotatable_bonds": 1, "ring_count": 1, "formal_charge": 0}
        reasons = hard_filter_reasons(props, {6, 8}, self.config["filters"])
        self.assertIn("heavy_atoms_low", reasons)
        self.assertIn("mw_low", reasons)


class ReferenceAndQuotaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config(Path(__file__).parents[1] / "config/sample_10000.json")

    def test_reference_checksums_and_bound_instance(self):
        manifest = validate_and_prepare(self.config, prepare_pdbqt=False)
        self.assertEqual(manifest["ligand_instances"]["C:1201"], 30)
        self.assertEqual(len(manifest["checksums"]), 10)
        self.assertEqual({item["pdb_id"] for item in manifest["target_structures"]}, {"8V1Q", "8V1R"})
        self.assertTrue(all(item["uniprot_id"] == "H9E937" for item in manifest["target_structures"]))

    def test_experimental_pnu_3d_self_check(self):
        refs = self.config["references"]
        project = Path(self.config["_project_dir"])
        bound = Chem.SDMolSupplier(str(project / refs["bound_ligand_sdf"]), removeHs=False)[0]
        pocket = parse_pocket_atoms(project / refs["pdb_file"], refs["ligand_resname"], bound,
                                    self.config["structure_3d"]["pocket_radius"])
        features = build_feature_model(bound, pocket, self.config["structure_3d"])
        init_worker(Chem.MolToMolBlock(bound), pocket, features, self.config["structure_3d"])
        combined, feature, shape, clash = score_conformer(Chem.Mol(bound), 0)
        postdock, post_feature, post_shape, _ = score_conformation_similarity(Chem.Mol(bound), 0)
        self.assertGreaterEqual(combined, 0.80)
        self.assertGreaterEqual(feature, 0.95)
        self.assertGreaterEqual(shape, 0.99)
        self.assertGreaterEqual(postdock, 0.99)
        self.assertGreaterEqual(post_feature, 0.99)
        self.assertGreaterEqual(post_shape, 0.99)

    def test_stereo_verified_random_coordinate_fallback(self):
        refs = self.config["references"]
        project = Path(self.config["_project_dir"])
        bound = Chem.SDMolSupplier(str(project / refs["bound_ligand_sdf"]), removeHs=False)[0]
        pocket = parse_pocket_atoms(project / refs["pdb_file"], refs["ligand_resname"], bound,
                                    self.config["structure_3d"]["pocket_radius"])
        features = build_feature_model(bound, pocket, self.config["structure_3d"])
        init_worker(Chem.MolToMolBlock(bound), pocket, features, self.config["structure_3d"])
        difficult = "C=C1C(=O)[C@]23C[C@H]1CC[C@H]2[C@@]12CO[C@@]3(O)[C@@H](O)[C@@H]1C(C)(C)C=CC2=O"
        _, result = score_task((0, difficult, "0a92533592526ceb5bc2256c459707a1cfdc69b6",
                                "scaffold_diversity"))
        self.assertEqual(result["structure3d_status"], "feasible_only")
        self.assertIn("stereo_verified", result["structure3d_embedding_method"])

    def test_dynamic_all_activity_policy(self):
        final = self.config["final_selection"]
        self.assertEqual(final["activity_recorded_policy"], "all_3d_feasible")
        self.assertTrue(final["require_all_activity_recorded_3d"])
        self.assertEqual(final["unlabeled_count"], 167)
        protocol = self.config["docking"]["protocols"][0]
        self.assertEqual(protocol["scope"]["max_parents"], 0)
        self.assertEqual(self.config["workers"], 64)
        self.assertEqual(self.config["docking"]["parallel_jobs"], 64)

    def test_one_ph_adjusted_state_per_parent(self):
        row = {
            "structure_key": "1" * 40, "standardized_smiles": "CNC",
            "ph_protonated_smiles": "C[NH2+]C", "compound_id": "one",
            "source_pool": "unlabeled", "selection_channel": "diversity",
        }
        parent, states = _generate_task((row, self.config["ligand_states"]))
        self.assertEqual(parent, row["structure_key"])
        self.assertEqual(len(states), 1)
        metadata, mol_block = states[0]
        self.assertEqual(metadata["state_id"], "1111111111111111_s01")
        self.assertEqual(metadata["state_rank"], 1)
        self.assertEqual(metadata["formal_charge"], 1)
        self.assertIn("openbabel_pH_7.4", metadata["state_origin"])
        self.assertIsNotNone(Chem.MolFromMolBlock(mol_block, removeHs=False))

    def test_200k_quota_math(self):
        fractions = {"pnu_2d": 0.20, "pharm2d": 0.15, "scaffold_diversity": 0.45, "exploration": 0.20}
        quotas = fraction_quotas(14837, fractions)
        self.assertEqual(sum(quotas.values()), 14837)
        self.assertEqual(quotas, {"pnu_2d": 2967, "pharm2d": 2226, "scaffold_diversity": 6677, "exploration": 2967})
        unlabeled = fraction_quotas(185163, fractions)
        self.assertEqual(unlabeled, {"pnu_2d": 37033, "pharm2d": 27774, "scaffold_diversity": 83323, "exploration": 37033})

    def test_prescreen_has_no_duplicates(self):
        records = []
        for index in range(100):
            records.append({"structure_key": f"{index:040x}", "pnu_consensus": index / 100,
                            "pharm2d_similarity": (99-index) / 100, "quality_score": 0.5,
                            "library_evidence_score": index / 100,
                            "soft_alert_count": 0, "murcko_scaffold": f"s{index//2}"})
        selected, channels, quotas = select_source(records, 80,
            {"pnu_2d": 0.20, "pharm2d": 0.15, "scaffold_diversity": 0.45, "exploration": 0.20}, 1)
        self.assertEqual(len(selected), 80)
        self.assertEqual(len(set(selected)), 80)
        self.assertEqual(sum(quotas.values()), 80)


class IntegratedScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config(Path(__file__).parents[1] / "config/sample_10000.json")

    def relaxed_settings(self):
        return {**self.config["integrated_scoring"], "required_protocols": [],
                "postdock_3d_required": False}

    def test_scaled_affinity_is_monotonic(self):
        settings = self.config["integrated_scoring"]["affinity"]
        self.assertGreater(scaled_affinity(8.0, settings), scaled_affinity(6.0, settings))
        self.assertAlmostEqual(scaled_affinity(6.0, settings), 0.5)

    def test_final_formula_contains_only_requested_components(self):
        weights = self.config["integrated_scoring"]["profiles"]["wt"]["weights"]
        self.assertEqual(weights, {
            "wt_affinity": 0.55, "postdock_3d_similarity": 0.20,
            "docking_consensus": 0.15, "qed": 0.10,
        })

    def test_external_affinity_drives_parent_ranking(self):
        parents = [
            {"structure_key": "p1", "compound_id": "one", "source_pool": "unlabeled",
             "selection_channel": "diversity", "qed": "0.7"},
            {"structure_key": "p2", "compound_id": "two", "source_pool": "unlabeled",
             "selection_channel": "diversity", "qed": "0.7"},
        ]
        docking = [
            {"pose_id": "a", "parent_structure_key": "p1", "state_id": "p1s", "protocol_id": "wt",
             "receptor_id": "r", "box_id": "b", "mutant_id": "WT", "smina_score": "-8"},
            {"pose_id": "b", "parent_structure_key": "p2", "state_id": "p2s", "protocol_id": "wt",
             "receptor_id": "r", "box_id": "b", "mutant_id": "WT", "smina_score": "-7"},
        ]
        affinity = [
            {"pose_id": "a", "predicted_paffinity": "8.0"},
            {"pose_id": "b", "predicted_paffinity": "6.0"},
        ]
        ranked, diagnostics = aggregate_scores(
            parents, docking, affinity, self.relaxed_settings(), "wt")
        self.assertEqual([row["parent_structure_key"] for row in ranked], ["p1", "p2"])
        self.assertEqual(diagnostics["affinity_pose_coverage"], 1.0)
        self.assertEqual(diagnostics["affinity_parent_coverage"], 1.0)

    def test_affinity_unknown_pose_is_rejected(self):
        parents = [{"structure_key": "p1", "qed": "0.7"}]
        docking = [{"pose_id": "known", "parent_structure_key": "p1", "state_id": "s1",
                    "protocol_id": "wt", "mutant_id": "WT", "smina_score": "-7"}]
        affinity = [{"pose_id": "unknown", "predicted_paffinity": "7"}]
        with self.assertRaisesRegex(ValueError, "unknown pose_id"):
            aggregate_scores(parents, docking, affinity, self.relaxed_settings(), "wt")

    def test_affinity_parent_coverage_is_enforced(self):
        parents = [
            {"structure_key": "p1", "qed": "0.7"},
            {"structure_key": "p2", "qed": "0.7"},
        ]
        docking = [
            {"pose_id": "a", "parent_structure_key": "p1", "state_id": "s1",
             "protocol_id": "wt", "mutant_id": "WT", "smina_score": "-7"},
            {"pose_id": "b", "parent_structure_key": "p2", "state_id": "s2",
             "protocol_id": "wt", "mutant_id": "WT", "smina_score": "-7"},
        ]
        affinity = [{"pose_id": "a", "predicted_paffinity": "7"}]
        relaxed = {**self.relaxed_settings(),
                   "affinity": {**self.config["integrated_scoring"]["affinity"],
                                "minimum_pose_coverage": 0.5}}
        with self.assertRaisesRegex(ValueError, "parent coverage"):
            aggregate_scores(parents, docking, affinity, relaxed, "wt")

    def test_single_8v1q_protocol_uses_postdock_similarity(self):
        parents = [{"structure_key": "p1", "qed": "0.7"}]
        docking = [{"pose_id": "open", "parent_structure_key": "p1", "state_id": "s1",
                    "protocol_id": "8v1q_open_wt", "mutant_id": "WT", "smina_score": "-7"}]
        affinity = [{"pose_id": "open", "predicted_paffinity": "7"}]
        similarity = [{"pose_id": "open", "postdock_3d_status": "ok",
                       "postdock_3d_similarity": "0.8"}]
        ranked, diagnostics = aggregate_scores(
            parents, docking, affinity, self.config["integrated_scoring"], "wt", similarity)
        self.assertEqual(ranked[0]["wt_protocol_count"], 1)
        self.assertAlmostEqual(ranked[0]["postdock_3d_similarity"], 0.8)
        self.assertEqual(diagnostics["scoring_docking_pose_count"], 1)

    def test_non_wt_pose_is_rejected(self):
        parents = [{"structure_key": "p1", "qed": "0.7"}]
        docking = [{"pose_id": "m", "parent_structure_key": "p1", "state_id": "s",
                    "protocol_id": "mutant", "mutant_id": "M1", "smina_score": "-7"}]
        affinity = [{"pose_id": "m", "predicted_paffinity": "7"}]
        with self.assertRaisesRegex(ValueError, "WT-only"):
            aggregate_scores(parents, docking, affinity, self.relaxed_settings(), "wt")

    def test_postdock_similarity_changes_ranking_when_affinity_is_equal(self):
        parents = [
            {"structure_key": "p1", "qed": "0.7"},
            {"structure_key": "p2", "qed": "0.7"},
        ]
        docking = [
            {"pose_id": "a", "parent_structure_key": "p1", "state_id": "s1",
             "protocol_id": "8v1q_open_wt", "mutant_id": "WT", "smina_score": "-7"},
            {"pose_id": "b", "parent_structure_key": "p2", "state_id": "s2",
             "protocol_id": "8v1q_open_wt", "mutant_id": "WT", "smina_score": "-7"},
        ]
        affinity = [{"pose_id": pose_id, "predicted_paffinity": "7"}
                    for pose_id in ("a", "b")]
        similarity = [
            {"pose_id": "a", "postdock_3d_status": "ok", "postdock_3d_similarity": "0.9"},
            {"pose_id": "b", "postdock_3d_status": "ok", "postdock_3d_similarity": "0.1"},
        ]
        settings = {**self.config["integrated_scoring"], "required_protocols": []}
        ranked, _ = aggregate_scores(parents, docking, affinity, settings, "wt", similarity)
        self.assertEqual([row["parent_structure_key"] for row in ranked], ["p1", "p2"])


class DockingCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config(Path(__file__).parents[1] / "config/sample_10000.json")

    def test_smina_title_restores_traceability_and_pdbbind_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_dir = root / "05_final_selection"
            states_dir = root / "06_ligand_states"
            output_dir = root / "07_smina" / "outputs" / "8v1q_open_wt"
            selection_dir.mkdir(parents=True)
            states_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            with (selection_dir / "selected.csv").open(
                    "w", encoding="utf-8", newline="") as handle:
                fields = [
                    "structure_key", "compound_id", "source_pool", "selection_channel",
                    "standardized_smiles", "qed", "pnu_ecfp_similarity",
                    "pnu_fcfp_similarity", "pharm2d_similarity", "pnu_consensus",
                    "quality_score", "library_evidence_score", "structure3d_score",
                    "structure3d_feature_score", "structure3d_shape_similarity",
                    "structure3d_clash_fraction",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "structure_key": "parent_001", "compound_id": "compound_001",
                    "source_pool": "unlabeled", "selection_channel": "diversity",
                    "standardized_smiles": "CCO", "qed": "0.61",
                    "pnu_ecfp_similarity": "0.1", "pnu_fcfp_similarity": "0.2",
                    "pharm2d_similarity": "0.3", "pnu_consensus": "0.2",
                    "quality_score": "0.61", "library_evidence_score": "0.4",
                    "structure3d_score": "0.5", "structure3d_feature_score": "0.4",
                    "structure3d_shape_similarity": "0.6",
                    "structure3d_clash_fraction": "0.0",
                })
            with (states_dir / "ligand_states.csv").open("w", encoding="utf-8", newline="") as handle:
                fields = ["state_id", "parent_structure_key", "compound_id", "source_pool",
                          "selection_channel", "state_smiles", "embed_status"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "state_id": "state_001", "parent_structure_key": "parent_001",
                    "compound_id": "compound_001", "source_pool": "unlabeled",
                    "selection_channel": "diversity", "state_smiles": "CCO",
                    "embed_status": "ok",
                })

            docked_path = output_dir / "docked_00001.sdf"
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=7), 0)
            mol.SetProp("_Name", "state_001")
            mol.SetProp("minimizedAffinity", "-7.25")
            writer = Chem.SDWriter(str(docked_path))
            writer.write(mol)
            writer.close()

            jobs_dir = root / "07_smina"
            with (jobs_dir / "jobs.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["protocol_id", "output"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"protocol_id": "8v1q_open_wt", "output": str(docked_path)})

            config = deepcopy(self.config)
            config["output_dir"] = str(root)
            config["workers"] = 1
            summary = collect(config, force=True)
            self.assertEqual(summary["pose_count"], 1)
            consolidated = Path(summary["consolidated_pose_sdf"])
            pose = next(m for m in Chem.SDMolSupplier(str(consolidated), removeHs=False) if m)
            self.assertEqual(pose.GetProp("_Name"), "8v1q_open_wt|state_001|1")
            self.assertEqual(pose.GetProp("parent_structure_key"), "parent_001")
            self.assertEqual(pose.GetNumConformers(), 1)
            with (root / "08_docking_results" / "docking_poses.csv").open(
                    encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["ligand_pose_sdf"], str(consolidated))
            self.assertEqual(row["ligand_record_index"], "1")
            self.assertTrue(Path(row["receptor_pdb"]).is_file())
            postdock = run_postdock3d(config, force=True)
            self.assertEqual(postdock["input_pose_count"], 1)
            self.assertEqual(postdock["pose_coverage"], 1.0)
            with Path(postdock["output"]).open(encoding="utf-8", newline="") as handle:
                similarity = next(csv.DictReader(handle))
            self.assertEqual(similarity["pose_id"], "8v1q_open_wt|state_001|1")
            self.assertEqual(similarity["postdock_3d_status"], "ok")
            affinity = run_affinity_input(config, force=True)
            self.assertEqual(affinity["pose_count"], 1)
            self.assertEqual(affinity["states_per_parent"], 1)
            self.assertTrue(Path(affinity["receptor_pdb"]).is_file())
            self.assertTrue(Path(affinity["ligand_pose_sdf"]).is_file())
            with Path(affinity["prediction_table"]).open(
                    encoding="utf-8", newline="") as handle:
                affinity_row = next(csv.DictReader(handle))
            self.assertEqual(affinity_row["predicted_paffinity"], "")
            self.assertEqual(affinity_row["qed"], "0.61")
            self.assertEqual(affinity_row["smina_score"], "-7.25")
            self.assertEqual(affinity_row["docking_percentile"], "1.0")
            self.assertNotIn("pose_quality", affinity_row)
            self.assertNotIn("ligand_efficiency", affinity_row)
            self.assertNotIn("chemical_quality", affinity_row)


if __name__ == "__main__":
    unittest.main()
