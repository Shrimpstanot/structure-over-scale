"""Offline regression tests for the paper config materializer."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from materialize_config import render  # noqa: E402


class ReproductionConfigTest(unittest.TestCase):
    def test_every_matrix_entry_materializes(self):
        matrix_path = REPO_ROOT / "configs" / "deepmath-qwen3-8b" / "matrix.yaml"
        run_ids = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["runs"]
        self.assertEqual(len(run_ids), 19)
        for run_id in run_ids:
            config, provenance = render(run_id, 42, 64, 4)
            self.assertEqual(config["max_steps"], 674)
            self.assertEqual(provenance["expected_full_optimizer_steps"], 674)
            self.assertNotIn("fsdp_version", config)

    def test_scheduler_neutral_launchers_reference_matrix_entries(self):
        matrix_path = REPO_ROOT / "configs" / "deepmath-qwen3-8b" / "matrix.yaml"
        run_ids = set(yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["runs"])
        pattern = re.compile(
            r"\b(?:lora-r\d+|dora-r\d+(?:-no-kv)?|"
            r"quanta-d\d+(?:-no-v|-no-kv)?|fft)\b"
        )
        for launcher in (
            "run_paper_matrix.sh",
            "run_batch_ablation.sh",
            "run_rank_batch_replication.sh",
            "run_group_size_reconstruction.sh",
            "run_multiseed.sh",
            "run_quanta_target_ablation.sh",
        ):
            text = (REPO_ROOT / "scripts" / launcher).read_text(encoding="utf-8")
            referenced = set(pattern.findall(text))
            self.assertTrue(referenced, launcher)
            self.assertLessEqual(referenced, run_ids, launcher)

        paper_matrix = (REPO_ROOT / "scripts" / "run_paper_matrix.sh").read_text(
            encoding="utf-8"
        )
        self.assertEqual(set(pattern.findall(paper_matrix)), run_ids)

    def test_paper_batch_horizons(self):
        expected = {
            64: (8, 674, 7),
            128: (16, 337, 7),
            256: (32, 168, 23),
        }
        for batch, (unique_prompts, steps, dropped_rows) in expected.items():
            config, provenance = render("lora-r8", 42, batch, 4)
            self.assertEqual(config["max_steps"], steps)
            self.assertEqual(
                provenance["unique_prompts_per_optimizer_step"], unique_prompts
            )
            self.assertEqual(provenance["dropped_rows_at_epoch_end"], dropped_rows)

    def test_primary_quanta_geometry(self):
        config, _ = render("quanta-d3-no-kv", 42, 64, 4)
        self.assertEqual(config["trainer_cls"], "quanta_axolotl.QuanTAGRPOTrainer")
        self.assertEqual(config["quanta"]["per_dim_features"], [16, 16, 16])
        self.assertEqual(
            config["quanta"]["target_modules"],
            ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    def test_rank_sweep_and_group_size_horizons(self):
        matrix_path = REPO_ROOT / "configs" / "deepmath-qwen3-8b" / "matrix.yaml"
        run_ids = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["runs"]
        for rank in (1, 4, 8, 16, 32, 64, 128, 256):
            config, _ = render(f"lora-r{rank}", 42, 64, 4)
            self.assertEqual(config["lora_r"], rank)
            self.assertNotIn("fsdp_version", config)

        expected = {4: (32, 4, 674), 8: (64, 8, 674), 16: (128, 16, 674)}
        for group_size, (completion_batch, accumulation, steps) in expected.items():
            config, provenance = render(
                "lora-r8", 42, completion_batch, 4, group_size=group_size
            )
            self.assertEqual(config["trl"]["num_generations"], group_size)
            self.assertEqual(config["gradient_accumulation_steps"], accumulation)
            self.assertEqual(config["max_steps"], steps)
            self.assertEqual(provenance["unique_prompts_per_optimizer_step"], 8)
            group_component = f"/g-{group_size}" if group_size != 8 else ""
            self.assertTrue(
                config["output_dir"].endswith(
                    f"/lora-r8/seed-42{group_component}/bs-{completion_batch}"
                )
            )

        default_config, _ = render("lora-r8", 42, 64, 4, group_size=8)
        self.assertTrue(default_config["output_dir"].endswith("/lora-r8/seed-42/bs-64"))


if __name__ == "__main__":
    unittest.main()
