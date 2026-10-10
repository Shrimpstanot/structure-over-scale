"""Render one immutable Axolotl config from the paper run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_BASE = REPO_ROOT / "configs" / "deepmath-qwen3-8b" / "base.yaml"
DEFAULT_MATRIX = REPO_ROOT / "configs" / "deepmath-qwen3-8b" / "matrix.yaml"
DEEPMATH_ROWS = 5_399
SUPPORTED_GROUP_SIZES = (4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-batch-size", type=int, default=64)
    parser.add_argument(
        "--group-size", type=int, choices=SUPPORTED_GROUP_SIZES, default=None
    )
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--runs-root", default="/mnt/runs")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def render(
    run_id: str,
    seed: int,
    paper_batch_size: int,
    nproc_per_node: int,
    base_path: Path = DEFAULT_BASE,
    matrix_path: Path = DEFAULT_MATRIX,
    runs_root: str = "/mnt/runs",
    group_size: int | None = None,
) -> tuple[dict, dict]:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    runs = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["runs"]
    if run_id not in runs:
        choices = ", ".join(sorted(runs))
        raise KeyError(f"Unknown run id {run_id!r}; choose one of: {choices}")

    config = deep_merge(base, runs[run_id])
    if group_size is not None:
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Unsupported group size {group_size}; choose from {SUPPORTED_GROUP_SIZES}"
            )
        config["trl"]["num_generations"] = group_size

    generations = int(config["trl"]["num_generations"])
    if seed < 0 or nproc_per_node < 1 or paper_batch_size < 1:
        raise ValueError("seed, process count, and completion batch must be positive")
    if paper_batch_size % generations:
        raise ValueError("completion batch size must be divisible by group size")

    per_rank_microbatch = int(config["micro_batch_size"])
    accumulation_denominator = nproc_per_node * per_rank_microbatch
    if paper_batch_size % accumulation_denominator:
        raise ValueError(
            "completion batch size must be divisible by process count times "
            "micro_batch_size"
        )
    gradient_accumulation = paper_batch_size // accumulation_denominator
    if gradient_accumulation < 1:
        raise ValueError("completion batch is smaller than one distributed microbatch")
    unique_prompts = paper_batch_size // generations
    optimizer_steps = DEEPMATH_ROWS // unique_prompts
    result_dir = f"{runs_root.rstrip('/')}/{run_id}/seed-{seed}"
    if generations != 8:
        result_dir += f"/g-{generations}"
    result_dir += f"/bs-{paper_batch_size}"
    config["seed"] = seed
    config["gradient_accumulation_steps"] = gradient_accumulation
    config["max_steps"] = optimizer_steps
    config["output_dir"] = result_dir

    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": seed,
        "paper_batch_size_completions": paper_batch_size,
        "num_generations_per_prompt": generations,
        "unique_prompts_per_optimizer_step": unique_prompts,
        "micro_batch_size_per_rank": per_rank_microbatch,
        "requested_nproc_per_node": nproc_per_node,
        "gradient_accumulation_steps": gradient_accumulation,
        "expected_full_optimizer_steps": optimizer_steps,
        "dropped_rows_at_epoch_end": DEEPMATH_ROWS % unique_prompts,
        "base_config": str(base_path),
        "matrix_config": str(matrix_path),
        "result_dir": result_dir,
    }
    return config, provenance


def main() -> None:
    args = parse_args()
    config, provenance = render(
        run_id=args.run_id,
        seed=args.seed,
        paper_batch_size=args.paper_batch_size,
        nproc_per_node=args.nproc_per_node,
        base_path=args.base.resolve(),
        matrix_path=args.matrix.resolve(),
        runs_root=args.runs_root,
        group_size=args.group_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=False)
    args.output.write_text(config_text, encoding="utf-8")
    provenance["resolved_config_sha256"] = hashlib.sha256(
        config_text.encode("utf-8")
    ).hexdigest()
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
