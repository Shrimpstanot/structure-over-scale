"""Opt-in offline integrity report for mounted reproduction assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path


EXPECTED_DATASET_ROWS = 5_399
EXPECTED_COLUMNS = {
    "question",
    "final_answer",
    "difficulty",
    "topic",
    "r1_solution_1",
    "r1_solution_2",
    "r1_solution_3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> set[str]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 2 or header_length > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(handle.read(header_length))
    return {key for key in header if key != "__metadata__"}


def check_model(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    missing = [name for name in shards if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model shards: {missing}")

    seen = set()
    for shard in shards:
        seen.update(read_safetensors_header(model_dir / shard))
    absent = sorted(set(weight_map) - seen)
    if absent:
        raise ValueError(f"Indexed tensors absent from shards: {absent[:5]}")
    return {
        "path": str(model_dir),
        "shards": shards,
        "indexed_tensors": len(weight_map),
    }


def check_dataset(dataset_root: Path, verify_hashes: bool) -> dict:
    train_dir = dataset_root / "train" if (dataset_root / "train").is_dir() else dataset_root
    provenance_path = train_dir.parent / "preprocessing.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["output_rows"] != EXPECTED_DATASET_ROWS:
        raise ValueError(f"Expected {EXPECTED_DATASET_ROWS} rows")

    for item in provenance["output_files"]:
        path = train_dir / item["path"]
        if path.stat().st_size != item["size"]:
            raise ValueError(f"Size mismatch: {path}")
        if verify_hashes and sha256(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")

    info = json.loads((train_dir / "dataset_info.json").read_text(encoding="utf-8"))
    columns = set(info["features"])
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {sorted(columns)}")
    return {
        "path": str(train_dir),
        "rows": provenance["output_rows"],
        "difficulty_distribution": provenance["difficulty_distribution"],
        "hashes_verified": verify_hashes,
    }


def check_sif(sif_path: Path) -> dict:
    result = subprocess.run(
        ["singularity", "inspect", str(sif_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"path": str(sif_path), "inspect": result.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--sif", type=Path)
    parser.add_argument("--quick", action="store_true", help="skip dataset hashing")
    args = parser.parse_args()

    report = {}
    if args.model:
        report["model"] = check_model(args.model.resolve())
    if args.dataset:
        report["dataset"] = check_dataset(args.dataset.resolve(), not args.quick)
    if args.sif:
        report["sif"] = check_sif(args.sif.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
