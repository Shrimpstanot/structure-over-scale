"""Merge a trusted LoRA, DoRA, or QuanTA checkpoint into a dense model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_config_path = args.adapter / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    peft_type = str(adapter_config.get("peft_type", "")).upper()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if peft_type == "QUANTA":
        from quanta import PeftModel

        adapted = PeftModel.from_pretrained(base_model, str(args.adapter))
        adapted.eval()
        dense_model = adapted.get_base_model()
    else:
        from peft import PeftModel

        adapted = PeftModel.from_pretrained(base_model, str(args.adapter))
        dense_model = adapted.merge_and_unload(safe_merge=True)

    args.output.mkdir(parents=True, exist_ok=True)
    dense_model.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    print(f"Merged {peft_type or 'PEFT'} checkpoint into {args.output}")


if __name__ == "__main__":
    main()
