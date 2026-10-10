"""Exercise the bound training code against APIs installed in the SIF.

This is an explicit diagnostic, not a scheduler-submission or launcher preflight.
Run it inside the container after mounting the repository, model, dataset, and a
materialized paper config. The two monkeypatches below stand in only for GPU
capability probes that cannot run on a CPU-only build node.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import axolotl.cli.config as config_api
    import axolotl.loaders.adapter as adapter_module
    import axolotl.loaders.model as model_module
    import axolotl.utils.trainer as trainer_utils
    from axolotl.cli.train import do_cli
    from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer
    from datasets import load_from_disk
    from quanta import PeftModel
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOTrainer
    from vllm import LLM

    from quanta_axolotl import (
        QuanTAGRPOTrainer,
        QuanTAPlugin,
        QuanTASettings,
        _load_quanta,
        install_quanta_loader_patch,
    )
    from rewards import (
        accuracy_r,
        deepmath_transform,
        strict_format_reward_shaped_4_empty,
    )

    assert callable(do_cli)
    assert issubclass(AxolotlGRPOTrainer, GRPOTrainer)
    assert issubclass(QuanTAGRPOTrainer, AxolotlGRPOTrainer)
    assert hasattr(GRPOTrainer, "_move_model_to_vllm")
    assert hasattr(GRPOTrainer, "_fix_param_name_to_vllm")
    assert hasattr(LLM, "reset_prefix_cache")
    assert QuanTAPlugin().get_input_args() == "quanta_axolotl.QuanTAInputArgs"

    install_quanta_loader_patch()
    assert adapter_module.load_adapter is model_module.load_adapter

    tiny_model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
    )
    settings = QuanTASettings(
        d=3,
        per_dim_features=[2, 2, 2],
        target_modules=["q_proj"],
    )
    wrapped, _ = _load_quanta(tiny_model, SimpleNamespace(quanta=settings))
    assert isinstance(wrapped, PeftModel)
    assert any("quanta_" in name for name, _ in wrapped.named_parameters())
    assert sum(p.numel() for p in wrapped.parameters() if p.requires_grad) > 0

    dataset = load_from_disk(str(args.dataset))
    assert len(dataset) == 5_399
    transform, _ = deepmath_transform(None)
    transformed = transform(dataset[0])
    assert transformed["prompt"].startswith("A conversation between")
    assert str(transformed["solution"])
    assert accuracy_r(
        [r"<think>2+2</think> The answer is \boxed{4}"], ["4"]
    ) == [1.0]
    assert strict_format_reward_shaped_4_empty(
        ["reasoning words here </think> answer"]
    ) == [0.75]

    # load_cfg has two hardware probes after schema/plugin validation. Replace
    # only those probes so the complete config path can run on a CPU build node.
    config_api.is_torch_bf16_gpu_available = lambda: True
    trainer_utils.check_cuda_p2p_ib_support = lambda: True
    cfg = config_api.load_cfg(str(args.config))
    assert cfg.rl == "grpo"
    assert cfg.trainer_cls == "quanta_axolotl.QuanTAGRPOTrainer"
    assert cfg.adapter == "quanta"
    assert cfg.trl.vllm_mode == "colocate"
    assert Path(cfg.base_model).is_dir()
    assert Path(cfg.datasets[0].path).is_dir()

    print("container-api-smoke-ok")


if __name__ == "__main__":
    main()
