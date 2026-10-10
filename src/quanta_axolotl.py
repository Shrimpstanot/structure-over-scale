"""Compatibility bridge between the paper's QuanTA fork and Axolotl 0.13.

The released Axolotl build only dispatches native PEFT LoRA/DoRA. The custom
``quanta`` dependency is a separate PEFT fork, so TRL also cannot identify it
when copying policy weights into colocated vLLM. This module supplies both
missing pieces while retaining Axolotl's GRPO runner and trainer lifecycle.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field, model_validator

from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer
from axolotl.integrations.base import BasePlugin


class QuanTASettings(BaseModel):
    d: int
    per_dim_features: list[int]
    dropout: float = 0.0
    merge_weights: bool = True
    target_modules: list[str] = Field(min_length=1)
    initialize_mode: str = "sum_opposite_freeze_one"
    sum_mode: bool = False

    @model_validator(mode="after")
    def dimensions_match(self):
        if self.d != len(self.per_dim_features):
            raise ValueError("quanta.d must equal len(quanta.per_dim_features)")
        return self


class QuanTAInputArgs(BaseModel):
    quanta: QuanTASettings | None = None


class QuanTAPlugin(BasePlugin):
    """Register QuanTA-only fields with Axolotl's configuration schema."""

    def get_input_args(self) -> str:
        return "quanta_axolotl.QuanTAInputArgs"

    def register(self, cfg: dict):
        if cfg.get("adapter") == "quanta" and not cfg.get("quanta"):
            raise ValueError("adapter: quanta requires a quanta block")


def _load_quanta(model, cfg):
    from quanta import QuanTAConfig, get_peft_model

    settings = cfg.quanta
    if settings is None:
        raise ValueError("adapter: quanta requires QuanTAPlugin and a quanta block")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    quanta_config = QuanTAConfig(
        d=settings.d,
        per_dim_features=list(settings.per_dim_features),
        per_dim_features2=None,
        merge_weights=settings.merge_weights,
        target_modules=list(settings.target_modules),
        sum_mode=settings.sum_mode,
        initialize_mode=settings.initialize_mode,
        bias="none",
        task_type="CAUSAL_LM",
        quanta_dropout=settings.dropout,
    )
    peft_model = get_peft_model(model, quanta_config)
    peft_model.print_trainable_parameters()
    return peft_model, quanta_config


_PATCHED = False


def install_quanta_loader_patch() -> None:
    """Patch Axolotl's adapter dispatch before training starts."""
    global _PATCHED
    if _PATCHED:
        return

    import axolotl.loaders.adapter as adapter_module

    original_load_adapter = adapter_module.load_adapter

    def load_adapter(model, cfg, adapter, inference=False):
        if adapter == "quanta":
            if inference:
                raise NotImplementedError(
                    "Load checkpoints with quanta.PeftModel.from_pretrained."
                )
            return _load_quanta(model, cfg)
        return original_load_adapter(model, cfg, adapter, inference=inference)

    adapter_module.load_adapter = load_adapter
    try:
        import axolotl.loaders.model as model_module

        model_module.load_adapter = load_adapter
    except ImportError:
        pass
    _PATCHED = True


class QuanTAGRPOTrainer(AxolotlGRPOTrainer):
    """GRPO trainer that sends effective QuanTA weights to colocated vLLM."""

    @torch.no_grad()
    def _move_model_to_vllm(self):
        from quanta import PeftModel as QuanTAPeftModel

        unwrapped = self.accelerator.unwrap_model(self.model)
        if not isinstance(unwrapped, QuanTAPeftModel):
            return super()._move_model_to_vllm()

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3 or self.is_fsdp_enabled:
            raise NotImplementedError(
                "QuanTA synchronization implements the paper's DDP path only."
            )

        was_training = unwrapped.training
        unwrapped.eval()
        try:
            base_model = unwrapped.get_base_model()
            for name, parameter in base_model.named_parameters():
                if "quanta_" in name:
                    continue
                fixed_name = self._fix_param_name_to_vllm(name)
                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(fixed_name, parameter.data)
                elif self.vllm_mode == "colocate":
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights([(fixed_name, parameter.data)])
        finally:
            unwrapped.train(was_training)

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
