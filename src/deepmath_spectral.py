from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import opt_einsum as oe


EXPECTED_TARGET_MODULES = (
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_MODEL_DIR = REPO_ROOT / "models" / "qwen3-8b"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "deepmath" / "qwen"
DEFAULT_SCORES_FILE = REPO_ROOT / "scores.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "deepmath_spectral"
DEFAULT_REPRESENTATIVE_LAYER = "model.layers.18.self_attn.q_proj"

LAYER_WEIGHT_RE = re.compile(
    r"^(model\.layers\.(\d+)\.(?:self_attn|mlp)\.(" + "|".join(EXPECTED_TARGET_MODULES) + r"))\.weight$"
)


class SpectralEvalError(RuntimeError):
    """Base error for spectral evaluation failures."""


class AuditError(SpectralEvalError):
    """Raised when a strict artifact audit fails."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


@dataclass
class OrientationResolution:
    fan_in_fan_out: bool
    source: str


@dataclass
class DeltaMatrices:
    spectral: torch.Tensor
    full: torch.Tensor


@dataclass
class RunSpec:
    name: str
    label: str
    kind: str
    root: Path
    score: float
    config_path: Path | None
    artifact_path: Path
    config: dict[str, Any] = field(default_factory=dict)
    realized_modules: list[str] = field(default_factory=list)
    realized_module_types: list[str] = field(default_factory=list)
    missing_modules: list[str] = field(default_factory=list)
    missing_module_types: list[str] = field(default_factory=list)
    extra_modules: list[str] = field(default_factory=list)
    extra_module_types: list[str] = field(default_factory=list)
    orientation: dict[str, OrientationResolution] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        data["config_path"] = str(self.config_path) if self.config_path else None
        data["artifact_path"] = str(self.artifact_path)
        data["orientation"] = {
            module: asdict(resolution) for module, resolution in self.orientation.items()
        }
        return data


def _transpose_if_needed(weight: torch.Tensor, fan_in_fan_out: bool) -> torch.Tensor:
    return weight.T if fan_in_fan_out else weight


def _module_type(module_path: str) -> str:
    return module_path.rsplit(".", 1)[-1]


def _parse_score_value(raw_value: str) -> float:
    value = raw_value.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    return float(value)


def parse_scores(scores_path: str | Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    current_section: str | None = None

    for line in Path(scores_path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            header = stripped[:-1]
            if header.startswith("LoRA"):
                current_section = "LoRA"
            elif header.startswith("DoRA"):
                current_section = "DoRA"
            elif header == "QuanTA":
                current_section = "QuanTA"
            elif header == "QuanTA (with kv projections)":
                current_section = "QuanTA (kv)"
            elif header == "QuanTA (no kv projections)":
                current_section = "QuanTA (no-kv)"
            elif header == "Qwen 3 8B base DeepMath":
                current_section = None
            else:
                current_section = header
            continue

        if ":" not in stripped:
            continue

        key, raw_value = [part.strip() for part in stripped.split(":", 1)]
        score = _parse_score_value(raw_value)

        if current_section == "LoRA" and key.startswith("Rank "):
            label = f"LoRA r={key.split()[-1]}"
        elif current_section == "DoRA" and key.startswith("Rank "):
            label = f"DoRA r={key.split()[-1]}"
        elif current_section == "QuanTA" and key.startswith("d="):
            label = f"QuanTA {key.split()[0]}"
        elif current_section == "QuanTA (kv)" and key.startswith("d="):
            label = f"QuanTA {key.split()[0]} (kv)"
        elif current_section == "QuanTA (no-kv)" and key.startswith("d="):
            label = f"QuanTA {key.split()[0]} (no-kv)"
        elif key == "Full FT":
            label = "FFT"
        else:
            continue
        scores[label] = score

    return scores


def _label_and_kind_for_run(run_name: str) -> tuple[str, str]:
    if run_name.startswith("qwen_lora_rank"):
        rank = run_name.removeprefix("qwen_lora_rank")
        return f"LoRA r={rank}", "lora"
    if run_name.startswith("qwen_dora_rank"):
        rank = run_name.removeprefix("qwen_dora_rank")
        return f"DoRA r={rank}", "dora"
    if run_name == "qwen_quanta_d3_kv":
        return "QuanTA d=3 (kv)", "quanta"
    if run_name == "qwen_quanta_d3_no-kv":
        return "QuanTA d=3 (no-kv)", "quanta"
    if run_name == "qwen_quanta_d4_kv":
        return "QuanTA d=4 (kv)", "quanta"
    if run_name == "qwen_quanta_d4_no-kv":
        return "QuanTA d=4 (no-kv)", "quanta"
    if run_name.startswith("qwen_quanta_d3"):
        return "QuanTA d=3", "quanta"
    if run_name.startswith("qwen_quanta_d4"):
        return "QuanTA d=4", "quanta"
    if run_name == "qwen_full_ft":
        return "FFT", "fft"
    raise SpectralEvalError(f"Unrecognized run directory name: {run_name}")


def _strip_json_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(_strip_json_comments(path.read_text()))


def _artifact_contains_dora_magnitude(path: Path) -> bool:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return any(key.endswith(".lora_magnitude_vector") for key in handle.keys())


def discover_runs(
    runs_root: str | Path,
    scores_path: str | Path = DEFAULT_SCORES_FILE,
    labels: Iterable[str] | None = None,
) -> list[RunSpec]:
    scores = parse_scores(scores_path)
    selected_labels = set(labels or [])
    runs: list[RunSpec] = []

    for child in sorted(Path(runs_root).iterdir()):
        if not child.is_dir():
            continue
        try:
            label, kind = _label_and_kind_for_run(child.name)
        except SpectralEvalError:
            continue

        if selected_labels and label not in selected_labels:
            continue
        if kind in {"lora", "dora", "quanta"}:
            config_path = child / "adapter_config.json"
        else:
            config_path = child / "config.json"

        if kind in {"lora", "dora"}:
            artifact_path = child / "adapter_model.safetensors"
        elif kind == "quanta":
            if (child / "adapter_model.bin").exists():
                artifact_path = child / "adapter_model.bin"
            elif (child / "checkpoint-674" / "model.safetensors").exists():
                artifact_path = child / "checkpoint-674" / "model.safetensors"
            else:
                raise SpectralEvalError(f"Could not find QuanTA artifact for {child}")
        else:
            artifact_path = child / "model.safetensors"

        if not artifact_path.exists():
            raise SpectralEvalError(f"Missing artifact for {child}: {artifact_path}")

        if kind == "dora" and artifact_path.suffix == ".safetensors":
            if not _artifact_contains_dora_magnitude(artifact_path):
                continue

        if label not in scores:
            continue

        runs.append(
            RunSpec(
                name=child.name,
                label=label,
                kind=kind,
                root=child,
                score=scores[label],
                config_path=config_path if config_path.exists() else None,
                artifact_path=artifact_path,
                config=_read_json_if_exists(config_path),
            )
        )

    if selected_labels:
        found = {run.label for run in runs}
        missing = sorted(selected_labels - found)
        if missing:
            raise SpectralEvalError(f"Requested labels not found: {', '.join(missing)}")

    return runs


class ShardedSafetensorIndex:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.index_path = self.model_dir / "model.safetensors.index.json"
        if not self.index_path.exists():
            raise SpectralEvalError(f"Missing base model index: {self.index_path}")
        payload = json.loads(self.index_path.read_text())
        self.weight_map: dict[str, str] = payload["weight_map"]

    def has_key(self, key: str) -> bool:
        return key in self.weight_map

    def keys(self) -> list[str]:
        return sorted(self.weight_map)

    @lru_cache(maxsize=8192)
    def get_tensor(self, key: str) -> torch.Tensor:
        shard_path = self.model_dir / self.weight_map[key]
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    @lru_cache(maxsize=8192)
    def get_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self.get_tensor(key).shape)


class SingleSafetensorStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise SpectralEvalError(f"Missing safetensors file: {self.path}")
        self._keys: list[str] | None = None

    def keys(self) -> list[str]:
        if self._keys is None:
            with safe_open(str(self.path), framework="pt", device="cpu") as handle:
                self._keys = list(handle.keys())
        return self._keys

    @lru_cache(maxsize=8192)
    def get_tensor(self, key: str) -> torch.Tensor:
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


@lru_cache(maxsize=32)
def _load_safetensor_state(path: str) -> dict[str, torch.Tensor]:
    return load_file(path)


@lru_cache(maxsize=32)
def _load_torch_state(path: str) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def _load_run_state(run: RunSpec) -> dict[str, torch.Tensor]:
    if run.kind in {"lora", "dora"}:
        return _load_safetensor_state(str(run.artifact_path))
    if run.kind == "quanta":
        if run.artifact_path.suffix == ".bin":
            return _load_torch_state(str(run.artifact_path))
        return _load_safetensor_state(str(run.artifact_path))
    raise SpectralEvalError(f"Run kind {run.kind} does not load a small adapter state dict")


def collect_expected_modules(base_index: ShardedSafetensorIndex) -> list[str]:
    modules = []
    for key in base_index.keys():
        match = LAYER_WEIGHT_RE.match(key)
        if match:
            modules.append(match.group(1))
    return sorted(modules)


def extract_realized_modules(run: RunSpec) -> list[str]:
    if run.kind in {"lora", "dora"}:
        state = _load_run_state(run)
        modules = set()
        for key in state:
            if key.endswith(".lora_A.weight"):
                module = key[: -len(".lora_A.weight")]
            elif key.endswith(".lora_B.weight"):
                module = key[: -len(".lora_B.weight")]
            elif key.endswith(".lora_magnitude_vector"):
                module = key[: -len(".lora_magnitude_vector")]
            else:
                continue
            module = module.removeprefix("base_model.model.")
            modules.add(module)
        return sorted(modules)

    if run.kind == "quanta":
        state = _load_run_state(run)
        modules = set()
        for key in state:
            if ".quanta_weights." in key:
                module = key.split(".quanta_weights.", 1)[0]
            elif ".quanta_weights2." in key:
                module = key.split(".quanta_weights2.", 1)[0]
            else:
                continue
            module = module.removeprefix("base_model.model.")
            modules.add(module)
        return sorted(modules)

    if run.kind == "fft":
        store = SingleSafetensorStore(run.artifact_path)
        modules = set()
        for key in store.keys():
            match = LAYER_WEIGHT_RE.match(key)
            if match:
                modules.add(match.group(1))
        return sorted(modules)

    raise SpectralEvalError(f"Unsupported run kind for module extraction: {run.kind}")


def _normalize_fan_in_fan_out_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    return None


def _resolve_config_orientation(
    config: dict[str, Any],
    module_path: str,
) -> OrientationResolution | None:
    value = config.get("fan_in_fan_out")
    normalized = _normalize_fan_in_fan_out_value(value)
    if normalized is not None:
        return OrientationResolution(fan_in_fan_out=normalized, source="config")

    if isinstance(value, dict):
        for candidate in (module_path, _module_type(module_path)):
            if candidate in value:
                candidate_value = _normalize_fan_in_fan_out_value(value[candidate])
                if candidate_value is not None:
                    return OrientationResolution(
                        fan_in_fan_out=candidate_value,
                        source=f"config[{candidate}]",
                    )
    return None


def _rank_for_module(config: dict[str, Any], module_path: str) -> int:
    pattern = config.get("rank_pattern") or {}
    for candidate in (module_path, _module_type(module_path)):
        if candidate in pattern:
            return int(pattern[candidate])
    return int(config["r"])


def _alpha_for_module(config: dict[str, Any], module_path: str) -> float:
    pattern = config.get("alpha_pattern") or {}
    for candidate in (module_path, _module_type(module_path)):
        if candidate in pattern:
            return float(pattern[candidate])
    return float(config["lora_alpha"])


def _infer_orientation_from_shapes(
    module_path: str,
    base_index: ShardedSafetensorIndex,
    state: dict[str, torch.Tensor],
) -> tuple[bool | None, str]:
    base_shape = base_index.get_shape(f"{module_path}.weight")
    prefix = f"base_model.model.{module_path}"
    a_shape = tuple(state[f"{prefix}.lora_A.weight"].shape)
    b_shape = tuple(state[f"{prefix}.lora_B.weight"].shape)
    raw_shape = (b_shape[0], a_shape[1])
    transposed_shape = (raw_shape[1], raw_shape[0])

    raw_matches = raw_shape == base_shape
    transposed_matches = transposed_shape == base_shape

    if raw_matches and not transposed_matches:
        return False, "shape_match"
    if transposed_matches and not raw_matches:
        return True, "shape_match"
    if raw_matches and transposed_matches:
        return None, "square_shape_ambiguous"
    return None, f"adapter shape {raw_shape} does not match base shape {base_shape}"


def resolve_run_orientations(
    run: RunSpec,
    base_index: ShardedSafetensorIndex,
    module_paths: Iterable[str],
) -> dict[str, OrientationResolution]:
    if run.kind not in {"lora", "dora"}:
        return {}

    state = _load_run_state(run)
    resolutions: dict[str, OrientationResolution] = {}
    inferred_values: set[bool] = set()
    ambiguous_modules: list[str] = []

    for module_path in module_paths:
        config_resolution = _resolve_config_orientation(run.config, module_path)
        if config_resolution is not None:
            resolutions[module_path] = config_resolution
            inferred_values.add(config_resolution.fan_in_fan_out)
            continue

        inferred, source = _infer_orientation_from_shapes(module_path, base_index, state)
        if inferred is None:
            if source == "square_shape_ambiguous":
                ambiguous_modules.append(module_path)
                continue
            raise AuditError(
                f"Could not resolve fan_in_fan_out for {run.label} {module_path}: {source}"
            )
        resolutions[module_path] = OrientationResolution(fan_in_fan_out=inferred, source=source)
        inferred_values.add(inferred)

    if len(inferred_values) > 1:
        raise AuditError(
            f"Conflicting fan_in_fan_out resolutions for {run.label}: {sorted(inferred_values)}"
        )

    if ambiguous_modules:
        if len(inferred_values) != 1:
            raise AuditError(
                f"Could not resolve square-module fan_in_fan_out for {run.label}: "
                f"{', '.join(sorted(ambiguous_modules))}"
            )
        inferred_value = next(iter(inferred_values))
        for module_path in ambiguous_modules:
            resolutions[module_path] = OrientationResolution(
                fan_in_fan_out=inferred_value,
                source="inferred_from_non_square_shape",
            )

    return resolutions


def audit_runs(
    runs: list[RunSpec],
    base_index: ShardedSafetensorIndex,
    expected_modules: list[str],
    module_policy: str = "strict",
) -> dict[str, Any]:
    if module_policy not in {"strict", "actual"}:
        raise SpectralEvalError(f"Unsupported module policy: {module_policy}")

    expected_set = set(expected_modules)
    expected_type_counts = Counter(_module_type(module) for module in expected_modules)
    report_runs: list[dict[str, Any]] = []
    has_errors = False

    for run in runs:
        realized_modules = extract_realized_modules(run)
        run.realized_modules = realized_modules
        run.realized_module_types = sorted({_module_type(module) for module in realized_modules})

        realized_set = set(realized_modules)
        run.missing_modules = sorted(expected_set - realized_set)
        run.extra_modules = sorted(realized_set - expected_set)
        run.missing_module_types = sorted({_module_type(module) for module in run.missing_modules})
        run.extra_module_types = sorted({_module_type(module) for module in run.extra_modules})

        realized_type_counts = Counter(_module_type(module) for module in realized_modules)

        if module_policy == "strict" and run.kind != "fft" and run.missing_modules:
            run.errors.append(
                f"Run is missing expected modules: {', '.join(run.missing_module_types)}"
            )

        if run.kind in {"lora", "dora"}:
            try:
                run.orientation = resolve_run_orientations(run, base_index, realized_modules)
            except AuditError as exc:
                run.errors.append(str(exc))

        report_runs.append(
            {
                "label": run.label,
                "kind": run.kind,
                "artifact_path": str(run.artifact_path),
                "expected_module_type_counts": dict(expected_type_counts),
                "realized_module_type_counts": dict(realized_type_counts),
                "missing_module_types": run.missing_module_types,
                "extra_module_types": run.extra_module_types,
                "warnings": list(run.warnings),
                "errors": list(run.errors),
                "orientation": {
                    module: asdict(resolution) for module, resolution in run.orientation.items()
                },
            }
        )

        has_errors = has_errors or bool(run.errors)

    return {
        "module_policy": module_policy,
        "expected_target_modules": list(EXPECTED_TARGET_MODULES),
        "expected_module_count": len(expected_modules),
        "runs": report_runs,
        "has_errors": has_errors,
    }


def write_audit_report(audit_report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    txt_path = output_dir / "audit.txt"

    json_path.write_text(json.dumps(audit_report, indent=2, sort_keys=True))

    lines = [
        f"Module policy: {audit_report['module_policy']}",
        f"Expected module types: {', '.join(audit_report['expected_target_modules'])}",
        "",
    ]
    for run in audit_report["runs"]:
        lines.append(f"{run['label']} ({run['kind']})")
        lines.append(
            "  realized module counts: "
            + ", ".join(
                f"{name}={count}" for name, count in sorted(run["realized_module_type_counts"].items())
            )
        )
        if run["missing_module_types"]:
            lines.append("  missing module types: " + ", ".join(run["missing_module_types"]))
        if run["extra_module_types"]:
            lines.append("  extra module types: " + ", ".join(run["extra_module_types"]))
        if run["errors"]:
            lines.append("  errors:")
            for error in run["errors"]:
                lines.append(f"    - {error}")
        if run["warnings"]:
            lines.append("  warnings:")
            for warning in run["warnings"]:
                lines.append(f"    - {warning}")
        orientation_sources = Counter(
            resolution["source"] for resolution in run["orientation"].values()
        )
        if orientation_sources:
            lines.append(
                "  orientation sources: "
                + ", ".join(f"{name}={count}" for name, count in sorted(orientation_sources.items()))
            )
        lines.append("")
    txt_path.write_text("\n".join(lines))
    return json_path, txt_path


@lru_cache(maxsize=16)
def _quanta_eval_expression(
    d: int,
    per_dim_features: tuple[int, ...],
    per_dim_features2: tuple[int, ...],
) -> Any:
    current_symbols = list(range(d))
    initial_symbols = list(current_symbols)
    equation = ""

    for dim1, dim2 in itertools.combinations(range(-1, -d - 1, -1), 2):
        symbol1 = current_symbols[dim1]
        symbol2 = current_symbols[dim2]
        symbol3 = symbol1 + d
        symbol4 = symbol2 + d
        equation += (
            ","
            + oe.get_symbol(symbol4)
            + oe.get_symbol(symbol3)
            + oe.get_symbol(symbol2)
            + oe.get_symbol(symbol1)
        )
        current_symbols[dim1] = symbol3
        current_symbols[dim2] = symbol4

    equation += "->"
    equation += "".join(oe.get_symbol(index) for index in current_symbols)
    equation += "".join(oe.get_symbol(index) for index in initial_symbols)
    equation = equation[1:]

    shapes = []
    for dim1, dim2 in itertools.combinations(range(-1, -d - 1, -1), 2):
        shapes.append(
            (
                per_dim_features2[dim2],
                per_dim_features2[dim1],
                per_dim_features[dim2],
                per_dim_features[dim1],
            )
        )

    optimize = "optimal" if d <= 4 else "branch-all" if d <= 5 else "branch-2" if d <= 7 else "auto"
    return oe.contract_expression(equation, *shapes, optimize=optimize)


def reconstruct_quanta_dense(
    weights: list[torch.Tensor],
    *,
    d: int,
    per_dim_features: list[int],
    per_dim_features2: list[int] | None = None,
) -> torch.Tensor:
    per_dim_features2 = per_dim_features2 or per_dim_features
    expression = _quanta_eval_expression(
        d=d,
        per_dim_features=tuple(per_dim_features),
        per_dim_features2=tuple(per_dim_features2),
    )
    dense = expression(*[weight.float() for weight in weights])
    return dense.reshape(math.prod(per_dim_features2), math.prod(per_dim_features))


def reconstruct_lora_delta(
    run: RunSpec,
    module_path: str,
    base_index: ShardedSafetensorIndex,
) -> torch.Tensor:
    state = _load_run_state(run)
    prefix = f"base_model.model.{module_path}"
    a = state[f"{prefix}.lora_A.weight"].float()
    b = state[f"{prefix}.lora_B.weight"].float()
    r = _rank_for_module(run.config, module_path)
    alpha = _alpha_for_module(run.config, module_path)
    scaling = alpha / r
    raw_delta = (b @ a) * scaling

    if module_path not in run.orientation:
        raise SpectralEvalError(f"Missing orientation resolution for {run.label} {module_path}")
    fan_in_fan_out = run.orientation[module_path].fan_in_fan_out
    delta = _transpose_if_needed(raw_delta, fan_in_fan_out)
    base_shape = base_index.get_shape(f"{module_path}.weight")
    if tuple(delta.shape) != base_shape:
        raise SpectralEvalError(
            f"LoRA delta shape mismatch for {run.label} {module_path}: "
            f"{tuple(delta.shape)} vs base {base_shape}"
        )
    return delta


def reconstruct_dora_delta(
    run: RunSpec,
    module_path: str,
    base_index: ShardedSafetensorIndex,
) -> torch.Tensor:
    base_weight = base_index.get_tensor(f"{module_path}.weight").float()
    delta = reconstruct_lora_delta(run, module_path, base_index).float()
    state = _load_run_state(run)
    prefix = f"base_model.model.{module_path}"
    magnitude = state[f"{prefix}.lora_magnitude_vector"].float()
    fan_in_fan_out = run.orientation[module_path].fan_in_fan_out

    weight_for_norm = _transpose_if_needed(base_weight, fan_in_fan_out)
    delta_for_norm = _transpose_if_needed(delta, fan_in_fan_out)
    weight_norm = torch.linalg.norm(weight_for_norm + delta_for_norm, dim=1)
    dora_factor = magnitude / weight_norm
    broadcast = dora_factor.view(-1, 1)
    broadcast = _transpose_if_needed(broadcast, fan_in_fan_out)
    merged_weight = broadcast * (base_weight + delta)
    return merged_weight - base_weight


def reconstruct_quanta_delta(
    run: RunSpec,
    module_path: str,
    base_index: ShardedSafetensorIndex,
) -> DeltaMatrices:
    state = _load_run_state(run)
    d = int(run.config["d"])
    per_dim_features = list(run.config["per_dim_features"])
    per_dim_features2 = run.config.get("per_dim_features2") or per_dim_features
    prefix = f"base_model.model.{module_path}"

    key_pairs = list(itertools.combinations(range(-1, -d - 1, -1), 2))
    weights1 = [state[f"{prefix}.quanta_weights.{dim1} {dim2}"] for dim1, dim2 in key_pairs]
    weights2 = [state[f"{prefix}.quanta_weights2.{dim1} {dim2}"] for dim1, dim2 in key_pairs]
    dense1 = reconstruct_quanta_dense(
        weights1,
        d=d,
        per_dim_features=per_dim_features,
        per_dim_features2=per_dim_features2,
    )
    dense2 = reconstruct_quanta_dense(
        weights2,
        d=d,
        per_dim_features=per_dim_features,
        per_dim_features2=per_dim_features2,
    )
    support_delta = dense1 - dense2

    base_shape = base_index.get_shape(f"{module_path}.weight")
    out_features, in_features = base_shape
    padded = torch.nn.functional.pad(
        support_delta,
        (0, in_features - support_delta.shape[1], 0, out_features - support_delta.shape[0]),
        value=0.0,
    )
    fan_in_fan_out = bool(run.config.get("fan_in_fan_out", False))
    return DeltaMatrices(
        spectral=_transpose_if_needed(support_delta, fan_in_fan_out),
        full=_transpose_if_needed(padded, fan_in_fan_out),
    )


def reconstruct_fft_delta(
    run: RunSpec,
    module_path: str,
    base_index: ShardedSafetensorIndex,
) -> torch.Tensor:
    base_weight = base_index.get_tensor(f"{module_path}.weight").float()
    full_store = SingleSafetensorStore(run.artifact_path)
    tuned_weight = full_store.get_tensor(f"{module_path}.weight").float()
    return tuned_weight - base_weight


def reconstruct_delta(
    run: RunSpec,
    module_path: str,
    base_index: ShardedSafetensorIndex,
) -> DeltaMatrices:
    if run.kind == "lora":
        delta = reconstruct_lora_delta(run, module_path, base_index)
        return DeltaMatrices(spectral=delta, full=delta)
    if run.kind == "dora":
        delta = reconstruct_dora_delta(run, module_path, base_index)
        return DeltaMatrices(spectral=delta, full=delta)
    if run.kind == "quanta":
        return reconstruct_quanta_delta(run, module_path, base_index)
    if run.kind == "fft":
        delta = reconstruct_fft_delta(run, module_path, base_index)
        return DeltaMatrices(spectral=delta, full=delta)
    raise SpectralEvalError(f"Unsupported run kind: {run.kind}")


def singular_values(delta: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    matrix = delta.to(device=device, dtype=torch.float32)
    values = torch.linalg.svdvals(matrix)
    return values.cpu()


def effective_rank_from_singular_values(values: torch.Tensor, threshold: float = 1e-12) -> float:
    filtered = values[values > threshold]
    if filtered.numel() == 0:
        return 0.0
    probs = (filtered.square()) / filtered.square().sum()
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return float(torch.exp(entropy))


def normalized_decay(values: torch.Tensor, top_k: int) -> list[float]:
    if values.numel() == 0:
        return []
    head = values[:top_k]
    if float(head[0]) == 0.0:
        return [0.0 for _ in range(len(head))]
    return [float(value / head[0]) for value in head]


def module_layer_type_breakdown(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[f"{record['label']}::{record['module_type']}"].append(record["effective_rank"])

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for compound_key, values in grouped.items():
        label, module_type = compound_key.split("::", 1)
        result[label][module_type] = float(sum(values) / len(values))
    return dict(result)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = y.T @ x
    x_cov = x.T @ x
    y_cov = y.T @ y
    numerator = torch.linalg.norm(cross, ord="fro").square()
    denominator = torch.linalg.norm(x_cov, ord="fro") * torch.linalg.norm(y_cov, ord="fro")
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator / denominator)


def analyze_runs(
    runs: list[RunSpec],
    *,
    base_model_dir: str | Path = DEFAULT_BASE_MODEL_DIR,
    module_policy: str = "strict",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_k: int = 64,
    device: str = "cpu",
    representative_layer: str = DEFAULT_REPRESENTATIVE_LAYER,
    include_cka: bool = False,
) -> dict[str, Any]:
    base_index = ShardedSafetensorIndex(base_model_dir)
    expected_modules = collect_expected_modules(base_index)
    audit_report = audit_runs(runs, base_index, expected_modules, module_policy=module_policy)
    audit_json_path, audit_txt_path = write_audit_report(audit_report, output_dir)

    if audit_report["has_errors"] and module_policy == "strict":
        raise AuditError(
            f"Strict audit failed. See {audit_json_path} and {audit_txt_path}",
            report=audit_report,
        )

    if module_policy == "strict":
        analysis_modules = expected_modules
    else:
        analysis_modules = sorted(set.intersection(*(set(run.realized_modules) for run in runs)))

    if not analysis_modules:
        raise SpectralEvalError("No modules available for analysis")

    per_layer_records: list[dict[str, Any]] = []
    representative_decays: dict[str, list[float]] = {}
    summary_by_label: dict[str, dict[str, Any]] = {}

    for run in runs:
        run_records = []
        for module_path in analysis_modules:
            if module_path not in run.realized_modules and module_policy == "actual":
                continue
            delta = reconstruct_delta(run, module_path, base_index)
            values = singular_values(delta.spectral, device=device)
            record = {
                "label": run.label,
                "kind": run.kind,
                "module_path": module_path,
                "module_type": _module_type(module_path),
                "effective_rank": effective_rank_from_singular_values(values),
                "decay": normalized_decay(values, top_k=top_k),
            }
            per_layer_records.append(record)
            run_records.append(record)
            if module_path == representative_layer:
                representative_decays[run.label] = record["decay"]

        if not run_records:
            raise SpectralEvalError(f"No analysis records produced for {run.label}")
        summary_by_label[run.label] = {
            "kind": run.kind,
            "score": run.score,
            "mean_effective_rank": float(
                sum(record["effective_rank"] for record in run_records) / len(run_records)
            ),
            "num_modules": len(run_records),
        }

    cka_summary: dict[str, dict[str, float]] = {}
    if include_cka:
        dora_runs = [run for run in runs if run.kind == "dora"]
        quanta_runs = [run for run in runs if run.kind == "quanta"]
        for dora_run in dora_runs:
            for quanta_run in quanta_runs:
                pair_key = f"{dora_run.label} vs {quanta_run.label}"
                pair_scores: dict[str, float] = {}
                shared_modules = sorted(set(analysis_modules) & set(dora_run.realized_modules) & set(quanta_run.realized_modules))
                for module_path in shared_modules:
                    dora_delta = reconstruct_delta(dora_run, module_path, base_index).full
                    quanta_delta = reconstruct_delta(quanta_run, module_path, base_index).full
                    pair_scores[module_path] = linear_cka(dora_delta, quanta_delta)
                cka_summary[pair_key] = pair_scores

    layer_breakdown = module_layer_type_breakdown(per_layer_records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_layer_metrics.json").write_text(json.dumps(per_layer_records, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary_by_label, indent=2, sort_keys=True))
    (output_dir / "representative_decays.json").write_text(
        json.dumps(representative_decays, indent=2, sort_keys=True)
    )
    (output_dir / "layer_type_breakdown.json").write_text(
        json.dumps(layer_breakdown, indent=2, sort_keys=True)
    )
    if include_cka:
        (output_dir / "cka.json").write_text(json.dumps(cka_summary, indent=2, sort_keys=True))

    return {
        "audit": audit_report,
        "analysis_modules": analysis_modules,
        "per_layer_metrics": per_layer_records,
        "summary": summary_by_label,
        "representative_decays": representative_decays,
        "layer_type_breakdown": layer_breakdown,
        "cka": cka_summary,
        "audit_json_path": str(audit_json_path),
        "audit_txt_path": str(audit_txt_path),
    }


def plot_analysis(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
    representative_layer: str,
    label_aliases: dict[str, str] | None = None,
    score_zoom_min: float | None = None,
    offscale_labels: list[str] | None = None,
    decay_top_k: int | None = None,
    score_axis_label: str = "DeepMath score",
    scatter_title: str = "Effective rank vs score",
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    paths: list[str] = []
    label_aliases = label_aliases or {}
    offscale_label_set = set(offscale_labels or [])

    def display_label(label: str) -> str:
        return label_aliases.get(label, label)

    def save_figure(fig: Any, stem: str) -> list[str]:
        saved_paths: list[str] = []
        png_path = output_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        saved_paths.append(str(png_path))
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        saved_paths.append(str(pdf_path))
        return saved_paths

    ordered_labels: list[str] = []
    for source in (result.get("summary", {}), result.get("representative_decays", {}), result.get("layer_type_breakdown", {})):
        for label in source:
            if label not in ordered_labels:
                ordered_labels.append(label)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    color_map = {
        label: color_cycle[index % len(color_cycle)]
        for index, label in enumerate(ordered_labels)
    }

    decays = result["representative_decays"]
    if decays:
        fig = plt.figure(figsize=(10, 6))
        max_decay_points = decay_top_k or max((len(decay) for decay in decays.values()), default=1)
        denominator = max(1, max_decay_points - 1)
        for label, decay in decays.items():
            if decay:
                x_positions = [index / denominator for index in range(len(decay))]
                plt.plot(
                    x_positions,
                    decay,
                    label=display_label(label),
                    color=color_map.get(label),
                )
        plt.yscale("log")
        plt.xlim(0.0, 1.0)
        plt.xlabel(f"Normalized singular value index (top-{max_decay_points})")
        plt.ylabel("Normalized singular value")
        plt.title(f"Singular value decay: {representative_layer}")
        handles, labels = plt.gca().get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        plt.legend(deduped.values(), deduped.keys())
        plt.tight_layout()
        paths.extend(save_figure(fig, "representative_decay"))
        plt.close(fig)

    summary = result["summary"]
    if summary:
        fig, ax = plt.subplots(figsize=(11.75, 6.5))
        sidebar_left = 0.81
        for label, payload in summary.items():
            ax.scatter(
                payload["mean_effective_rank"],
                payload["score"],
                s=90,
                label=display_label(label),
                color=color_map.get(label),
            )
        if score_zoom_min is not None:
            visible_scores = [
                payload["score"]
                for label, payload in summary.items()
                if label not in offscale_label_set and payload["score"] >= score_zoom_min
            ]
            if visible_scores:
                ax.set_ylim(score_zoom_min, max(visible_scores) + 0.03)
        ax.set_xlabel("Mean effective rank")
        ax.set_ylabel(score_axis_label)
        ax.set_title(scatter_title)

        handles, labels = ax.get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        legend = ax.legend(
            deduped.values(),
            deduped.keys(),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
        )

        if offscale_label_set:
            offscale_payloads = [
                (label, payload)
                for label, payload in summary.items()
                if label in offscale_label_set
            ]
            if offscale_payloads:
                inset = fig.add_axes([sidebar_left, 0.14, 0.18, 0.23])
                x_values = [payload["mean_effective_rank"] for _, payload in offscale_payloads]
                y_values = [payload["score"] for _, payload in offscale_payloads]
                for label, payload in offscale_payloads:
                    inset.scatter(
                        payload["mean_effective_rank"],
                        payload["score"],
                        s=55,
                        color=color_map.get(label),
                    )
                x_min = min(x_values)
                x_max = max(x_values)
                x_pad = max(1.0, (x_max - x_min) * 3.0)
                inset.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
                inset.set_ylim(0.0, max(y_values) * 1.35)
                inset.set_title("Off-scale", fontsize=8)
                inset.tick_params(labelsize=7)
                inset.grid(alpha=0.2)
        fig.subplots_adjust(right=0.72)
        paths.extend(save_figure(fig, "effective_rank_vs_score"))
        plt.close(fig)

    breakdown = result["layer_type_breakdown"]
    if breakdown:
        labels = list(breakdown)
        module_types = sorted({module_type for payload in breakdown.values() for module_type in payload})
        x = torch.arange(len(labels), dtype=torch.float32)
        width = 0.8 / max(1, len(module_types))
        fig = plt.figure(figsize=(12, 6))
        for index, module_type in enumerate(module_types):
            values = [breakdown[label].get(module_type, float("nan")) for label in labels]
            offset = (index - (len(module_types) - 1) / 2.0) * width
            plt.bar((x + offset).tolist(), values, width=width, label=module_type)
        plt.xticks(x.tolist(), [display_label(label) for label in labels], rotation=45, ha="right")
        plt.ylabel("Mean effective rank")
        plt.title("Layer-type breakdown")
        plt.legend()
        plt.tight_layout()
        paths.extend(save_figure(fig, "layer_type_breakdown"))
        plt.close(fig)

    return paths
