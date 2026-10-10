"""
Frozen Base Model SVD Analysis
================================
Analyzes the spectral structure of *frozen* base model weights across three
models at different reported pre-training scales, as a diagnostic related to
the Model Maturity Hypothesis. This is not a causal test of pre-training scale.

Integrates with the existing deepmath_spectral pipeline:
  - Imports effective_rank_from_singular_values() and normalized_decay()
    from deepmath_spectral so that base-model PR values and fine-tuned
    checkpoint effective rank values sit on the same scale and are directly
    comparable in a single table.
  - Decay curve outputs are commensurable with normalized_decay() curves
    produced by deepmath_adapter_spectrum.py and deepmath_fft_spectrum.py.

Models (ordered by pre-training scale):
  apertus_8b    ./apertus-8b               ~15T tokens   (mid maturity)
  llama31_8b    ./llama-3.1-8b             ~15.6T tokens (mid maturity)
  qwen3_8b      ./qwen3-8b                ~36T tokens   (high maturity)

Expected result if maturity hypothesis holds:
  Effective rank (PR) and spectral decay AUC increase monotonically
  from Apertus -> Llama-3.1 -> Qwen3 in the frozen base weights,
  predicting adapter behavior before any fine-tuning occurs.

Requirements:
    Use the supported container/runtime environment documented in README.md.
    (deepmath_spectral must be importable; direct script execution adds src/.)

Usage:
    # Full run (resumes from cached results for completed models):
    python src/frozen_svd_analysis.py

    # Force re-run everything:
    python src/frozen_svd_analysis.py --no-resume

    # Only aggregate existing CSVs and re-generate plots:
    python src/frozen_svd_analysis.py --plot-only

Compute notes:
  - GPU not required. All ops run on CPU.
  - Peak RAM per model: ~40GB (bfloat16 load) + transient float32 per matrix.
  - SVD is computed one matrix at a time; float32 arrays are freed immediately.
  - Phase 1 (full metric sweep) and Phase 2 (spectra for decay curves) each
    load the model once. Total wall time: ~30-60 min per model on CPU.
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Import shared metrics from existing pipeline.
# Guarantees numerical identity with fine-tuned checkpoint analyses.
#
# effective_rank_from_singular_values(values: Tensor, threshold=1e-12) -> float
#   Filters sv <= threshold, computes PR = exp(entropy of normalized σ²).
#
# normalized_decay(values: Tensor, top_k: int) -> list[float]
#   Returns [σ_i / σ_1 for i in range(top_k)].
# ---------------------------------------------------------------------------
try:
    from deepmath_spectral import effective_rank_from_singular_values, normalized_decay
except ImportError:
    raise ImportError(
        "deepmath_spectral not found. Run from the repository root or add src/ to PYTHONPATH.\n"
        "  python src/frozen_svd_analysis.py"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "apertus_8b":  "./apertus-8b",       # ~15T tokens   (mid maturity)
    "llama31_8b":  "./llama-3.1-8b",     # ~15.6T tokens (mid maturity)
    "qwen3_8b":    "./qwen3-8b",         # ~36T tokens   (high maturity)
}

PRETRAINING_TOKENS = {
    "apertus_8b":  "~15T",
    "llama31_8b":  "~15.6T",
    "qwen3_8b":    "~36T",
}

MODEL_ORDER = ["apertus_8b", "llama31_8b", "qwen3_8b"]

# Linear layers targeted by adapters. Excludes embedding and LM head,
# which are not adapter targets and have qualitatively different rank dynamics.
EXCLUDE_KEYWORDS = ["embed_tokens", "lm_head"]

# Top-k singular values for decay curves, matching the paper summary.
TOP_K = 64

# Number of randomly sampled layers for decay curve storage (Phase 2).
# Full metric sweep (Phase 1) processes every layer.
MAX_LAYERS_FOR_SPECTRA = 50

OUTPUT_DIR = Path("./svd_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot colors — consistent with deepmath_metric_compare.py palette
COLORS = {
    "apertus_8b":  "#1D9E75",   # teal
    "llama31_8b":  "#185FA5",   # blue
    "qwen3_8b":    "#D85A30",   # coral
}
LABELS = {
    "apertus_8b":  "Apertus-8B (~15T tokens)",
    "llama31_8b":  "Llama-3.1-8B (~15.6T tokens)",
    "qwen3_8b":    "Qwen3-8B (~36T tokens)",
}


# ---------------------------------------------------------------------------
# Stable rank (supplementary metric; not in deepmath_spectral)
# ---------------------------------------------------------------------------

def stable_rank(singular_values: torch.Tensor) -> float:
    """
    Stable rank = ||W||_F^2 / ||W||_2^2 = sum(sigma_i^2) / sigma_max^2.
    Ranges from 1 (rank-1 matrix) to min(d_out, d_in) (uniform spectrum).
    Robust to near-zero singular values; useful as a sanity check on PR.
    Operates on a torch.Tensor to match deepmath_spectral conventions.
    """
    s2 = singular_values.square()
    return float(s2.sum() / s2.max())


# ---------------------------------------------------------------------------
# Layer-type tagging (covers Llama, Qwen, and Apertus naming conventions)
# ---------------------------------------------------------------------------

def tag_layer_type(layer_name: str) -> str:
    n = layer_name.lower()
    # Attention QKV projections
    if any(x in n for x in ["q_proj", "k_proj", "v_proj", "att_proj", "qkv_proj"]):
        return "attn_qkv"
    # Attention output projection
    if any(x in n for x in ["o_proj", "attn_out", "attn_output"]):
        return "attn_out"
    # MLP gate / up projections
    if any(x in n for x in ["gate_proj", "up_proj", "ff_proj", "w1", "w3"]):
        return "mlp_gate_up"
    # MLP down projection
    if any(x in n for x in ["down_proj", "ff_out", "w2"]):
        return "mlp_down"
    return "other"


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def has_existing_results(model_key: str) -> bool:
    """Check if both CSV and spectra .npy exist for a model."""
    csv_path = OUTPUT_DIR / f"{model_key}_svd_metrics.csv"
    npy_path = OUTPUT_DIR / f"{model_key}_spectra.npy"
    return csv_path.exists() and npy_path.exists()


def load_existing_results(model_key: str) -> pd.DataFrame:
    """Load previously saved per-layer metrics CSV."""
    csv_path = OUTPUT_DIR / f"{model_key}_svd_metrics.csv"
    print(f"  ✓ Loading cached results from {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Phase 1: Full metric sweep
# ---------------------------------------------------------------------------

def analyze_model(model_key: str, local_path: str) -> pd.DataFrame:
    """
    Load base model, compute SVD metrics for every linear layer,
    return a DataFrame with one row per layer.

    Uses effective_rank_from_singular_values() from deepmath_spectral
    so PR values are numerically identical to fine-tuned checkpoint metrics.
    """
    print(f"\n{'='*60}")
    print(f"[Phase 1] {model_key}  ({local_path})")
    print(f"{'='*60}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Count target layers for progress reporting
    all_linear = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and not any(kw in name for kw in EXCLUDE_KEYWORDS)
    ]
    total_linear = len(all_linear)
    print(f"  Found {total_linear} linear layers to analyze")

    records = []

    for idx, (name, module) in enumerate(all_linear):
        W = module.weight.data          # (d_out, d_in), bfloat16
        d_out, d_in = W.shape

        # Cast to float32 for SVD numerical accuracy, consistent with
        # deepmath_spectral which also operates in float32.
        W_f32 = W.float()

        # svdvals only — no U/V needed, ~3x faster than full SVD.
        sv = torch.linalg.svdvals(W_f32)   # descending, float32 tensor

        # --- Metrics (using imported functions for commensurability) ---

        # effective_rank_from_singular_values filters sv <= 1e-12 before
        # computing probabilities, matching the fine-tuned checkpoint pipeline
        # exactly. This resolves the numerical inconsistency in the previous
        # version which clipped probabilities after computing them.
        pr = effective_rank_from_singular_values(sv)

        # normalized_decay returns list[float] of sigma_i / sigma_1 for
        # top_k values, identical to the decay curves in
        # deepmath_adapter_spectrum.py.
        decay_list = normalized_decay(sv, top_k=TOP_K)   # list[float]
        decay_auc = float(np.mean(decay_list)) if decay_list else 0.0

        sr = stable_rank(sv)

        records.append({
            "model":               model_key,
            "layer_name":          name,
            "layer_type":          tag_layer_type(name),
            "d_out":               d_out,
            "d_in":                d_in,
            "rank_upper_bound":    min(d_out, d_in),
            "stable_rank":         sr,
            "participation_ratio": pr,              # exp(H), same measure as the paper
            "spectral_decay_auc":  decay_auc,       # mean of normalized_decay curve
            "sigma_max":           float(sv[0]),
            "nuclear_norm":        float(sv.sum()),
            "frobenius_norm":      float(sv.square().sum().sqrt()),
        })

        # Progress logging
        if (idx + 1) % 25 == 0 or idx == total_linear - 1:
            print(f"  [{idx+1}/{total_linear}] {name} ({d_out}×{d_in}) — "
                  f"SR={sr:.1f}, PR={pr:.1f}, AUC={decay_auc:.4f}")

        del W_f32, sv, decay_list
        gc.collect()

    del model
    gc.collect()

    df = pd.DataFrame(records)
    out_path = OUTPUT_DIR / f"{model_key}_svd_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} layer records -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# Phase 2: Spectral decay curve storage (sampled layers)
# ---------------------------------------------------------------------------

def extract_spectra(model_key: str, local_path: str) -> np.ndarray:
    """
    Load base model, sample up to MAX_LAYERS_FOR_SPECTRA linear layers,
    store their normalized top-TOP_K singular value curves.

    Output shape: (n_sampled_layers, TOP_K).
    Saved to {model_key}_spectra.npy for use in plot_mean_decay_curves().

    Uses normalized_decay() from deepmath_spectral, producing curves directly
    comparable to those from deepmath_adapter_spectrum.py and
    deepmath_fft_spectrum.py — enabling a unified base+fine-tuned decay figure.
    """
    print(f"\n[Phase 2] {model_key} — extracting decay curves")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    all_linear = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and not any(kw in name for kw in EXCLUDE_KEYWORDS)
    ]

    rng = np.random.default_rng(seed=42)
    n_sample = min(MAX_LAYERS_FOR_SPECTRA, len(all_linear))
    indices = sorted(rng.choice(len(all_linear), size=n_sample, replace=False))
    sampled = [all_linear[i] for i in indices]

    spectra = []
    for name, module in sampled:
        sv = torch.linalg.svdvals(module.weight.data.float())
        # normalized_decay returns list[float]; pad if smaller than TOP_K
        curve_list = normalized_decay(sv, top_k=TOP_K)
        if len(curve_list) < TOP_K:
            curve_list.extend([0.0] * (TOP_K - len(curve_list)))
        curve = np.array(curve_list)
        spectra.append(curve)
        del sv

    del model
    gc.collect()

    spectra_arr = np.stack(spectra)   # (n_sampled, TOP_K)
    out_path = OUTPUT_DIR / f"{model_key}_spectra.npy"
    np.save(out_path, spectra_arr)
    print(f"  Saved {n_sample} decay curves -> {out_path}")
    return spectra_arr


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_dfs: list) -> tuple:
    """
    Compute summary statistics per model, suitable for the paper table.

    Returns:
        summary:  Per-model and per-layer-type summary stats.
        combined: Concatenated per-layer DataFrame with layer_type column.
    """
    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure layer_type is present (may be missing in old cached CSVs)
    if "layer_type" not in combined.columns:
        combined["layer_type"] = combined["layer_name"].apply(tag_layer_type)

    # Check for untagged layers
    other_count = (combined["layer_type"] == "other").sum()
    if other_count > 0:
        other_names = combined.loc[combined["layer_type"] == "other", "layer_name"].unique()
        print(f"\n⚠  {other_count} layers tagged as 'other'. Sample names:")
        for name in other_names[:10]:
            print(f"     {name}")

    # Per-model, per-layer-type breakdown
    by_type = (
        combined
        .groupby(["model", "layer_type"])
        .agg(
            n_layers=("participation_ratio", "count"),
            stable_rank_mean=("stable_rank", "mean"),
            stable_rank_std=("stable_rank", "std"),
            pr_mean=("participation_ratio", "mean"),
            pr_std=("participation_ratio", "std"),
            decay_auc_mean=("spectral_decay_auc", "mean"),
            decay_auc_std=("spectral_decay_auc", "std"),
        )
        .reset_index()
    )

    # Model-level aggregate (all layers pooled) — the primary table for the paper
    by_model = (
        combined
        .groupby("model")
        .agg(
            n_layers=("participation_ratio", "count"),
            stable_rank_mean=("stable_rank", "mean"),
            stable_rank_std=("stable_rank", "std"),
            pr_mean=("participation_ratio", "mean"),
            pr_std=("participation_ratio", "std"),
            decay_auc_mean=("spectral_decay_auc", "mean"),
            decay_auc_std=("spectral_decay_auc", "std"),
        )
        .reset_index()
    )
    by_model["layer_type"] = "ALL"

    summary = pd.concat([by_type, by_model], ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)
    return summary, combined


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pr_by_layer_type(combined: pd.DataFrame):
    """
    Box plots of participation ratio per layer type per model.
    If the maturity hypothesis holds, boxes should increase left-to-right
    (Apertus -> Llama-3.1 -> Qwen3) in every subplot.
    """
    layer_types = ["attn_qkv", "attn_out", "mlp_gate_up", "mlp_down"]
    palette = [COLORS[m] for m in MODEL_ORDER]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax, lt in zip(axes, layer_types):
        subset = combined[combined["layer_type"] == lt]
        sns.boxplot(
            data=subset, x="model", y="participation_ratio",
            order=MODEL_ORDER, ax=ax, palette=palette,
        )
        ax.set_title(lt, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Participation ratio (PR)" if lt == layer_types[0] else "")
        ax.set_xticklabels([LABELS[m].split(" ")[0] for m in MODEL_ORDER], rotation=15)

    plt.suptitle(
        "Effective rank of frozen base model weights by layer type\n"
        "Monotonic increase left→right supports the Model Maturity Effect",
        fontsize=12,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "pr_by_layer_type.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_mean_decay_curves():
    """
    Mean ± std normalized singular value decay curves for all three models.
    Uses *_spectra.npy written by extract_spectra().

    Directly comparable to fine-tuned checkpoint decay curves from
    deepmath_adapter_spectrum.py — can be overlaid in a unified figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, TOP_K + 1)

    for key in MODEL_ORDER:
        path = OUTPUT_DIR / f"{key}_spectra.npy"
        if not path.exists():
            print(f"  Spectra not found for {key}, skipping decay plot.")
            continue
        spectra = np.load(path)
        mean_c = spectra.mean(axis=0)
        std_c = spectra.std(axis=0)
        ax.plot(x, mean_c, label=LABELS[key], color=COLORS[key], linewidth=2)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                        alpha=0.15, color=COLORS[key])

    ax.set_xlabel("Singular value rank index", fontsize=12)
    ax.set_ylabel("Normalized singular value (σᵢ / σ₁)", fontsize=12)
    ax.set_title(
        "Singular value decay — frozen base model weights\n"
        "Flatter curve = more distributed representations",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(1, TOP_K)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = OUTPUT_DIR / "mean_decay_curves.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_paper_table(summary: pd.DataFrame):
    rows = (
        summary[summary["layer_type"] == "ALL"]
        .set_index("model")
        .loc[MODEL_ORDER]
    )
    print("\n" + "=" * 72)
    print("FROZEN BASE MODEL SVD SUMMARY  (all linear layers, PR from deepmath_spectral)")
    print("=" * 72)
    print(f"{'Model':<16} {'Pre-train':>10} {'Stable rank':>16} "
          f"{'Part. ratio (PR)':>18} {'Decay AUC':>11}")
    print("-" * 72)
    for m in MODEL_ORDER:
        r = rows.loc[m]
        print(
            f"{m:<16} "
            f"{PRETRAINING_TOKENS[m]:>10} "
            f"{r['stable_rank_mean']:>10.1f} ± {r['stable_rank_std']:>5.1f}  "
            f"{r['pr_mean']:>11.1f} ± {r['pr_std']:>5.1f}  "
            f"{r['decay_auc_mean']:>11.4f}"
        )
    print("=" * 72)
    print("\nMonotonic increase across rows in all three columns")
    print("= evidence that pre-training scale predicts spectral maturity.")


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------

def save_metadata():
    """Save library versions and run timestamp for reproducibility."""
    metadata = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "transformers_version": transformers.__version__,
        "timestamp": datetime.now().isoformat(),
        "models": {k: str(Path(v).resolve()) for k, v in MODELS.items()},
        "spectra_seed": 42,
        "spectra_top_k": TOP_K,
        "max_layers_for_spectra": MAX_LAYERS_FOR_SPECTRA,
        "exclude_keywords": EXCLUDE_KEYWORDS,
        "pr_source": "deepmath_spectral.effective_rank_from_singular_values",
        "decay_source": "deepmath_spectral.normalized_decay",
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reproducibility metadata to {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen base model SVD analysis for the Model Maturity Effect.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip model analysis; only aggregate existing CSVs and regenerate plots.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Force re-analysis of all models even if results already exist.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    print("Frozen Base Model SVD Analysis")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Using effective_rank_from_singular_values and normalized_decay")
    print(f"from deepmath_spectral — results commensurable with fine-tuned checkpoints.\n")

    save_metadata()

    all_dfs = []

    if args.plot_only:
        # Load all existing CSVs, skip any model loading
        print("\n** Plot-only mode: loading existing CSVs **")
        for model_key in MODEL_ORDER:
            csv_path = OUTPUT_DIR / f"{model_key}_svd_metrics.csv"
            if csv_path.exists():
                all_dfs.append(load_existing_results(model_key))
            else:
                print(f"  ⚠  No results found for {model_key}, skipping.")
    else:
        # Phase 1: full metric sweep across all layers
        for model_key, local_path in MODELS.items():
            if not args.no_resume and has_existing_results(model_key):
                print(f"\n✓ {model_key}: results already exist, skipping "
                      f"(use --no-resume to force).")
                all_dfs.append(load_existing_results(model_key))
            else:
                df = analyze_model(model_key, local_path)
                all_dfs.append(df)

        # Phase 2: sampled decay curves for plotting
        for model_key, local_path in MODELS.items():
            npy_path = OUTPUT_DIR / f"{model_key}_spectra.npy"
            if not args.no_resume and npy_path.exists():
                print(f"\n✓ {model_key}: spectra already exist, skipping.")
            else:
                extract_spectra(model_key, local_path)

    if not all_dfs:
        print("\nNo results to aggregate. Run without --plot-only first.")
        return

    # Aggregate and print summary
    summary, combined = aggregate_results(all_dfs)
    print_paper_table(summary)

    # Phase 3: plots
    print("\n[Phase 3] Generating plots...")
    plot_pr_by_layer_type(combined)
    plot_mean_decay_curves()

    print(f"\nDone. Key outputs in {OUTPUT_DIR.resolve()}:")
    print("  metadata.json                 — reproducibility info (lib versions, config)")
    print("  summary_table.csv             — per-model and per-layer-type stats")
    print("  *_svd_metrics.csv             — per-layer raw metrics (all layers)")
    print("  *_spectra.npy                 — normalized top-64 SV curves (sampled)")
    print("  pr_by_layer_type.pdf          — box plots by layer type")
    print("  mean_decay_curves.pdf         — primary figure for paper/response")


if __name__ == "__main__":
    main()
