# Structure Over Scale

This source release supports selected Qwen3-8B / DeepMath-Hard training and
spectral-analysis workflows for *Structure Over Scale: Rethinking Adaptation for
Reinforcement Learning with Verifiable Rewards*. It contains code and
configurations only: model weights, datasets, checkpoints, logs, plots, and the
paper PDF are not included. The included workflows are not a claim that every
paper experiment can be reproduced from this tree; see
[`docs/reproducibility.md`](docs/reproducibility.md).

## Supported setup

The supported training route is the pinned Singularity image. A local build
requires Singularity and passwordless `sudo` (`sudo -n`) for the build and final
ownership change:

```sh
./scripts/build_sif.sh
```

The default image output, `containers/structure-over-scale-cu128.sif`, is
ignored by Git. The build downloads the runtime dependencies, so it requires
network access. Runtime training does not use `sudo`. If your site supplies an
image, point `SIF_PATH` at it instead. The wrapper does not provide a generic
`pip install` alternative; the large CUDA, FlashAttention, vLLM, and QuanTA
stack is pinned in `containers/requirements-runtime.txt` and has not been
validated as a portable host environment.

Supply these assets outside this repository:

- A local Qwen3-8B **base** checkpoint, not an instruction-tuned checkpoint.
- A Hugging Face `datasets` saved dataset containing the already-curated
  5,399-row DeepMath-Hard training split at `train/`. The release does not
  contain or construct that split.
- Writable run and cache directories.

Defaults expect `models/qwen3-8b`, `datasets/deepmath-hard`, `runs`, and `cache`
beside this release directory. Override any path for your machine. For example:

```sh
SIF_PATH=/path/to/runtime.sif \
MODEL_DIR=/data/models/qwen3-8b \
DATASET_DIR=/data/deepmath-hard \
RUNS_DIR=/scratch/my-runs \
CACHE_DIR=/scratch/my-cache \
  ./scripts/run_grpo.sh lora-r8 4 42 64
```

This requests LoRA r=8, four local processes, seed 42, and a completion batch
of 64. Allocate the matching GPU resources before running: launchers are
scheduler-neutral and deliberately do not check allocations, GPU count, or
assets. The default `NPROC` is four for one four-GPU node. Runs write resolved
YAML and provenance JSON alongside outputs under
`$RUNS_DIR/<run-id>/seed-<seed>/bs-<batch>/`. Nondefault group sizes use
`$RUNS_DIR/<run-id>/seed-<seed>/g-<G>/bs-<batch>/` to avoid collisions; G=8
keeps the original path.

## Paper configuration families

`configs/deepmath-qwen3-8b/matrix.yaml` defines the method variants;
`base.yaml` contains their shared DeepMath-Hard protocol. The materializer
creates a resolved config and provenance record:

```sh
./scripts/run_grpo.sh quanta-d3-no-kv 4 42 64
```

Useful scheduler-neutral launchers are:

```sh
./scripts/run_paper_matrix.sh             # seed-42, batch-64 method/rank matrix
./scripts/run_batch_ablation.sh           # batch 64/128/256, seed 42
./scripts/run_rank_batch_replication.sh   # selected rank/batch runs, seed 43
./scripts/run_group_size_reconstruction.sh # optional reconstructed G=4/16 runs
./scripts/run_multiseed.sh                # selected structured methods and FFT, seeds 42-44
./scripts/run_quanta_target_ablation.sh   # QuanTA target-module variants
```

The optional group-size reconstruction uses completion batches 32 for G=4 and
128 for G=16, keeping eight unique prompts per optimizer update. Since this
changes completion batch while varying G, it is **not** the paper's described
controlled G-only protocol and is not presented as a reproduction. Both this
geometry and the per-rank microbatch are reconstructed settings, not explicitly
specified in the paper. The paper's
main G=8 protocol uses completion batch 64. See
[`docs/reproducibility.md`](docs/reproducibility.md).

The matrix covers LoRA ranks 1/4/8/16/32/64/128/256; DoRA ranks 8/16/32/64;
QuanTA d=3 `[16,16,16]` and d=4 `[16,8,8,4]`, including the represented target
variants; and full fine-tuning (FFT). It does not imply that every entry has a
paper evaluation implementation in this release.

The configured reward functions are available source implementations, not a
claim of exact Appendix A.3.2 behavior. The format reward has known differences
from the paper's additive description; see the reward note in
[`docs/reproducibility.md`](docs/reproducibility.md) for the behavior implemented
in `src/rewards.py`.

## Analysis and checks

- `src/deepmath_spectral.py` contains checkpoint auditing, update reconstruction,
  effective-rank summaries, and plotting helpers. It needs model/checkpoint
  artifacts and a user-supplied score file; those are not included. The
  analysis API expects a flat directory of recognized checkpoint names (for
  example `qwen_lora_rank8`) and is not wired directly to the training
  launcher's nested run layout. A staged analysis directory may be needed:

  ```python
  from src.deepmath_spectral import analyze_runs, discover_runs, plot_analysis

  runs = discover_runs("/data/analysis-runs", scores_path="/data/scores.txt")
  result = analyze_runs(
      runs,
      base_model_dir="/data/models/qwen3-8b",
      output_dir="/data/analysis-output",
      representative_layer="model.layers.18.self_attn.q_proj",
  )
  plot_analysis(
      result,
      output_dir="/data/analysis-output",
      representative_layer="model.layers.18.self_attn.q_proj",
  )
  ```
- Plotting helpers import `matplotlib` and `seaborn`, which are not in the
  training container's runtime lock. `requirements-analysis.txt` lists these
  as an optional, intentionally unpinned add-on. Readers may install this
  add-on into a separately managed analysis environment if needed; existing
  prerequisites such as PyTorch, NumPy, pandas, SciPy, and safetensors must
  already be available there. This add-on and the analysis route have not been
  installed or validated; it is not part of the paper environment freeze.
- `src/frozen_svd_analysis.py` analyzes the frozen Apertus-8B, Llama 3.1-8B,
  and Qwen3-8B weights. Its default input names are `./apertus-8b`,
  `./llama-3.1-8b`, and `./qwen3-8b`; it is a CPU analysis with substantial RAM
  requirements. From a directory containing those model folders, run
  `python3 /path/to/release/src/frozen_svd_analysis.py`; generated files go to
  `svd_results/` in the current directory.
- `src/preflight.py` is an opt-in asset integrity check, not a launcher gate.
- `tests/container_api_smoke.py` is a separate runtime diagnostic that requires
  the built image, mounted model/dataset, and a materialized config.

Run the offline unit tests and syntax checks from the repository root:

```sh
python3 -m unittest discover -s tests
python3 -m compileall -q src tests
for script in scripts/*.sh; do sh -n "$script"; done
```

These checks do not train a model, build the container, validate external data,
or establish that paper metrics are reproduced. Runtime/library-dependent
requirements and validation limits are documented rather than claimed as
working by default.

## License

The code and documentation in this repository are openly licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You may use, share, and adapt them, including commercially, provided you give
appropriate attribution and indicate changes. See [LICENSE](LICENSE) for the
full terms. Third-party dependencies, datasets, and model weights remain subject
to their respective licenses.

## Citation

BibTeX citation pending.
