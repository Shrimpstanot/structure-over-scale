import os
import sys
import shlex
import subprocess
import argparse
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Axolotl YAML config.",
    )
    p.add_argument(
        "--nproc_per_node",
        type=int,
        default=4,
        help="Number of processes per node (default: paper setup of 4).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # NCCL, tokenizer, and allocator defaults for multi-GPU runs.
    # Override these in the environment if your cluster requires different values.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Respect caller-provided Hugging Face cache/offline settings.
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
        os.environ.setdefault("HF_DATASETS_CACHE", hf_home)

    # Build Axolotl config path
    config_path = pathlib.Path(args.config).as_posix()

    # Construct torch.distributed.run command
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m",
        "axolotl_train_entry",
        config_path,
    ]

    print("EXEC:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    # Ensure subprocesses can import local modules such as rewards.py.
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(SRC_DIR)]
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
