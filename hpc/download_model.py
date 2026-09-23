#!/usr/bin/env python3
"""Download a HuggingFace model snapshot to $SCRATCH_DIR/models/<name>.

    source hpc/env.sh && activate_env
    python hpc/download_model.py --repo Qwen/Qwen3.5-27B-GPTQ-Int4

Streams files straight to disk. Do NOT use transformers' ``from_pretrained``
to fetch weights: it materialises the whole model in CPU RAM (and upcasts to
fp32 unless told otherwise), which for a 7B model is ~28 GB of RAM for a job
whose only purpose is writing files.

Needs no GPU — run it on the login node and keep your GPU allocation for vLLM.
Weights land on scratch and survive across allocations, so this is a one-time
cost per model, not something you repeat each time you get a node.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before huggingface_hub is imported to take effect. (hf_transfer
# is retired upstream; Xet is its replacement.)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=os.environ.get("MODEL_REPO", "Qwen/Qwen3.5-27B-GPTQ-Int4"),
        help="HuggingFace repo id (default: $MODEL_REPO)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output dir (default: $SCRATCH_DIR/models/<repo_tail>)",
    )
    args = parser.parse_args()

    scratch = os.environ.get("SCRATCH_DIR")
    if not scratch:
        raise SystemExit(
            "SCRATCH_DIR is unset. Run 'source hpc/env.sh' first — guessing "
            "/scratch/<username> gives a path you cannot write to on this cluster."
        )

    tail = args.repo.split("/")[-1]
    out_dir = Path(args.out) if args.out else Path(scratch) / "models" / tail
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo} -> {out_dir}")
    snapshot_download(repo_id=args.repo, local_dir=str(out_dir))
    print(f"Done: {out_dir}")
    print("Serve it with: qsub hpc/serve_llm.pbs")


if __name__ == "__main__":
    main()
