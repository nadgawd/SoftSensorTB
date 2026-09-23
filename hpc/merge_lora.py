#!/usr/bin/env python3
"""Merge LoRA adapter into base weights for vLLM serving."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model directory")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--out", required=True, help="Merged output directory")
    args = p.parse_args()

    print(f"Loading base: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    print(f"Saving merged model: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
