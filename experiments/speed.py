"""Where do time and memory go as the context grows?

For each context length this measures prefill throughput, the KV cache size (measured and from
the formula), and decode latency with the full cache, an evicted cache, and oracle top-k attention.

    python -m experiments.speed --lengths 1000 2000 4000 8000
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")  # fail loudly instead of swapping on 8 GB Macs
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.niah import load_stories
from kvlab.generate import prefill
from kvlab.model import Qwen3
from kvlab.policies import FullCache, StreamingLLM, TopKAttention


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def decode_latency_ms(model, cache, policy, steps: int) -> float:
    """Median time per decode step. The token fed in is fixed, so only the cache size varies."""
    token = torch.tensor([13], device=model.device)
    times = []
    for step in range(steps + 2):
        start = time.perf_counter()
        model.forward(token, cache, policy)
        synchronize(model.device)
        if step >= 2:  # skip warm-up steps
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("results/speed.json"))
    args = parser.parse_args()

    model, tokenizer = Qwen3.from_pretrained()
    stories, _ = load_stories(tokenizer)
    haystack = tokenizer.encode("\n\n".join(stories[:3000]), add_special_tokens=False).ids
    print(f"device={model.device} dtype={model.dtype} "
          f"KV per token={model.cfg.kv_bytes_per_token() / 1024:.0f} KiB (fp16)", flush=True)

    rows = []
    for n_ctx in args.lengths:
        ids = torch.tensor(haystack[:n_ctx], device=model.device)
        cache = model.new_cache(n_ctx + args.steps + 2)
        synchronize(model.device)
        start = time.perf_counter()
        prefill(model, ids, cache)
        synchronize(model.device)
        prefill_seconds = time.perf_counter() - start
        cache.n_ctx = n_ctx

        row = {
            "context_tokens": n_ctx,
            "prefill_tokens_per_second": n_ctx / prefill_seconds,
            "kv_cache_bytes_measured": cache.nbytes(),
            "kv_cache_bytes_formula": n_ctx * model.cfg.kv_bytes_per_token(),
        }
        for label, policy in [("full", FullCache()), ("streaming_llm", StreamingLLM(args.budget)),
                              ("topk_oracle", TopKAttention(args.budget))]:
            compressed = policy.compress(cache, None)
            row[f"decode_ms_{label}"] = decode_latency_ms(model, compressed, policy, args.steps)
            if label == "streaming_llm":
                row["kv_cache_bytes_streaming_llm"] = compressed.nbytes()
            cache.rewind(n_ctx)
            del compressed
        rows.append(row)
        print({key: round(value, 1) if isinstance(value, float) else value for key, value in row.items()}, flush=True)
        del cache
        if model.device.type == "mps":
            torch.mps.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": str(model.device), "dtype": str(model.dtype),
                                    "weights_bytes": sum(t.nbytes for t in [model.embed, model.final_norm]
                                                         + [w for layer in model.layers for w in layer.values()]),
                                    "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
