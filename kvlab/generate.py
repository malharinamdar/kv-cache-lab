"""Prefill and greedy decoding: the two phases of LLM inference.

Prefill reads the whole prompt in large parallel blocks. It is compute-bound.
Decode produces one token per forward pass and re-reads the weights plus the entire KV cache
every step. It is memory-bound, which is why the size of the KV cache matters so much.
"""

from __future__ import annotations

import torch


@torch.inference_mode()
def prefill(model, ids: torch.Tensor, cache, policy=None, chunk_size: int = 512, capture_last: int = 0):
    """Feed a prompt through the model in chunks, filling the KV cache.

    Chunking (as vLLM and SGLang do) bounds the attention score matrix at chunk_size x cache_length
    instead of prompt_length^2, which is what lets an 8K prompt fit on an 8 GB laptop.
    Returns (logits of the last prompt token, last `capture_last` queries of each layer or None).
    """
    captured = [None] * model.cfg.num_hidden_layers if capture_last else None
    logits = None
    for start in range(0, ids.shape[0], chunk_size):
        logits = model.forward(ids[start:start + chunk_size], cache, policy, captured, capture_last)
        if model.device.type == "mps" and ids.shape[0] > chunk_size:
            # Every chunk's attention matrix has a new shape. The MPS allocator caches each one
            # and, left alone, grew past 8 GB of RAM on an 8K prompt.
            torch.mps.empty_cache()
    return logits, captured


@torch.inference_mode()
def greedy_decode(model, cache, logits: torch.Tensor, policy=None, max_new_tokens: int = 16,
                  stop_ids: tuple[int, ...] = ()) -> list[int]:
    """Pick the most likely token, feed it back, repeat. Each step appends one entry to the cache."""
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        token = int(logits[-1].argmax())
        if token in stop_ids:
            break
        tokens.append(token)
        logits = model.forward(torch.tensor([token], device=model.device), cache, policy)
    return tokens
