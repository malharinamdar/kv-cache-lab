"""KV-cache policies. Each one answers one or both of two questions:

1. compress(cache, queries): once the long context has been read, which entries do we keep, and at what precision?
2. attention_mask(q, keys, mask, cache): when a later token attends to the cache, which entries may it look at?

Eviction and quantization only use (1). Dynamic sparse attention only uses (2).
Budgets are fractions of the context's KV entries, so every method is compared at the same budget.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def n_keep(budget: float, n_ctx: int, minimum: int = 1) -> int:
    return max(minimum, min(n_ctx, round(budget * n_ctx)))


def grouped_attention(q: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """softmax(q . k / sqrt(d)) averaged over the query heads that share each KV head.

    q: [1, heads, n_q, d], keys: [1, kv_heads, n_k, d]  ->  [1, kv_heads, n_q, n_k]
    Eviction can only drop a whole KV head entry, so importance is scored per KV head.
    """
    _, heads, n_q, d = q.shape
    kv_heads = keys.shape[1]
    grouped = q.view(1, kv_heads, heads // kv_heads, n_q, d)
    logits = (grouped @ keys[:, :, None].transpose(-1, -2)).float() / d ** 0.5  # [1, kv_heads, group, n_q, n_k]
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return logits.softmax(dim=-1).mean(dim=2)


def fake_quantize(x: torch.Tensor, bits: int, group: int, dim: int) -> torch.Tensor:
    """Asymmetric min-max quantization in groups of `group` consecutive numbers along `dim`, then dequantize."""
    moved = x.float().movedim(dim, -1)
    n = moved.shape[-1]
    pad = -n % group
    if pad:  # repeat the last value so padding cannot stretch a group's min/max
        moved = torch.cat([moved, moved[..., -1:].expand(*moved.shape[:-1], pad)], dim=-1)
    grouped = moved.unflatten(-1, (-1, group))
    low = grouped.amin(dim=-1, keepdim=True)
    scale = (grouped.amax(dim=-1, keepdim=True) - low).clamp(min=1e-6) / (2 ** bits - 1)
    restored = (grouped - low).div_(scale).round_().clamp_(0, 2 ** bits - 1).mul_(scale).add_(low)  # in place: 8K caches are big
    return restored.flatten(-2)[..., :n].movedim(-1, dim).to(x.dtype)


class FullCache:
    """No compression. The accuracy ceiling."""

    name = "full"
    question_aware = False

    def __init__(self, budget: float = 1.0):
        self.budget = budget

    def compress(self, cache, queries):
        return cache

    def attention_mask(self, q, keys, mask, cache):
        return mask

    def memory_fraction(self, kept: int, n_ctx: int) -> float:
        """Stored context KV as a fraction of the full 16-bit cache."""
        return kept / n_ctx

    def attended_fraction(self, kept: int, n_ctx: int) -> float:
        """Fraction of the context each new query actually attends to."""
        return kept / n_ctx


class StreamingLLM(FullCache):
    """Keep the first few tokens (attention sinks) and the most recent ones. Xiao et al., ICLR 2024."""

    name = "streaming_llm"

    def __init__(self, budget: float, sinks: int = 4):
        super().__init__(budget)
        self.sinks = sinks

    def compress(self, cache, queries):
        n_ctx = cache.n_ctx
        keep = n_keep(self.budget, n_ctx, minimum=self.sinks + 1)
        index = torch.cat([torch.arange(self.sinks), torch.arange(n_ctx - (keep - self.sinks), n_ctx)])
        n_layers, kv_heads = len(cache.k), cache.k[0].shape[1]
        return cache.select_context(index.to(cache.k[0].device).expand(n_layers, kv_heads, keep))


class RandomEviction(FullCache):
    """Sanity baseline: keep the attention sinks plus a random subset of the context."""

    name = "random"

    def __init__(self, budget: float, sinks: int = 4, seed: int = 0):
        super().__init__(budget)
        self.sinks = sinks
        self.generator = torch.Generator().manual_seed(seed)

    def compress(self, cache, queries):
        n_ctx = cache.n_ctx
        keep = n_keep(self.budget, n_ctx, minimum=self.sinks + 1)
        scores = torch.rand(len(cache.k), cache.k[0].shape[1], n_ctx, generator=self.generator)
        scores[:, :, :self.sinks] = float("inf")
        index = scores.topk(keep, dim=-1).indices.sort(dim=-1).values
        return cache.select_context(index.to(cache.k[0].device))


class SnapKV(FullCache):
    """Keep the context entries that the last `window` prompt tokens attend to most. Li et al., NeurIPS 2024.

    This bets that tokens important to the end of the prompt stay important later, the
    "persistence of importance" hypothesis from Scissorhands (Liu, Desai et al., NeurIPS 2023).
    - question_aware=True: compress after the question is read, so the window is the question itself.
      This is the setting of the original paper.
    - question_aware=False: compress before the question arrives, as when a long document is cached
      once and reused for many different questions. The window is then the last context tokens.
    """

    name = "snapkv"

    def __init__(self, budget: float, question_aware: bool = False, window: int = 32, kernel: int = 7):
        super().__init__(budget)
        self.question_aware, self.window, self.kernel = question_aware, window, kernel
        if question_aware:
            self.name = "snapkv_question_aware"

    def compress(self, cache, queries):
        n_ctx, total = cache.n_ctx, cache.length
        keep = n_keep(self.budget, n_ctx)
        index = []
        for layer, q in enumerate(queries):
            n_q = q.shape[2]
            mask = torch.ones(n_q, total, dtype=torch.bool, device=q.device).tril(diagonal=total - n_q)
            probs = grouped_attention(q, cache.k[layer][:, :, :total], mask)[0]  # [kv_heads, n_q, total]
            scores = probs.sum(dim=1)[:, :n_ctx]
            # Pooling keeps the neighbours of an important token, so a whole phrase survives, not one piece of it.
            scores = F.avg_pool1d(scores[None], self.kernel, stride=1, padding=self.kernel // 2)[0]
            window_start = total - n_q
            if window_start < n_ctx:
                scores[:, window_start:] = float("inf")  # the observation window itself is always kept
            index.append(scores.topk(keep, dim=-1).indices.sort(dim=-1).values)
        return cache.select_context(torch.stack(index))


class TopKAttention(FullCache):
    """Oracle dynamic sparse attention: store the whole cache, but every new query attends only to the
    `budget` fraction of context entries it scores highest, re-chosen at every step and layer.

    "Oracle" means it finds those entries by scoring all of them, so here it saves no compute.
    It is the accuracy ceiling for methods that approximate top-k cheaply, such as Quest,
    HashAttention and vAttention. Scoring and granularity match SnapKV: mean attention
    probability per KV head. Only *when* the choice is made differs.
    """

    name = "topk_dynamic"

    def attention_mask(self, q, keys, mask, cache):
        n_ctx = cache.n_ctx
        keep = n_keep(self.budget, n_ctx)
        probs = grouped_attention(q, keys, mask)[..., :n_ctx]  # [1, kv_heads, n_q, n_ctx]
        chosen = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, probs.topk(keep, dim=-1).indices, True)
        allowed = torch.ones(1, keys.shape[1], 1, q.shape[2], keys.shape[2], dtype=torch.bool, device=q.device)
        allowed[:, :, 0, :, :n_ctx] = chosen  # one choice per KV head, shared by its query heads
        return allowed if mask is None else allowed & mask

    def memory_fraction(self, kept: int, n_ctx: int) -> float:
        return 1.0

    def attended_fraction(self, kept: int, n_ctx: int) -> float:
        return n_keep(self.budget, n_ctx) / n_ctx


class QuantizedKV(FullCache):
    """Keep every context entry but store it in `bits` bits, KIVI-style (Liu et al., ICML 2024).

    Keys are quantized per channel and values per token, with asymmetric min-max scaling in groups
    of `group` numbers. Quantization is simulated (quantize, then immediately dequantize), so the
    accuracy numbers are real but the memory fraction is computed, not measured.
    """

    name = "kv_quant"

    def __init__(self, bits: int, group: int = 32):
        super().__init__((bits + 32 / group) / 16)
        self.bits, self.group = bits, group
        self.name = f"kv_int{bits}"

    def compress(self, cache, queries):
        new = cache.clone()
        n_ctx = cache.n_ctx
        for layer in range(len(new.k)):
            k, v = new.k[layer][:, :, :n_ctx], new.v[layer][:, :, :n_ctx]
            new.k[layer][:, :, :n_ctx] = fake_quantize(k, self.bits, self.group, dim=2)  # groups of tokens, per channel
            new.v[layer][:, :, :n_ctx] = fake_quantize(v, self.bits, self.group, dim=3)  # groups of channels, per token
        return new

    def memory_fraction(self, kept: int, n_ctx: int) -> float:
        return (self.bits + 32 / self.group) / 16  # each group also stores a 16-bit scale and zero point

    def attended_fraction(self, kept: int, n_ctx: int) -> float:
        return 1.0
