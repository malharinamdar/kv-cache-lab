"""A pre-allocated KV cache with just enough operations to evict entries or copy it."""

from __future__ import annotations

import torch


BUCKET = 1024  # capacities are rounded up to this, so freed buffers can be reused instead of fragmenting memory


class KVCache:
    """One key buffer and one value buffer per layer, each shaped [1, kv_heads, capacity, head_dim].

    - `length`   number of stored entries.
    - `next_pos` RoPE position the next token will get. After an eviction it no longer equals `length`.
    - `n_ctx`    how many of the stored entries belong to the long context, the part a policy may
                 compress. Entries after it (question, generated answer) are always kept.
    """

    def __init__(self, n_layers: int, kv_heads: int, head_dim: int, capacity: int,
                 device: torch.device, dtype: torch.dtype):
        capacity = -(-capacity // BUCKET) * BUCKET
        shape = (1, kv_heads, capacity, head_dim)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.length = 0
        self.next_pos = 0
        self.n_ctx = 0

    @property
    def capacity(self) -> int:
        return self.k[0].shape[2]

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store a block of new keys/values and return views of every entry for this layer, new block included."""
        start, end = self.length, self.length + k.shape[2]
        if end > self.capacity:
            raise RuntimeError(f"KV cache full: need {end} entries but capacity is {self.capacity}")
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n_new: int) -> None:
        """Called once per forward pass, after every layer has written its block."""
        self.length += n_new
        self.next_pos += n_new

    def rewind(self, length: int) -> None:
        """Drop the most recent entries, so one context prefill can be reused for many questions."""
        self.next_pos -= self.length - length
        self.length = length

    def nbytes(self) -> int:
        """Bytes held by the stored entries (not the unused capacity)."""
        per_entry = 2 * len(self.k) * self.k[0].shape[1] * self.k[0].shape[3] * self.k[0].element_size()
        return self.length * per_entry

    def select_context(self, keep: torch.Tensor) -> KVCache:
        """New cache that keeps only the context entries in `keep`, plus every entry after the context.

        keep: LongTensor [n_layers, kv_heads, n_keep] of indices into the context region. Each
        (layer, KV head) can keep different tokens, but all keep the same number of them.
        The new cache keeps the same free room after the context, for the question and answer.
        """
        n_layers, kv_heads, n_keep = keep.shape
        head_dim = self.k[0].shape[3]
        tail = self.length - self.n_ctx
        capacity = n_keep + self.capacity - self.n_ctx
        new = KVCache(n_layers, kv_heads, head_dim, capacity, self.k[0].device, self.k[0].dtype)
        for layer in range(n_layers):
            index = keep[layer][None, :, :, None].expand(1, kv_heads, n_keep, head_dim)
            new.k[layer][:, :, :n_keep] = self.k[layer][:, :, :self.n_ctx].gather(2, index)
            new.v[layer][:, :, :n_keep] = self.v[layer][:, :, :self.n_ctx].gather(2, index)
            new.k[layer][:, :, n_keep:n_keep + tail] = self.k[layer][:, :, self.n_ctx:self.length]
            new.v[layer][:, :, n_keep:n_keep + tail] = self.v[layer][:, :, self.n_ctx:self.length]
        new.length, new.next_pos, new.n_ctx = n_keep + tail, self.next_pos, n_keep
        return new

    def clone(self) -> KVCache:
        n_layers, (_, kv_heads, capacity, head_dim) = len(self.k), self.k[0].shape
        new = KVCache(n_layers, kv_heads, head_dim, capacity, self.k[0].device, self.k[0].dtype)
        for layer in range(n_layers):
            new.k[layer][:, :, :self.length] = self.k[layer][:, :, :self.length]
            new.v[layer][:, :, :self.length] = self.v[layer][:, :, :self.length]
        new.length, new.next_pos, new.n_ctx = self.length, self.next_pos, self.n_ctx
        return new
