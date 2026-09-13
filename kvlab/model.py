"""Qwen3 decoder written from scratch in plain PyTorch (transformers is only used by the tests).

The whole inference path fits in this file:

    tokens -> embedding -> 28 x [RMSNorm -> attention (reads/writes KV cache) -> RMSNorm -> SwiGLU MLP]
           -> RMSNorm -> lm_head -> logits

The model does not own its KV cache. Every forward call appends the keys/values of the new
tokens to a `KVCache` passed in by the caller, and an optional policy may restrict which
cached entries each attention call can see. Those two hooks are all the compression
experiments need.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tokenizers import Tokenizer

from kvlab.cache import KVCache

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MODEL_FILES = ["config.json", "tokenizer.json", "model.safetensors"]


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path: Path) -> Qwen3Config:
        raw = json.loads(Path(path).read_text())
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})

    def kv_bytes_per_token(self, bytes_per_number: int = 2) -> int:
        """Keys and values, for every layer and every KV head: 2 * L * H_kv * d_head numbers per token."""
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * bytes_per_number


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    x_float = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x_float.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def causal_mask(n_new: int, n_total: int, device: torch.device) -> torch.Tensor | None:
    """The new tokens are the last `n_new` cache entries; token i may attend to every entry up to itself."""
    if n_new == 1:
        return None  # a single new token may see the whole cache
    return torch.ones(n_new, n_total, dtype=torch.bool, device=device).tril(diagonal=n_total - n_new)


def grouped_query_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                            mask: torch.Tensor | None) -> torch.Tensor:
    """softmax(q k^T / sqrt(d)) v, where every KV head is shared by a group of query heads (GQA).

    q: [1, heads, n_new, d]; keys, values: [1, kv_heads, n_total, d]
    mask: bool, broadcastable to [1, kv_heads, group, n_new, n_total]; True = may attend.
    Broadcasting each KV head across its group avoids copying the cache, which on every
    decode step would otherwise cost as much memory traffic as the attention itself.
    """
    _, heads, n_new, d = q.shape
    kv_heads = keys.shape[1]
    grouped = q.view(1, kv_heads, heads // kv_heads, n_new, d)
    scores = (grouped @ keys[:, :, None].transpose(-1, -2)).float() / d ** 0.5  # [1, kv_heads, group, n_new, n_total]
    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))
    weights = scores.softmax(dim=-1).to(values.dtype)
    return (weights @ values[:, :, None]).reshape(1, heads, n_new, d)


class Qwen3:
    def __init__(self, cfg: Qwen3Config, weights: dict[str, torch.Tensor], device: torch.device,
                 dtype: torch.dtype, max_positions: int = 32768):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.embed = weights["model.embed_tokens.weight"]
        self.final_norm = weights["model.norm.weight"]
        self.lm_head = self.embed if cfg.tie_word_embeddings else weights["lm_head.weight"]
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{i}."
            self.layers.append({
                name: weights[prefix + full_name]
                for name, full_name in [
                    ("input_layernorm", "input_layernorm.weight"),
                    ("q_proj", "self_attn.q_proj.weight"),
                    ("k_proj", "self_attn.k_proj.weight"),
                    ("v_proj", "self_attn.v_proj.weight"),
                    ("o_proj", "self_attn.o_proj.weight"),
                    ("q_norm", "self_attn.q_norm.weight"),
                    ("k_norm", "self_attn.k_norm.weight"),
                    ("post_attention_layernorm", "post_attention_layernorm.weight"),
                    ("gate_proj", "mlp.gate_proj.weight"),
                    ("up_proj", "mlp.up_proj.weight"),
                    ("down_proj", "mlp.down_proj.weight"),
                ]
            })

        # RoPE: position p rotates each pair of dimensions (i, i + d/2) by angle p * theta^(-2i/d).
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        angles = torch.arange(max_positions, dtype=torch.float32)[:, None] * inv_freq[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        self.cos = angles.cos().to(device=device, dtype=dtype)
        self.sin = angles.sin().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_MODEL, device: torch.device | None = None,
                        dtype: torch.dtype | None = None) -> tuple[Qwen3, Tokenizer]:
        path = Path(name) if Path(name).is_dir() else Path(snapshot_download(name, allow_patterns=MODEL_FILES))
        cfg = Qwen3Config.from_json(path / "config.json")
        device = device or default_device()
        if dtype is None:
            # bfloat16 only where the hardware has it: Apple GPUs before M3 and older NVIDIA cards
            # (e.g. a Colab T4) emulate it several times slower, so they get float16.
            has_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
            dtype = torch.float32 if device.type == "cpu" else torch.bfloat16 if has_bf16 else torch.float16
        weights = {}
        with safe_open(path / "model.safetensors", framework="pt") as f:
            for key in f.keys():
                if key == "lm_head.weight" and cfg.tie_word_embeddings:
                    continue  # tied: the output projection reuses the embedding matrix
                weights[key] = f.get_tensor(key).to(device=device, dtype=dtype)  # one tensor at a time keeps peak RAM low
        return cls(cfg, weights, device, dtype), Tokenizer.from_file(str(path / "tokenizer.json"))

    def new_cache(self, capacity: int) -> KVCache:
        cfg = self.cfg
        return KVCache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, capacity, self.device, self.dtype)

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, cache: KVCache, policy=None, captured: list | None = None,
                capture_last: int = 0, all_logits: bool = False) -> torch.Tensor:
        """Run a block of new tokens (1-D LongTensor) that continues whatever is already in `cache`.

        Returns float32 logits for the last position, or for every position if `all_logits`.
        If `captured` is a list, the last `capture_last` query vectors of every layer are kept in it.
        SnapKV scores the cache with those.
        """
        n_new = ids.shape[0]
        positions = torch.arange(cache.next_pos, cache.next_pos + n_new, device=self.device)
        cos, sin = self.cos[positions], self.sin[positions]
        eps = self.cfg.rms_norm_eps

        x = F.embedding(ids, self.embed)[None]  # [1, n_new, hidden]
        for i, layer in enumerate(self.layers):
            h = rms_norm(x, layer["input_layernorm"], eps)
            x = x + self._attention(i, layer, h, cos, sin, cache, policy, captured, capture_last)
            h = rms_norm(x, layer["post_attention_layernorm"], eps)
            x = x + F.linear(F.silu(F.linear(h, layer["gate_proj"])) * F.linear(h, layer["up_proj"]), layer["down_proj"])
        cache.advance(n_new)

        if not all_logits:
            x = x[:, -1:]  # the full [n_new, vocab] matrix for an 8K prompt would be ~5 GB
        return F.linear(rms_norm(x, self.final_norm, eps), self.lm_head)[0].float()

    def _attention(self, i, layer, x, cos, sin, cache, policy, captured, capture_last):
        cfg = self.cfg
        _, n_new, _ = x.shape
        q = F.linear(x, layer["q_proj"]).view(1, n_new, cfg.num_attention_heads, cfg.head_dim)
        k = F.linear(x, layer["k_proj"]).view(1, n_new, cfg.num_key_value_heads, cfg.head_dim)
        v = F.linear(x, layer["v_proj"]).view(1, n_new, cfg.num_key_value_heads, cfg.head_dim)

        # Qwen3 RMS-normalises every query and key head ("QK-norm") before applying RoPE.
        q = rms_norm(q, layer["q_norm"], cfg.rms_norm_eps).transpose(1, 2)  # [1, heads, n_new, head_dim]
        k = rms_norm(k, layer["k_norm"], cfg.rms_norm_eps).transpose(1, 2)  # [1, kv_heads, n_new, head_dim]
        v = v.transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # Keys are cached after RoPE, so a cached entry keeps its position even if its neighbours are evicted.
        keys, values = cache.write(i, k, v)
        if captured is not None:
            previous = captured[i]
            captured[i] = (q if previous is None else torch.cat([previous, q], dim=2))[:, :, -capture_last:]

        mask = causal_mask(n_new, keys.shape[2], x.device)
        if policy is not None:
            mask = policy.attention_mask(q, keys, mask, cache)
        out = grouped_query_attention(q, keys, values, mask)
        return F.linear(out.transpose(1, 2).reshape(1, n_new, -1), layer["o_proj"])
