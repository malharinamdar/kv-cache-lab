"""GPT-2 (124M) inference from scratch in PyTorch, with a KV cache.

The whole model is the `forward` function below:

    token + position embeddings -> 12 x [LayerNorm -> attention -> LayerNorm -> MLP] -> LayerNorm -> logits

To generate text you run the model once per new token. Without a cache, every step re-processes
the entire sequence so far. With a KV cache, each layer keeps the keys and values of the tokens it
has already seen, so every step only has to process the one new token.
"""

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

N_LAYERS, N_HEADS, D_MODEL, HEAD_DIM = 12, 12, 768, 64


def load():
    """Pretrained GPT-2 weights (a plain dict of tensors) and its tokenizer."""
    weights = load_file(hf_hub_download("gpt2", "model.safetensors"))
    tokenizer = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
    return weights, tokenizer


def forward(w, ids, cache=None, pos=0):
    """Run the new tokens `ids` (a 1-D tensor) through GPT-2 and return one row of logits per token.

    cache: None to run without a cache, or a list with one (keys, values) pair per layer. The new
           tokens' keys and values are appended to it in place.
    pos:   position of the first new token in the text.
    Call it either with a whole prompt and an empty cache, or with one token at a time.
    """
    x = w["wte.weight"][ids] + w["wpe.weight"][pos:pos + len(ids)]  # [tokens, 768]
    for i in range(N_LAYERS):
        p = f"h.{i}."
        # Attention. GPT-2 stores its weights as [in, out], so it is x @ W with no transpose.
        h = F.layer_norm(x, (D_MODEL,), w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        q, k, v = (h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]).split(D_MODEL, dim=-1)
        q, k, v = (t.view(-1, N_HEADS, HEAD_DIM).transpose(0, 1) for t in (q, k, v))  # [12 heads, tokens, 64]
        if cache is not None:
            if cache[i] is not None:  # put the earlier tokens' keys and values in front of the new ones
                k = torch.cat([cache[i][0], k], dim=1)
                v = torch.cat([cache[i][1], v], dim=1)
            cache[i] = (k, v)
        # A whole prompt needs a causal mask (no peeking ahead). A single new token may look at everything.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=len(ids) > 1)
        x = x + y.transpose(0, 1).reshape(-1, D_MODEL) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        # MLP
        h = F.layer_norm(x, (D_MODEL,), w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        h = F.gelu(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"], approximate="tanh")
        x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
    x = F.layer_norm(x, (D_MODEL,), w["ln_f.weight"], w["ln_f.bias"])
    return x @ w["wte.weight"].T  # the output layer reuses the embedding matrix


@torch.no_grad()
def generate(w, ids, n_new, use_cache=True):
    """Greedy decoding: repeatedly append the most likely next token."""
    if not use_cache:
        for _ in range(n_new):
            logits = forward(w, ids)  # the whole sequence, every single step
            ids = torch.cat([ids, logits[-1:].argmax(-1)])
        return ids

    cache = [None] * N_LAYERS
    logits = forward(w, ids, cache)  # prefill: read the prompt once
    for _ in range(n_new):
        next_id = logits[-1:].argmax(-1)
        ids = torch.cat([ids, next_id])
        logits = forward(w, next_id, cache, pos=len(ids) - 1)  # decode: only the newest token
    return ids


def evict(cache, window, sinks=0):
    """Shrink every layer's cache to `window` entries: the first `sinks` tokens plus the most recent ones."""
    recent = window - sinks
    return [(torch.cat([k[:, :sinks], k[:, -recent:]], dim=1), torch.cat([v[:, :sinks], v[:, -recent:]], dim=1))
            for k, v in cache]
