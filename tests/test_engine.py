"""Correctness checks for the from-scratch engine, run in float32 on CPU so differences are real bugs.

    python -m tests.test_engine        (or: pytest tests/)
"""

import gc
from functools import lru_cache

import torch

from kvlab.generate import prefill
from kvlab.model import DEFAULT_MODEL, Qwen3
from kvlab.policies import RandomEviction, SnapKV, StreamingLLM, TopKAttention, fake_quantize

CONTEXT = ("<|im_start|>user\nMira kept a jar of buttons on the windowsill. Every morning she counted them, "
           "and every evening her brother Tomas hid one somewhere in the garden. The secret code for the red "
           "lighthouse is 4829173. One day the jar was empty, and Mira went looking.")
QUESTION = "\n\nWhat is the secret code for the red lighthouse?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOLERANCE = 1e-3


@lru_cache(maxsize=1)
def cpu_model():
    return Qwen3.from_pretrained(device=torch.device("cpu"), dtype=torch.float32)


def encode(text):
    model, tokenizer = cpu_model()
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False).ids)


def test_logits_match_huggingface():
    model, _ = cpu_model()
    ids = torch.cat([encode(CONTEXT), encode(QUESTION)])
    ours = model.forward(ids, model.new_cache(len(ids)), all_logits=True)

    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.float32)
    with torch.inference_mode():
        theirs = hf_model(ids[None]).logits[0].float()
    del hf_model
    gc.collect()

    max_diff = (ours - theirs).abs().max().item()
    print(f"  max |logit difference| vs Hugging Face over {len(ids)} positions: {max_diff:.2e}")
    assert torch.equal(ours.argmax(-1), theirs.argmax(-1))
    assert max_diff < TOLERANCE


def test_chunked_prefill_then_decode_matches_one_pass():
    model, _ = cpu_model()
    ids = torch.cat([encode(CONTEXT), encode(QUESTION)])
    one_pass = model.forward(ids, model.new_cache(len(ids)), all_logits=True)

    cache = model.new_cache(len(ids))
    n_decode = 6
    prefill(model, ids[:-n_decode], cache, chunk_size=7)  # odd chunk size on purpose
    stepwise = torch.stack([model.forward(ids[i:i + 1], cache)[-1] for i in range(len(ids) - n_decode, len(ids))])

    max_diff = (stepwise - one_pass[-n_decode:]).abs().max().item()
    print(f"  chunked prefill + cached decode vs one pass: {max_diff:.2e}")
    assert max_diff < TOLERANCE


def test_policies_at_full_budget_change_nothing():
    model, _ = cpu_model()
    context, question = encode(CONTEXT), encode(QUESTION)
    reference = model.forward(torch.cat([context, question]), model.new_cache(len(context) + len(question)))

    base = model.new_cache(len(context) + len(question))
    _, window_queries = prefill(model, context, base, capture_last=32)
    base.n_ctx = len(context)
    for policy in [StreamingLLM(1.0), RandomEviction(1.0), SnapKV(1.0), TopKAttention(1.0)]:
        cache = policy.compress(base, window_queries)
        logits, _ = prefill(model, question, cache, policy)
        max_diff = (logits - reference).abs().max().item()
        print(f"  {policy.name:>14} at budget 1.0 vs full cache: {max_diff:.2e}")
        assert max_diff < TOLERANCE
        base.rewind(len(context))


def synthetic_cache(n_ctx=100, n_tail=5, layers=2, kv_heads=2, head_dim=4):
    from kvlab.cache import KVCache
    cache = KVCache(layers, kv_heads, head_dim, n_ctx + n_tail + 8, torch.device("cpu"), torch.float32)
    for layer in range(layers):  # entry i stores the value i, so kept indices can be read back
        ramp = torch.arange(n_ctx + n_tail, dtype=torch.float32)[None, None, :, None].expand(1, kv_heads, -1, head_dim)
        cache.write(layer, ramp, ramp)
    cache.advance(n_ctx + n_tail)
    cache.n_ctx = n_ctx
    return cache


def test_streaming_llm_keeps_sinks_recent_tokens_and_tail():
    cache = synthetic_cache()
    kept = StreamingLLM(0.2, sinks=4).compress(cache, None)
    ids = kept.k[0][0, 0, :kept.length, 0].long().tolist()
    assert ids == [0, 1, 2, 3] + list(range(84, 100)) + list(range(100, 105))
    assert (kept.n_ctx, kept.next_pos) == (20, 105)  # positions continue where the full cache stopped


def test_topk_mask_allows_budget_plus_everything_after_context():
    cache = synthetic_cache()
    keys = torch.randn(1, 2, 105, 4)
    allowed = TopKAttention(0.1).attention_mask(torch.randn(1, 4, 3, 4), keys, None, cache)  # 3 queries, 2 KV heads
    assert allowed.shape == (1, 2, 1, 3, 105)
    assert (allowed[..., :100].sum(-1) == 10).all() and allowed[..., 100:].all()


def test_fake_quantize_error_is_bounded():
    x = torch.randn(1, 8, 100, 128)  # 128 channels = 4 groups of 32, no padding
    groups = x.unflatten(-1, (-1, 32))
    for bits in (2, 4, 8):
        restored = fake_quantize(x, bits, group=32, dim=3)
        step = (groups.amax(-1) - groups.amin(-1)).max() / (2 ** bits - 1)
        assert (restored - x).abs().max() <= step / 2 + 1e-5


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            test()
    print("all tests passed")
