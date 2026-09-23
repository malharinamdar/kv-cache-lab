"""Two checks: our GPT-2 matches Hugging Face's, and the KV cache does not change what gets generated.

    python test_gpt2.py
"""

import torch

from gpt2 import forward, generate, load

w, tokenizer = load()
PROMPT = "Once upon a time, a little robot wanted to learn how to"


@torch.no_grad()
def test_matches_huggingface():
    from transformers import GPT2LMHeadModel

    ids = torch.tensor(tokenizer.encode(PROMPT).ids)
    ours = forward(w, ids)
    theirs = GPT2LMHeadModel.from_pretrained("gpt2")(ids[None]).logits[0]
    diff = (ours - theirs).abs().max().item()
    print(f"max logit difference vs Hugging Face: {diff:.1e}")
    assert diff < 1e-3


def test_cache_gives_identical_text():
    ids = torch.tensor(tokenizer.encode(PROMPT).ids)
    with_cache = generate(w, ids, 40)
    without_cache = generate(w, ids, 40, use_cache=False)
    print("generated:", repr(tokenizer.decode(with_cache.tolist())))
    assert torch.equal(with_cache, without_cache)


if __name__ == "__main__":
    test_matches_huggingface()
    test_cache_gives_identical_text()
    print("both tests passed")
