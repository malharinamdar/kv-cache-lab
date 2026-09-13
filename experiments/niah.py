"""Multi-key needle-in-a-haystack under different KV-cache policies.

Each sample hides several "secret code" sentences between TinyStories stories, then asks for one
of them. The context is prefilled once with a full cache. Every policy then compresses that same
cache and answers, so all methods see identical inputs.

    python -m experiments.niah --lengths 2000 4000 8000 --samples 40
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")  # fail loudly instead of swapping on 8 GB Macs
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import argparse
import gc
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from kvlab.generate import greedy_decode, prefill
from kvlab.model import Qwen3
from kvlab.policies import FullCache, QuantizedKV, RandomEviction, SnapKV, StreamingLLM, TopKAttention

KEYS = [
    "the red lighthouse", "Captain Orla", "the silver kettle", "Professor Banyan", "the north tower",
    "the glass garden", "Uncle Tavish", "the copper bridge", "the midnight train", "the old observatory",
    "Aunt Priya", "the paper boat", "the iron gate", "Doctor Wren", "the blue caravan", "the clock museum",
    "the marble fountain", "Grandpa Ilya", "the hidden library", "the frozen lake", "Sergeant Pike",
    "the lemon orchard", "the stone well", "the velvet curtain",
]
HEADER = "<|im_start|>user\nRead the stories below. A few secret codes are hidden among them.\n\n"
# The empty think block is Qwen3's non-thinking mode. The answer is started for the model (as RULER does),
# so it replies with the number itself instead of sometimes spending its token budget on a sentence.
QUESTION = ("\n\nQuestion: What is the secret code for {key}? Answer with the number only.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nThe secret code for {key} is")
STOP_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>
WINDOW = 32


@dataclass
class Sample:
    context_ids: list[int]
    question_ids: list[int]
    answer: str
    distractors: list[str]
    target_depth: float  # position of the asked-for needle in the context: 0 = start, 1 = end


def load_stories(tokenizer, n_stories: int = 6000):
    path = hf_hub_download("roneneldan/TinyStories", "data/validation-00000-of-00001-869c898b519ad725.parquet",
                           repo_type="dataset")
    texts = [t.strip() for t in pq.read_table(path).column("text").to_pylist()[:n_stories] if len(t.strip()) > 100]
    lengths = [len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False)]
    return texts, lengths


def build_sample(tokenizer, stories, story_lengths, context_tokens: int, n_needles: int, rng: random.Random) -> Sample:
    keys = rng.sample(KEYS, n_needles)
    codes = [str(code) for code in rng.sample(range(1_000_000, 10_000_000), n_needles)]
    needles = [f"The secret code for {key} is {code}." for key, code in zip(keys, codes)]

    room = context_tokens - 20 - 16 * n_needles  # header and needles
    picked = []
    for i in rng.sample(range(len(stories)), len(stories)):
        if story_lengths[i] + 2 <= room:
            picked.append(stories[i])
            room -= story_lengths[i] + 2
        if room < 40:
            break

    slots = sorted(rng.choices(range(len(picked) + 1), k=n_needles))
    pieces, target = [], rng.randrange(n_needles)
    for position in range(len(picked) + 1):
        for needle_index, slot in enumerate(slots):
            if slot == position:
                pieces.append(needles[needle_index])
        if position < len(picked):
            pieces.append(picked[position])
    context = HEADER + "\n\n".join(pieces)

    return Sample(
        context_ids=tokenizer.encode(context, add_special_tokens=False).ids,
        question_ids=tokenizer.encode(QUESTION.format(key=keys[target]), add_special_tokens=False).ids,
        answer=codes[target],
        distractors=[code for i, code in enumerate(codes) if i != target],
        target_depth=context.index(needles[target]) / len(context),
    )


def free_cached_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def agnostic_policies(budgets, seed):
    """Policies that compress the context before the question arrives."""
    policies = [FullCache()]
    for budget in budgets:
        policies += [StreamingLLM(budget), RandomEviction(budget, seed=seed), SnapKV(budget), TopKAttention(budget)]
    return policies + [QuantizedKV(bits=4), QuantizedKV(bits=2)]


def first_number(text: str) -> str | None:
    match = re.search(r"\d+", text)
    return match.group() if match else None


def evaluate_sample(model, tokenizer, sample: Sample, budgets, seed: int, max_new_tokens: int = 12):
    device = model.device
    context = torch.tensor(sample.context_ids, device=device)
    question = torch.tensor(sample.question_ids, device=device)
    n_ctx = len(context)
    base = model.new_cache(n_ctx + len(question) + max_new_tokens)
    _, window_queries = prefill(model, context, base, capture_last=WINDOW)
    base.n_ctx = n_ctx

    rows = []

    def record(policy, cache, tokens):
        text = tokenizer.decode(tokens)
        number = first_number(text)
        rows.append({
            "method": policy.name,
            "budget": policy.budget,
            "memory_fraction": policy.memory_fraction(cache.n_ctx, n_ctx),
            "attended_fraction": policy.attended_fraction(cache.n_ctx, n_ctx),
            "correct": number == sample.answer,
            "wrong_code": number in sample.distractors,  # confidently returned a different needle's code
            "output": text,
        })

    # Question-agnostic: compress first, then read the question through the compressed cache.
    for policy in agnostic_policies(budgets, seed):
        cache = policy.compress(base, window_queries)
        logits, _ = prefill(model, question, cache, policy)
        record(policy, cache, greedy_decode(model, cache, logits, policy, max_new_tokens, STOP_IDS))
        base.rewind(n_ctx)  # policies that reuse `base` wrote the question and answer into it
        del cache
        free_cached_memory(device)

    # Question-aware SnapKV (the original paper's setting): read the question with the full cache, then compress.
    logits, question_queries = prefill(model, question, base, capture_last=WINDOW)
    for budget in budgets:
        policy = SnapKV(budget, question_aware=True)
        cache = policy.compress(base, question_queries)
        record(policy, cache, greedy_decode(model, cache, logits, policy, max_new_tokens, STOP_IDS))
        del cache
        free_cached_memory(device)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[2000, 4000, 8000])
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--needles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/niah.jsonl"))
    args = parser.parse_args()

    model, tokenizer = Qwen3.from_pretrained()
    stories, story_lengths = load_stories(tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():  # resume an interrupted run
        done = {(r["context_tokens"], r["sample"]) for r in map(json.loads, args.out.read_text().splitlines())}

    for context_tokens in args.lengths:
        for index in range(args.samples):
            if (context_tokens, index) in done:
                continue
            seed = args.seed * 1_000_003 + context_tokens * 1009 + index
            sample = build_sample(tokenizer, stories, story_lengths, context_tokens, args.needles, random.Random(seed))
            start = time.time()
            rows = evaluate_sample(model, tokenizer, sample, args.budgets, seed)
            free_cached_memory(model.device)
            with args.out.open("a") as f:
                for row in rows:
                    row.update(context_tokens=context_tokens, sample=index, n_ctx=len(sample.context_ids),
                               target_depth=round(sample.target_depth, 4))
                    f.write(json.dumps(row) + "\n")
            full = next(r for r in rows if r["method"] == "full")
            print(f"[{context_tokens} #{index}] n_ctx={len(sample.context_ids)} full={'ok ' if full['correct'] else 'MISS'} "
                  f"output={re.sub(r'\\s+', ' ', full['output'])[:30]!r} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
