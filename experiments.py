"""The three measurements in the README: speed, memory, attention sinks. About 10 minutes on a laptop CPU.

    python experiments.py
"""

import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from gpt2 import HEAD_DIM, N_HEADS, N_LAYERS, evict, forward, generate, load

w, tokenizer = load()
results = {}

# 1. Speed: how long it takes to generate n tokens, with and without the cache.
prompt = torch.tensor(tokenizer.encode("Once upon a time").ids)
results["speed"] = []
for n in [64, 128, 256, 512]:
    row = {"new_tokens": n}
    for use_cache in [False, True]:
        start = time.perf_counter()
        generate(w, prompt, n, use_cache)
        row["seconds_with_cache" if use_cache else "seconds_without_cache"] = time.perf_counter() - start
    results["speed"].append(row)
    print(row, flush=True)

# 2. Memory: every token stores a key and a value vector per head, in every layer (float32 = 4 bytes).
bytes_per_token = 2 * N_LAYERS * N_HEADS * HEAD_DIM * 4
text = torch.tensor(tokenizer.encode(open("data/stories.txt").read()).ids)
cache = [None] * N_LAYERS
with torch.no_grad():
    forward(w, text[:1024], cache)
measured = sum(k.nbytes + v.nbytes for k, v in cache)
weights = sum(t.nbytes for name, t in w.items() if not name.endswith(".attn.bias"))  # h.*.attn.bias = causal masks
results["memory"] = {"bytes_per_token_formula": bytes_per_token, "bytes_per_token_measured": measured / 1024,
                     "cache_mb_at_1024_tokens": measured / 2**20, "weights_mb": weights / 2**20}
print(results["memory"], flush=True)


# 3. Attention sinks: read a text one token at a time with a capped cache, and measure perplexity.
@torch.no_grad()
def perplexity(ids, window=None, sinks=0):
    cache, total_loss = [None] * N_LAYERS, 0.0
    for t in range(len(ids) - 1):
        logits = forward(w, ids[t:t + 1], cache, pos=t)
        total_loss -= F.log_softmax(logits[-1], dim=-1)[ids[t + 1]].item()
        if window and cache[0][0].shape[1] > window:
            cache = evict(cache, window, sinks)
    return math.exp(total_loss / (len(ids) - 1))


chunks = [text[:1024], text[1024:2048]]  # two 1,024-token passages; results are averaged over them
results["sinks"] = {"full_cache": sum(perplexity(c) for c in chunks) / 2, "windows": []}
print(results["sinks"], flush=True)
for window in [64, 128, 256]:
    row = {"window": window}
    for sinks in [0, 4]:
        row[f"perplexity_{sinks}_sinks"] = sum(perplexity(c, window, sinks) for c in chunks) / 2
    results["sinks"]["windows"].append(row)
    print(row, flush=True)

with open("results/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Figure: one panel per experiment.
SURFACE, INK, INK_2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK_2,
                     "axes.titlecolor": INK, "axes.titleweight": "bold", "xtick.color": INK_2, "ytick.color": INK_2,
                     "grid.color": GRID, "legend.frameon": False, "legend.labelcolor": INK_2})
style = dict(linewidth=1.8, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
fig, (speed_ax, sink_ax) = plt.subplots(1, 2, figsize=(8.4, 3.3))

ns = [r["new_tokens"] for r in results["speed"]]
for key, label, colour in [("seconds_without_cache", "Without cache", ORANGE), ("seconds_with_cache", "With KV cache", BLUE)]:
    ys = [r[key] for r in results["speed"]]
    speed_ax.plot(ns, ys, color=colour, label=label, **style)
    speed_ax.annotate(f"{ys[-1]:.0f} s", (ns[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                      color=INK_2)
speed_ax.set(xlabel="Tokens generated", ylabel="Seconds", xticks=ns, title="Generation time, GPT-2 on a laptop CPU")
speed_ax.set_ylim(0, None)

windows = [r["window"] for r in results["sinks"]["windows"]]
for key, label, colour in [("perplexity_0_sinks", "Last W tokens only", ORANGE),
                           ("perplexity_4_sinks", "First 4 + last W-4 tokens", BLUE)]:
    ys = [r[key] for r in results["sinks"]["windows"]]
    sink_ax.plot(windows, ys, color=colour, label=label, **style)
    sink_ax.annotate(f"{ys[-1]:.0f}", (windows[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                     color=INK_2)
full = results["sinks"]["full_cache"]
sink_ax.axhline(full, color=INK_2, linewidth=1)
sink_ax.annotate(f"full cache ({full:.1f})", (windows[0], full), xytext=(0, -4), textcoords="offset points",
                 va="top", color=INK_2)
sink_ax.set_yscale("log")
sink_ax.set(xlabel="Cache size W (tokens kept)", ylabel="Perplexity (log scale, lower is better)", xticks=windows,
            title="Evicting the first tokens breaks the model")
sink_ax.set_xticklabels([str(x) for x in windows])
for ax in (speed_ax, sink_ax):
    ax.grid(axis="y", linewidth=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.tight_layout()
fig.savefig("figures/results.png", dpi=200, bbox_inches="tight")
print("saved figures/results.png and results/results.json")
