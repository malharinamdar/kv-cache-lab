"""Turn results/*.jsonl into figures/ and results/summary.md.

    python -m experiments.plot
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Colour follows the method in every figure; marker shape is the second channel.
METHODS = {
    "topk_dynamic": ("Top-k attention, dynamic (oracle)", "#2a78d6", "o"),
    "snapkv_question_aware": ("SnapKV, question-aware", "#eb6834", "s"),
    "snapkv": ("SnapKV, question-agnostic", "#1baf7a", "^"),
    "streaming_llm": ("StreamingLLM", "#eda100", "v"),
    "random": ("Random eviction", "#e87ba4", "D"),
    "kv_quant": ("KV quantization (KIVI-style)", "#008300", "h"),
}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load_accuracy(path: Path):
    """{(context_tokens, method, budget): [correct, wrong_code, n, memory_fraction, attended_fraction]}"""
    table = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    for row in map(json.loads, path.read_text().splitlines()):
        cell = table[(row["context_tokens"], row["method"], round(row["budget"], 4))]
        cell[0] += row["correct"]
        cell[1] += row["wrong_code"]
        cell[2] += 1
        cell[3], cell[4] = row["memory_fraction"], row["attended_fraction"]
    return table


def percent(x, _=None):
    return f"{x * 100:g}%"


def plot_niah(table, out: Path):
    lengths = sorted({key[0] for key in table})
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.6 * len(lengths), 3.6), sharey=True, squeeze=False)
    for ax, length in zip(axes[0], lengths):
        n = table[(length, "full", 1.0)][2]
        full = table[(length, "full", 1.0)][0] / max(n, 1)
        ax.axhline(full, color=INK_2, linewidth=1)
        ax.annotate(f"full cache ({full:.0%})", (0.021, full), xytext=(0, 4), textcoords="offset points", color=INK_2,
                    fontsize=8, va="bottom")
        for slot, (method, (label, colour, marker)) in enumerate(METHODS.items()):
            # Quantization points sit at their memory fraction; every other method at its budget.
            cells = sorted((cell[3] if name.startswith("kv_int") else budget, name, cell)
                           for (l, name, budget), cell in table.items()
                           if l == length and (name == method or (method == "kv_quant" and name.startswith("kv_int"))))
            if not cells:
                continue
            # Methods that tie (often all at 0%) would hide each other, so each one is nudged sideways a little.
            dodge = 1.0 if method == "kv_quant" else 2 ** ((slot - 2) * 0.05)
            xs, ys = [x * dodge for x, _, _ in cells], [cell[0] / cell[2] for _, _, cell in cells]
            for x, (_, _, cell) in zip(xs, cells):
                low, high = wilson(cell[0], cell[2])
                ax.plot([x, x], [low, high], color=colour, alpha=0.35, linewidth=1, solid_capstyle="butt")
            style = dict(color=colour, marker=marker, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
            if method == "kv_quant":
                ax.plot(xs, ys, linestyle="none", **style)
                for (x, name, _), y in zip(cells, ys):
                    ax.annotate(f"{name[len('kv_int'):]}-bit", (x, y), xytext=(0, -9), textcoords="offset points",
                                color=INK_2, fontsize=8, ha="center", va="top")
            else:
                ax.plot(xs, ys, linewidth=1.8, solid_joinstyle="round", solid_capstyle="round", **style)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.02, 0.4)
        ax.set_xticks([0.025, 0.05, 0.1, 0.2])
        ax.xaxis.set_major_formatter(FuncFormatter(percent))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(-0.03, 1.12)
        ax.yaxis.set_major_formatter(FuncFormatter(percent))
        ax.grid(axis="y")
        ax.set_title(f"Needle retrieval at {length // 1000}K tokens (Qwen3-0.6B, n={n})", loc="left")
        ax.set_xlabel("KV budget (share of context KV entries)")
    axes[0][0].set_ylabel("Needle retrieval accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_speed(speed: dict, out: Path):
    rows = speed["rows"]
    lengths = [r["context_tokens"] for r in rows]
    fig, (memory_ax, latency_ax) = plt.subplots(1, 2, figsize=(8.2, 3.2))

    per_token = rows[0]["kv_cache_bytes_formula"] / rows[0]["context_tokens"]
    xs = list(range(0, 32_001, 500))
    weights_gb = speed["weights_bytes"] / 1e9
    memory_ax.plot(xs, [x * per_token / 1e9 for x in xs], color="#2a78d6", linewidth=1.8, label="KV cache (formula)")
    memory_ax.plot(lengths, [r["kv_cache_bytes_measured"] / 1e9 for r in rows], linestyle="none", marker="o",
                   markersize=6, color="#2a78d6", markeredgecolor=SURFACE, markeredgewidth=1.2, label="KV cache (measured)")
    memory_ax.axhline(weights_gb, color=INK_2, linewidth=1)
    memory_ax.annotate(f"model weights, {weights_gb:.2f} GB", (31_500, weights_gb), xytext=(0, -4), textcoords="offset points",
                       color=INK_2, fontsize=8, ha="right", va="top")
    crossover = speed["weights_bytes"] / per_token
    memory_ax.annotate(f"cache outgrows weights\nat {crossover / 1000:.1f}K tokens", (crossover, weights_gb),
                       xytext=(14, -34), textcoords="offset points", color=INK_2, fontsize=8,
                       arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    memory_ax.set_xlim(0, 32_000)
    memory_ax.set_ylim(0, None)
    memory_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}K" if x else "0"))
    memory_ax.set_xlabel("Context length (tokens)")
    memory_ax.set_ylabel("GB (16-bit)")
    memory_ax.set_title("Qwen3-0.6B: KV cache vs weights", loc="left")
    memory_ax.grid(axis="y")
    memory_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)

    series = [("decode_ms_full", "Full cache", INK_2, "o"),
              ("decode_ms_streaming_llm", "Evicted to 10% (StreamingLLM)", "#eda100", "v"),
              ("decode_ms_topk_oracle", "Top-k 10%, oracle scoring", "#2a78d6", "o")]
    for key, label, colour, marker in series:
        latency_ax.plot(lengths, [r[key] for r in rows], color=colour, marker=marker, linewidth=1.8, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    latency_ax.set_xticks(lengths)
    latency_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}K"))
    latency_ax.set_ylim(0, None)
    latency_ax.set_xlabel("Context length (tokens)")
    latency_ax.set_ylabel("ms per generated token")
    latency_ax.set_title("Decode latency on an 8 GB M2 (PyTorch MPS)", loc="left")
    latency_ax.grid(axis="y")
    latency_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    for key, _, _, _ in series:  # value at the end of each line
        latency_ax.annotate(f"{rows[-1][key]:.0f}", (lengths[-1], rows[-1][key]), xytext=(6, 0),
                            textcoords="offset points", va="center", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(table, speed, out: Path):
    lengths = sorted({key[0] for key in table})
    lines = ["# Results summary", "", "Needle retrieval accuracy, % correct [95% Wilson CI]. "
             "`wrong` = answered with a *different* needle's code.", ""]
    lines.append("| method | budget | " + " | ".join(f"{l // 1000}K acc | {l // 1000}K wrong" for l in lengths) + " |")
    lines.append("|---|---|" + "---|---|" * len(lengths))
    order = ["full", "topk_dynamic", "snapkv_question_aware", "snapkv", "streaming_llm", "random", "kv_int4", "kv_int2"]
    keys = sorted({(m, b) for (_, m, b) in table}, key=lambda mb: (order.index(mb[0]), mb[1]))
    for method, budget in keys:
        cells = []
        for length in lengths:
            correct, wrong, n, memory, attended = table.get((length, method, budget), [0, 0, 0, 0, 0])
            if n == 0:
                cells += ["", ""]
                continue
            low, high = wilson(correct, n)
            cells += [f"{100 * correct / n:.0f} [{100 * low:.0f}–{100 * high:.0f}]", f"{100 * wrong / n:.0f}"]
        budget_text = f"{budget * 100:g}% bits" if method.startswith("kv_int") else f"{budget * 100:g}%"
        lines.append(f"| {method} | {budget_text} | " + " | ".join(cells) + " |")
    if speed:
        lines += ["", "## Systems (Qwen3-0.6B, fp16, PyTorch MPS on an 8 GB M2)", "",
                  "| context | prefill tok/s | KV cache MB | decode ms/token: full | evicted to 10% | top-k 10% (oracle) |",
                  "|---|---|---|---|---|---|"]
        for r in speed["rows"]:
            lines.append(f"| {r['context_tokens']} | {r['prefill_tokens_per_second']:.0f} | {r['kv_cache_bytes_measured'] / 2**20:.0f} "
                         f"| {r['decode_ms_full']:.0f} | {r['decode_ms_streaming_llm']:.0f} | {r['decode_ms_topk_oracle']:.0f} |")
    out.write_text("\n".join(lines) + "\n")


def main():
    Path("figures").mkdir(exist_ok=True)
    speed = json.loads(Path("results/speed.json").read_text()) if Path("results/speed.json").exists() else None
    table = load_accuracy(Path("results/niah.jsonl"))
    plot_niah(table, Path("figures/niah_accuracy.png"))
    if speed:
        plot_speed(speed, Path("figures/memory_and_latency.png"))
    write_summary(table, speed, Path("results/summary.md"))
    print(Path("results/summary.md").read_text())


if __name__ == "__main__":
    main()
