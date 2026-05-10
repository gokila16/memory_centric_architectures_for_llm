"""
plotting.py
All matplotlib figure generation for the memory-centric LLM architecture project.
Each function saves a PNG and returns the figure object.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]


# ── Optimisation plots ────────────────────────────────────────────────────────

def plot_kv_compression(opt1_results: list, out_path: str = "graphs/opt1_kv_compression.png"):
    """FP16 vs INT4 KV cache size and HBM freed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    seqs = [r["seq_len"]      for r in opt1_results]
    fp16 = [r["kv_fp16_mb"]   for r in opt1_results]
    int4 = [r["kv_int4_mb"]   for r in opt1_results]
    freed = [r["hbm_freed_mb"] for r in opt1_results]

    ax1.plot(seqs, fp16, "r-o", label="FP16 KV (baseline)", linewidth=2)
    ax1.plot(seqs, int4, "g-o", label="INT4 KV (compressed)", linewidth=2)
    ax1.axhline(y=4, color="blue", linestyle="--", label="SRAM budget (4 MB)")
    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("KV Cache Size (MB)")
    ax1.set_title("Optimization 1 — KV Cache: FP16 vs INT4")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.bar([str(s) for s in seqs], freed, color="green", alpha=0.7)
    ax2.set_xlabel("Sequence Length (tokens)")
    ax2.set_ylabel("HBM Freed (MB)")
    ax2.set_title("HBM Freed by INT4 Compression")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return fig


def plot_attention_windowing(opt2_results: list, out_path: str = "graphs/opt2_attention_windowing.png"):
    """Full vs windowed KV read traffic."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    seqs   = [r["seq_len"]   for r in opt2_results]
    full   = [r["full_mb"]   for r in opt2_results]
    window = [r["window_mb"] for r in opt2_results]
    pcts   = [r["saved_pct"] for r in opt2_results]

    ax1.plot(seqs, full,   "r-o", label="Full attention", linewidth=2)
    ax1.plot(seqs, window, "g-o", label=f"Window={opt2_results[0]['window']}", linewidth=2)
    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("KV Read Traffic (MB)")
    ax1.set_title("Optimization 2 — Attention Windowing")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(seqs, pcts, "b-o", linewidth=2)
    ax2.set_xlabel("Sequence Length (tokens)")
    ax2.set_ylabel("Traffic Saved (%)")
    ax2.set_title("% Traffic Saved by Windowing")
    ax2.set_ylim(0, 100); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return fig


def plot_operator_fusion(opt3_results: list, out_path: str = "graphs/opt3_operator_fusion.png"):
    """Unfused vs fused total traffic and % reduction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    seqs     = [r["seq_len"]        for r in opt3_results]
    unfused  = [r["unfused_mb"]     for r in opt3_results]
    fused    = [r["fused_mb"]       for r in opt3_results]
    red_pcts = [r["reduction_pct"]  for r in opt3_results]

    ax1.plot(seqs, unfused, "r-o", label="Without fusion", linewidth=2)
    ax1.plot(seqs, fused,   "g-o", label="With fusion",    linewidth=2)
    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("Total Bytes Moved (MB)")
    ax1.set_title("Optimization 3 — Operator Fusion Impact")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.bar([str(s) for s in seqs], red_pcts, color="purple", alpha=0.7)
    ax2.set_xlabel("Sequence Length (tokens)")
    ax2.set_ylabel("Traffic Reduction (%)")
    ax2.set_title("% Traffic Reduction from Fusion")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return fig


def plot_optimization_summary(opt1, opt2, opt3, out_path: str = "graphs/optimization_summary.png"):
    """Side-by-side savings comparison for all three optimizations."""
    seqs = [r["seq_len"] for r in opt1]
    x    = np.arange(len(seqs))
    w    = 0.25

    int4_savings    = [r["reduction_pct"] for r in opt1]
    window_savings  = [r["saved_pct"]     for r in opt2]
    fusion_savings  = [r["reduction_pct"] for r in opt3]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w,   int4_savings,   w, label="INT4 Quantization", color="green",  alpha=0.8)
    ax.bar(x,       window_savings, w, label="Attention Window",   color="orange", alpha=0.8)
    ax.bar(x + w,   fusion_savings, w, label="Operator Fusion",    color="purple", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seqs])
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Traffic/Capacity Reduction (%)")
    ax.set_title("All Three Optimizations Compared")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return fig


# ── Roofline and energy ───────────────────────────────────────────────────────

def plot_roofline(measurements: dict, out_path: str = "graphs/roofline_improved.png"):
    """Roofline model showing attention is permanently memory-bound."""
    peak_flops = 312e12   # H100 FP16
    peak_bw    = 3.35e12  # H100 HBM
    ridge      = peak_flops / peak_bw

    ai_range  = np.logspace(-3, 4, 1000)
    perf_ceil = np.minimum(peak_flops, peak_bw * ai_range)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.loglog(ai_range, perf_ceil, "k-", linewidth=2.5, label="Roofline (H100)")
    ax.axvline(x=ridge, color="gray", linestyle="--", alpha=0.7,
               label=f"Ridge = {ridge:.0f} FLOP/byte")
    ax.axvspan(1e-3, ridge, alpha=0.05, color="red")

    seq_lengths = measurements["seq_lengths_tested"]
    colors      = plt.cm.Reds(np.linspace(0.4, 0.95, len(seq_lengths)))

    for i, T in enumerate(seq_lengths):
        ai  = measurements["per_seq"][str(T)]["attn_ai"]
        prf = min(peak_flops, peak_bw * ai)
        ax.scatter(ai, prf, color=colors[i], s=120, zorder=5)
        ax.annotate(f"Attn T={T}", (ai, prf),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=colors[i])

    ffn_ai  = measurements["per_seq"]["128"]["ffn_ai"]
    ffn_prf = min(peak_flops, peak_bw * ffn_ai)
    ax.scatter(ffn_ai, ffn_prf, color="royalblue", s=150,
               marker="s", zorder=5, label="FFN (all seq lengths)")

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (FLOP/s)", fontsize=12)
    ax.set_title("Roofline Model — GPT-2 Medium Decode Phase\n"
                 "Attention always memory-bound, never approaches ridge", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-3, 1e4); ax.set_ylim(1e9, 1e15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


def plot_energy_breakdown(measurements: dict, out_path: str = "graphs/energy_breakdown.png"):
    """Memory vs compute energy breakdown per decode step."""
    E_HBM  = 15e-12    # pJ/byte
    E_FLOP = 0.5e-12   # pJ/FLOP

    seq_lengths = measurements["seq_lengths_tested"]
    mem_energy  = [measurements["per_seq"][str(T)]["total_bytes"] * E_HBM * 1e6
                   for T in seq_lengths]
    comp_energy = [measurements["per_seq"][str(T)]["total_flops"] * E_FLOP * 1e6
                   for T in seq_lengths]

    x, w = np.arange(len(seq_lengths)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, mem_energy,  w, label="Memory energy (µJ)", color="tomato",    alpha=0.85)
    ax.bar(x + w/2, comp_energy, w, label="Compute energy (µJ)", color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in seq_lengths])
    ax.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax.set_ylabel("Energy per decode step (µJ)", fontsize=11)
    ax.set_title("Energy Breakdown — Memory vs Compute per Decode Step")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ratio = mem_energy[-1] / comp_energy[-1]
    ax.text(0.05, 0.90, f"Memory dominates compute\nby {ratio:.0f}× at T=4096",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return fig


# ── Policy comparison ─────────────────────────────────────────────────────────

def plot_policy_comparison(df: pd.DataFrame, out_path: str = "graphs/policy_comparison.png"):
    """4-panel LRU vs ASEP simulation results."""
    lru  = df[df["policy"] == "LRU"].reset_index(drop=True)
    asep = df[df["policy"] == "Dynamic"].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    panels = [
        (axes[0, 0], "hbm_occupancy",    lambda v: [x*100 for x in v], "HBM Occupancy (%)",
         "HBM Occupancy Over Conversation", {"lw": 1.5}, True),
        (axes[0, 1], "dram_used_mb",     None, "DRAM Used (MB)",
         "DRAM Usage Over Conversation",     {"lw": 1.5}, False),
        (axes[1, 0], "evictions",        None, "Cumulative Evictions",
         "Total Evictions Over Conversation",{"lw": 1.5}, False),
        (axes[1, 1], "latency_ns",       lambda v: [x/1e6 for x in v], "Decode Latency (ms)",
         "Decode Latency Over Conversation", {"lw": 1.5, "alpha": 0.7}, False),
    ]

    for ax, col, transform, ylabel, title, kw, add_threshold in panels:
        tokens = lru["token"]
        lru_vals  = transform(lru[col])  if transform else lru[col]
        asep_vals = transform(asep[col]) if transform else asep[col]
        ax.plot(tokens, lru_vals,  "r-", label="LRU",  **kw)
        ax.plot(tokens, asep_vals, "g-", label="ASEP", **kw)
        if add_threshold:
            ax.axhline(y=80, color="black", linestyle="--", label="80% threshold")
        ax.set_xlabel("Token Step"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("LRU vs ASEP — Full Conversation Simulation (1000 tokens)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return fig


def plot_asep_vs_lru_slides(df: pd.DataFrame, out_path: str = "graphs/asep_vs_lru_slides.png"):
    """Clean 2-panel wasted evictions and promotion cost for paper/slides."""
    lru  = df[df["policy"] == "LRU"].reset_index(drop=True)
    asep = df[df["policy"] == "Dynamic"].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, col, ylabel, title in [
        (axes[0], "wasted_evictions",  "Wasted Evictions",
         "Wasted Evictions — LRU vs ASEP"),
        (axes[1], "promotion_cost_mb", "Cumulative Promotion Cost (MB)",
         "Promotion Cost — LRU vs ASEP"),
    ]:
        ax.plot(lru["token"],  lru[col],  "r-", label="LRU (baseline)", linewidth=2.5)
        ax.plot(asep["token"], asep[col], "g-", label="ASEP (ours)",    linewidth=2.5)
        ax.fill_between(lru["token"], lru[col], asep[col], alpha=0.15, color="red")
        ax.set_xlabel("Token Step", fontsize=12)
        ax.set_ylabel(ylabel,       fontsize=12)
        ax.set_title(title,         fontsize=12)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    plt.suptitle("ASEP vs LRU — Controlled Policy Experiment\n"
                 "Same inputs, same eviction count, different choices",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return fig
