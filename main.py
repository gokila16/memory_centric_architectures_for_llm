"""
main.py
Entry point for the memory-centric LLM architecture analysis.

Usage examples:
    # Run full analysis (all seq lengths, both policies, all plots):
    python main.py

    # Run only the policy simulation:
    python main.py --mode simulate --n_tokens 1000

    # Run only the memory traffic analysis:
    python main.py --mode analyze

    # Change HBM eviction policy comparison at specific seq length:
    python main.py --mode simulate --n_tokens 500
"""

import argparse
import csv
import json
import os

import pandas as pd

from memory_model import (
    run_all_seq_lengths,
    kv_int4_savings,
    attention_windowing_savings,
    operator_fusion_savings,
    LAYERS, HEADS, HEAD_DIM, BYTES_FP16,
)
from simulator import simulate_lru, simulate_asep
import plotting

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
GRAPHS_DIR  = "graphs"


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(GRAPHS_DIR, exist_ok=True)


def save_measurements(results: dict, path: str = "measurements.json") -> None:
    """Persist analytical memory traffic data to JSON."""
    payload = {
        "model":       "gpt2-medium",
        "model_config": {
            "n_layers": LAYERS, "n_heads": HEADS,
            "head_dim": HEAD_DIM, "bytes_per_elem": BYTES_FP16,
        },
        "seq_lengths_tested": SEQ_LENGTHS,
        "per_seq": results,
        "summary": {
            "attn_always_memory_bound": all(
                results[str(T)]["attn_ai"] < 156 for T in SEQ_LENGTHS
            ),
            "weight_bytes_total_constant_mb": round(
                results["128"]["weight_bytes_total"] / 1e6, 2
            ),
            "bandwidth_limited_stage": "attention",
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")


def save_simulation_results(lru_history, asep_history,
                            csv_path="simulator_results.csv",
                            json_path="simulator_summary.json") -> None:
    """Persist simulation traces and summary statistics."""
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "token", "policy", "hbm_used_mb", "dram_used_mb",
            "hbm_occupancy", "hbm_blocks", "dram_blocks",
            "evictions", "wasted_evictions", "promotion_cost_mb",
            "latency_ns", "bytes_moved_gb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in lru_history:
            writer.writerow({**h, "policy": "LRU"})
        for h in asep_history:
            writer.writerow({**h, "policy": "Dynamic"})
    print(f"Saved: {csv_path}")

    lru_final  = lru_history[-1]
    asep_final = asep_history[-1]
    summary = {
        "lru_final":     lru_final,
        "dynamic_final": asep_final,
        "improvement": {
            "wasted_evictions_reduction_pct": (
                (lru_final["wasted_evictions"] - asep_final["wasted_evictions"])
                / max(lru_final["wasted_evictions"], 1) * 100
            ),
            "promotion_cost_reduction_pct": (
                (lru_final["promotion_cost_mb"] - asep_final["promotion_cost_mb"])
                / max(lru_final["promotion_cost_mb"], 1) * 100
            ),
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_analysis():
    """Compute and save analytical memory traffic model."""
    print("=" * 60)
    print("Analytical memory traffic model — GPT-2 medium")
    print("=" * 60)
    results = run_all_seq_lengths(SEQ_LENGTHS)
    save_measurements(results)

    print(f"\n{'T':>6} | {'Total MB':>10} | {'Attn AI':>10} | {'FFN AI':>8} | Tier")
    print("-" * 55)
    for T in SEQ_LENGTHS:
        r = results[str(T)]
        print(f"{T:>6} | {r['total_bytes_mb']:>10.2f} | "
              f"{r['attn_ai']:>10.4f} | {r['ffn_ai']:>8.2f} | {r['kv_tier']}")
    return results


def run_optimizations(results: dict):
    """Compute and plot all three memory optimisations."""
    print("\n" + "=" * 60)
    print("Memory optimisations")
    print("=" * 60)

    opt1 = [kv_int4_savings(T)                                          for T in SEQ_LENGTHS]
    opt2 = [attention_windowing_savings(T, window=512)                   for T in SEQ_LENGTHS]
    opt3 = [operator_fusion_savings(T, results[str(T)]["total_bytes"])  for T in SEQ_LENGTHS]

    plotting.plot_kv_compression(opt1)
    plotting.plot_attention_windowing(opt2)
    plotting.plot_operator_fusion(opt3)
    plotting.plot_optimization_summary(opt1, opt2, opt3)
    return opt1, opt2, opt3


def run_simulation(n_tokens: int = 1000):
    """Run LRU vs ASEP simulation and generate comparison plots."""
    print("\n" + "=" * 60)
    print(f"Policy simulation — {n_tokens} tokens")
    print("=" * 60)

    print("  Simulating LRU ...")
    lru_history,  lru_final  = simulate_lru(n_tokens)
    print("  Simulating ASEP ...")
    asep_history, asep_final = simulate_asep(n_tokens)

    save_simulation_results(lru_history, asep_history)

    df = pd.read_csv("simulator_results.csv")
    plotting.plot_policy_comparison(df)
    plotting.plot_asep_vs_lru_slides(df)

    we_lru  = lru_final["wasted_evictions"]
    we_asep = asep_final["wasted_evictions"]
    pc_lru  = lru_final["promotion_cost_mb"]
    pc_asep = asep_final["promotion_cost_mb"]

    print(f"\n  Wasted evictions — LRU: {we_lru},  ASEP: {we_asep}  "
          f"(↓ {(we_lru - we_asep)/max(we_lru,1)*100:.1f}%)")
    print(f"  Promotion cost   — LRU: {pc_lru:.1f} MB,  ASEP: {pc_asep:.1f} MB  "
          f"(↓ {(pc_lru - pc_asep)/max(pc_lru,1)*100:.1f}%)")
    return lru_history, asep_history


def run_roofline_energy(measurements: dict):
    """Generate roofline and energy breakdown plots."""
    print("\n" + "=" * 60)
    print("Roofline model and energy breakdown")
    print("=" * 60)
    plotting.plot_roofline(measurements)
    plotting.plot_energy_breakdown(measurements)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Memory-centric LLM architecture analysis")
    p.add_argument("--mode", choices=["all", "analyze", "simulate", "plots"],
                   default="all",
                   help="Which phase to run (default: all)")
    p.add_argument("--n_tokens", type=int, default=1000,
                   help="Number of decode steps in policy simulation (default: 1000)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    if args.mode in ("all", "analyze"):
        results = run_analysis()
    else:
        # Load cached measurements if only running simulation or plots
        with open("measurements.json") as f:
            import json
            raw = json.load(f)
        results = raw["per_seq"]

    if args.mode in ("all", "analyze"):
        run_optimizations(results)
        run_roofline_energy(raw if args.mode == "plots" else
                            {**{"seq_lengths_tested": SEQ_LENGTHS}, "per_seq": results})

    if args.mode in ("all", "simulate"):
        run_simulation(args.n_tokens)

    print("\nDone. All outputs in ./graphs/ and current directory.")


if __name__ == "__main__":
    main()
