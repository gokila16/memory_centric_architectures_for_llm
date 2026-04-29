# Memory-Centric LLM Architecture — GPT-2 Medium
### ASEP: Attention-Score-Driven Eviction Policy for Tiered KV Cache Management

---

## Project Overview

This project designs and evaluates a **memory-centric architecture for LLM inference**, where the memory hierarchy — not the compute array — is the primary design driver. Using GPT-2 medium as the target model, we analytically characterize decode-phase memory traffic, implement a 3-tier memory hierarchy simulator, and propose **ASEP (Attention-Score-Driven Eviction Policy)** as a direct improvement over standard LRU eviction.

This work addresses **Track B: Hierarchical and Tiered Memory for LLMs**.

---

## Core Contribution

Prior tiered memory proposals for LLM inference assume LRU as the eviction policy — evicting the oldest KV block when HBM pressure exceeds a threshold. **ASEP challenges this assumption.**

LRU assumes recency equals importance. This fails in transformers because:
- Attention sink tokens (first ~5% of context) are always attended to regardless of age
- Recent filler words may carry lower attention weight than older content tokens
- Long-range dependencies require retaining specific past tokens

ASEP evicts by **attention score**, not age — keeping high-importance blocks in HBM and avoiding costly DRAM promotions.

| Metric | LRU | ASEP | Improvement |
|--------|-----|------|-------------|
| Wasted evictions | 1,200 | 720 | **↓ 40%** |
| Promotion cost | 1,999 MB | 858 MB | **↓ 57%** |
| Total evictions | 14,170 | 14,170 | Equal (by design) |

> ASEP does not reduce the *number* of evictions — both policies must evict the same count given a fixed HBM budget. The improvement comes entirely from *which* blocks are evicted.

---

## Memory Hierarchy Design

We implement a 3-tier memory hierarchy sized for per-user KV cache management on a shared GPU:

| Tier | Type | Capacity | Bandwidth | Latency |
|------|------|----------|-----------|---------|
| Tier 1 | SRAM | 4 MB | 10 TB/s | 1 ns |
| Tier 2 | HBM | 48 MB* | 3.35 TB/s | 10 ns |
| Tier 3 | Host DRAM | 512 GB | 68 GB/s | 80 ns |

*48 MB models a per-user HBM allocation on a GPU serving multiple concurrent inference requests.

### Data Placement Rules

- **SRAM:** Active attention tiles for the current decode step only. Purely a compute buffer — never holds KV blocks permanently.
- **HBM:** All active KV blocks for the current conversation. Eviction triggers at 80% occupancy.
- **DRAM:** Cold KV blocks evicted from HBM, stored in INT4 compressed format (4× size reduction). Model weights never live here.

---

## ASEP — Eviction Policy Details

### Attention Score Assignment

```
Recent tokens (last 20% of context):         score = 3.0  → retain in HBM
Attention sink tokens (first 5% of context): score = 2.0  → retain in HBM
Middle / filler tokens:                       score = 0.5  → evict first
```

### Migration Rules

| Action | Trigger | Compression |
|--------|---------|-------------|
| Eviction (HBM → DRAM) | HBM occupancy > 80% | FP16 → INT4 (4× size reduction) |
| Promotion (DRAM → HBM) | Block attention score rises ≥ 2.0 | INT4 → FP16 (decompressed) |
| Prefetch (HBM → SRAM) | Before each attention layer computation | None |

---

## Memory-Centric Optimizations

Three optimizations are implemented and quantified across sequence lengths 128–4096:

### 1. KV Cache INT4 Quantization
Compresses KV cache from FP16 to INT4, reducing KV memory by **75%** at all sequence lengths and quadrupling the number of tokens that fit within a fixed HBM budget.

### 2. Attention Windowing (W = 512)
Restricts attention to the most recent 512 tokens, cutting KV read traffic by up to **87.5%** at T=4096. Savings grow proportionally with context. Trade-off: tokens outside the window are dropped.

### 3. Operator Fusion
Eliminates intermediate activation buffers between attention output and layer norm — removing 3 buffer writes per layer for a constant ~0.4% traffic reduction at all sequence lengths. Most impactful at short contexts where weight traffic dominates.

**All three optimizations are complementary.** Windowing dominates at long context; INT4 dominates for capacity; fusion applies everywhere at no cost.

---

## Implementation

### What Was Built

- **Layer-level PyTorch instrumentation** of GPT-2 medium, extracting real KV tensors layer by layer and validating the analytical memory traffic formula within 0.01 MB against real tensor sizes.
- **Analytical memory traffic model** covering attention weights, FFN weights, KV reads, and KV writes per decode step across sequence lengths 128–4096.
- **Block-level tiered memory simulator** tracking per-block metadata (tier assignment, access count, attention score, eviction history) across a full 1,000-token conversation trace.
- **Controlled policy experiment** running LRU and ASEP on identical inputs — same 1,000-token trace, same HBM budget, same 80% eviction threshold — isolating policy as the single independent variable.
- **Roofline model** confirming attention is permanently memory-bound (AI ≈ 0.003–0.03 FLOP/byte vs. H100 ridge point of ~93 FLOP/byte).
- **Energy-per-token breakdown** showing memory energy dominates compute energy by ~100× at T=4096.

### Hardware Note

All experiments run on Google Colab CPU using GPT-2 medium. The LRU vs. ASEP comparison is hardware-independent — both policies execute on identical infrastructure, so relative improvements are valid regardless of platform. Absolute latency values reflect CPU execution.

---

## Repository Contents

```
arcitecture_1.ipynb        # Main analysis and simulation notebook
measurements.json          # Analytical memory traffic per sequence length
simulator_results.csv      # Token-by-token simulation history (LRU vs ASEP)
simulator_summary.json     # Final policy comparison summary
migration_policy.txt       # Tiered memory migration policy specification
```

### Generated Outputs

| File | Description |
|------|-------------|
| `roofline_improved.png` | Roofline model — attention permanently memory-bound |
| `energy_breakdown.png` | Energy per token broken down by component |
| `opt1_kv_compression.png` | FP16 vs INT4 KV cache size across sequence lengths |
| `opt2_attention_windowing.png` | KV read traffic reduction from windowing |
| `opt3_operator_fusion.png` | Activation buffer savings from operator fusion |
| `optimization_summary.png` | All three optimizations compared side-by-side |
| `policy_comparison.png` | LRU vs ASEP — full 1,000-token simulation |
| `asep_vs_lru_slides.png` | Wasted evictions and promotion cost comparison |

---

## Key Findings

1. **Attention is permanently memory-bound.** Arithmetic intensity ranges from 0.003 to 0.03 FLOP/byte across all sequence lengths — 3,000–50,000× below the H100 ridge point. No sequence length changes this.

2. **KV cache dominates total memory traffic at long context.** At T=4096, KV reads account for the vast majority of bytes moved per decode step. Weight traffic is constant and becomes relatively minor.

3. **Memory energy dominates compute energy by ~100× at T=4096.** Architectural decisions about data movement matter far more than compute efficiency.

4. **LRU eviction is provably suboptimal.** It removes tokens by age, not importance — discarding attention sinks that will be needed again immediately and forcing expensive DRAM promotions. ASEP reduces promotion cost by 57%.

5. **All three optimizations are complementary and additive.** Applying INT4 + windowing + fusion together provides maximum benefit across all sequence lengths and memory budgets.

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 medium |
| Layers | 24 |
| Attention heads | 16 |
| Embedding dim | 1024 |
| Head dim | 64 |
| Inference precision | FP16 |
| Sequence lengths tested | 128, 256, 512, 1024, 2048, 4096 |
| Simulation length | 1,000 tokens |
| HBM eviction threshold | 80% occupancy |

---

## Requirements

```bash
pip install transformers torch numpy matplotlib pandas
```

The notebook was developed in Google Colab. Cells using `google.colab.files` for downloads can be replaced with standard file I/O when running locally.

---

## Limitations

- Attention scores in the simulator use an analytical distribution (recent / sink / middle) rather than weights extracted from live inference. A production deployment would use real attention weight distributions.
- The 48 MB per-user HBM allocation is a modeling assumption representing shared-GPU inference; a dedicated GPU would have a significantly larger budget.
- Absolute latency numbers reflect CPU simulation, not GPU hardware execution.
- Hardware validation on a GPU with a larger model (e.g., LLaMA-2-7B) would strengthen the promotion cost findings and is the natural next step for this work.
