# Memory-Centric LLM Architecture — GPT-2 Medium

### ASEP: Attention-Score-Driven Eviction Policy for Tiered KV Cache Management

---

## Project Overview

This project designs and evaluates a memory-centric architecture for LLM inference,
where the memory hierarchy — not the compute array — is the primary design driver.
Using GPT-2 medium as the target model, we analytically characterise decode-phase
memory traffic, implement a 3-tier memory hierarchy simulator, and propose **ASEP
(Attention-Score-Driven Eviction Policy)** as a direct improvement over standard LRU.

---

## Repository Structure

```
.
├── memory_model.py          # Analytical memory traffic model + optimisation calculators
├── simulator.py             # Tiered memory simulator (LRU and ASEP policies)
├── plotting.py              # All matplotlib figure generation
├── main.py                  # CLI entry point — orchestrates all modules
├── tests/
│   └── test_simulator.py    # Pytest test suite (edge cases + end-to-end)
├── graphs/                  # Generated figures (created on first run)
├── measurements.json        # Analytical traffic data (generated)
├── simulator_results.csv    # Per-token simulation trace (generated)
├── simulator_summary.json   # Policy comparison summary (generated)
└── tiered_architecture.ipynb  # Original exploration notebook (reference only)
```

---

## Requirements

```bash
pip install transformers torch numpy matplotlib pandas pytest
```

Developed and tested on Python 3.10+. The simulation runs on CPU only
(no GPU required). Google Colab cells that call `google.colab.files` have been
replaced with standard file I/O in the modular version.

---

## Quickstart

**Run the full analysis (memory model + optimisations + policy simulation + all plots):**

```bash
python main.py
```

**Run only the analytical memory traffic model:**

```bash
python main.py --mode analyze
```

**Run only the policy simulation (LRU vs ASEP) for 500 tokens:**

```bash
python main.py --mode simulate --n_tokens 500
```

**Run the test suite:**

```bash
pytest tests/ -v
```

All generated figures are saved to `./graphs/`. JSON and CSV results are saved
to the project root.

---

## Core Contribution: ASEP

Prior tiered memory proposals assume LRU as the eviction policy — evicting the
oldest KV block when HBM pressure exceeds 80%. **ASEP challenges this assumption.**

LRU assumes recency equals importance. This fails in transformers because:

- Attention sink tokens (first ~5% of context) are always attended to regardless of age
- Recent filler words may carry lower attention weight than older content tokens
- Long-range dependencies require retaining specific past tokens

ASEP evicts by **attention score**, not age — keeping high-importance blocks in HBM
and avoiding costly DRAM promotions.

| Metric             | LRU   | ASEP | Improvement |
|--------------------|-------|------|-------------|
| Wasted evictions   | 1,200 | 720  | ↓ 40%       |
| Promotion cost     | 1,999 MB | 858 MB | ↓ 57%  |
| Total evictions    | 14,170 | 14,170 | Equal (by design) |

---

## Memory Hierarchy

| Tier | Type  | Capacity | Bandwidth | Latency |
|------|-------|----------|-----------|---------|
| 1    | SRAM  | 4 MB     | 10 TB/s   | 1 ns    |
| 2    | HBM   | 48 MB*   | 3.35 TB/s | 10 ns   |
| 3    | DRAM  | 512 GB   | 68 GB/s   | 80 ns   |

*48 MB models a per-user HBM allocation on a shared GPU.

---

## Module Overview

| Module | Responsibility |
|--------|---------------|
| `memory_model.py` | Analytical formula for bytes moved, FLOPs, arithmetic intensity; INT4/windowing/fusion calculators |
| `simulator.py`    | `KVBlock`, `TieredMemoryManager`, `lru_evict`, `asep_evict`, `simulate_lru`, `simulate_asep` |
| `plotting.py`     | One function per figure; saves PNG to `graphs/` |
| `main.py`         | CLI, orchestration, JSON/CSV persistence |
| `tests/`          | Pytest cases for model correctness, eviction edge cases, end-to-end policy comparison |

---

## Key Findings

1. **Attention is permanently memory-bound.** AI ≈ 0.003–0.03 FLOP/byte across all
   sequence lengths — thousands of times below the H100 ridge point of ~93 FLOP/byte.
2. **KV cache dominates traffic at long context.** At T=4096, KV reads account for
   the vast majority of bytes moved per decode step.
3. **Memory energy dominates compute by ~100× at T=4096.**
4. **LRU is provably suboptimal.** ASEP reduces promotion cost by 57%.
5. **All three optimisations are complementary** — INT4 + windowing + fusion together
   provide maximum benefit.

---

## Limitations

- ASEP attention scores use an analytical distribution (recent / sink / middle)
  rather than weights extracted from live inference.
- The 48 MB per-user HBM allocation is a modelling assumption.
- Absolute latency values reflect CPU simulation, not GPU hardware execution.
- GPU validation on a larger model (e.g. LLaMA-2-7B) is the natural next step.

---

## External Dependencies / Credits

This project uses the following open-source libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| [PyTorch](https://pytorch.org/) | ≥ 2.0 | Model loading, tensor operations |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | ≥ 4.35 | GPT-2 medium model + tokeniser |
| [NumPy](https://numpy.org/) | ≥ 1.24 | Numerical computations |
| [Matplotlib](https://matplotlib.org/) | ≥ 3.7 | Figure generation |
| [pandas](https://pandas.pydata.org/) | ≥ 2.0 | Simulation results I/O |
| [pytest](https://pytest.org/) | ≥ 7.0 | Test suite |

GPT-2 medium model weights are loaded from the Hugging Face model hub
(`gpt2-medium`), distributed under the MIT licence by OpenAI.
