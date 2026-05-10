"""
simulator.py
Block-level tiered memory simulator for LLM KV cache management.
Implements LRU and ASEP (Attention-Score-Driven Eviction Policy).

Memory hierarchy:
  Tier 1 — SRAM  :  4 MB,  10 TB/s,  1 ns  (active attention tiles)
  Tier 2 — HBM   : 48 MB,  3.35 TB/s, 10 ns (active KV blocks)
  Tier 3 — DRAM  : 512 GB, 68 GB/s,  80 ns  (evicted KV blocks, INT4)
"""

from dataclasses import dataclass, field
from typing import List

from memory_model import LAYERS, HEADS, HEAD_DIM, BYTES_FP16

# ── Hardware constants ────────────────────────────────────────────────────────
HBM_CAPACITY_BYTES  = 48 * 1024 * 1024    # 48 MB per-user HBM allocation
HBM_EVICT_THRESHOLD = 0.80                 # trigger eviction at 80% occupancy
DRAM_BW_BYTES_PER_NS = 68                  # 68 GB/s ≈ 68 bytes/ns
HBM_BW_BYTES_PER_NS  = 3350               # 3.35 TB/s ≈ 3350 bytes/ns
INT4_COMPRESSION     = 4                   # INT4 is 4× smaller than FP16
KV_BYTES_PER_BLOCK   = 2 * 1 * HEADS * HEAD_DIM * BYTES_FP16  # one token, all heads


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class KVBlock:
    """Represents the KV cache for one token across all attention heads."""
    token_idx:       int
    layer_idx:       int
    size_bytes:      int
    tier:            str   = "HBM"   # "HBM" or "DRAM"
    last_access:     int   = 0
    access_count:    int   = 0
    eviction_count:  int   = 0
    attention_score: float = 3.0     # new tokens start as high-attention


class TieredMemoryManager:
    """
    Manages KV blocks across the three-tier memory hierarchy.
    Tracks per-block metadata and computes simulation statistics.
    """

    def __init__(self):
        self.blocks: List[KVBlock] = []
        self.hbm_used_bytes = 0
        self.dram_used_bytes = 0
        self.total_evictions    = 0
        self.wasted_evictions   = 0   # evictions of high-attention blocks
        self.total_promotions   = 0
        self.promotion_cost_bytes = 0
        self.total_bytes_moved  = 0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def hbm_occupancy(self) -> float:
        return self.hbm_used_bytes / HBM_CAPACITY_BYTES

    @property
    def hbm_used_mb(self) -> float:
        return self.hbm_used_bytes / 1e6

    @property
    def dram_used_mb(self) -> float:
        return self.dram_used_bytes / 1e6

    # ── Block management ──────────────────────────────────────────────────

    def add_block(self, block: KVBlock) -> None:
        """Place a new KV block into HBM."""
        self.blocks.append(block)
        self.hbm_used_bytes += block.size_bytes

    def get_hbm_blocks(self) -> List[KVBlock]:
        return [b for b in self.blocks if b.tier == "HBM"]

    def get_dram_blocks(self) -> List[KVBlock]:
        return [b for b in self.blocks if b.tier == "DRAM"]

    def evict_to_dram(self, block: KVBlock) -> None:
        """Move a block from HBM to DRAM with INT4 compression."""
        compressed_size = block.size_bytes // INT4_COMPRESSION
        # Track wasted eviction if we evict a high-attention block
        if block.attention_score >= 2.0:
            self.wasted_evictions += 1
        block.tier = "DRAM"
        block.eviction_count += 1
        self.hbm_used_bytes  -= block.size_bytes
        self.dram_used_bytes += compressed_size
        self.total_evictions += 1
        self.total_bytes_moved += compressed_size

    def promote_to_hbm(self, block: KVBlock) -> None:
        """Move a block from DRAM back to HBM (decompressed)."""
        compressed_size = block.size_bytes // INT4_COMPRESSION
        block.tier = "HBM"
        self.dram_used_bytes  -= compressed_size
        self.hbm_used_bytes   += block.size_bytes
        self.total_promotions += 1
        promotion_bytes = block.size_bytes  # cost is the full FP16 read
        self.promotion_cost_bytes += promotion_bytes
        self.total_bytes_moved    += promotion_bytes

    def access_block(self, block: KVBlock, token_step: int) -> int:
        """
        Access a KV block; promote from DRAM if needed.
        Returns latency in nanoseconds.
        """
        block.last_access  = token_step
        block.access_count += 1
        if block.tier == "DRAM":
            self.promote_to_hbm(block)
            return 80  # DRAM latency ns
        return 10   # HBM latency ns

    def snapshot(self, token: int) -> dict:
        """Return a stats snapshot for the current simulation step."""
        return {
            "token":              token,
            "hbm_used_mb":        round(self.hbm_used_mb, 4),
            "dram_used_mb":       round(self.dram_used_mb, 4),
            "hbm_occupancy":      round(self.hbm_occupancy, 4),
            "hbm_blocks":         len(self.get_hbm_blocks()),
            "dram_blocks":        len(self.get_dram_blocks()),
            "evictions":          self.total_evictions,
            "wasted_evictions":   self.wasted_evictions,
            "promotion_cost_mb":  round(self.promotion_cost_bytes / 1e6, 4),
            "latency_ns":         0,         # filled in by simulate_*
            "bytes_moved_gb":     round(self.total_bytes_moved / 1e9, 6),
        }


# ── Eviction policy: ASEP ─────────────────────────────────────────────────────

def get_attention_score(token_idx: int, current_token: int) -> float:
    """
    Simulated attention score for a token at decode step `current_token`.

    Based on empirical observations from attention head research:
      - Recent tokens (last 20% of context): score 3.0  → retain in HBM
      - Attention sink tokens (first 5%):    score 2.0  → retain in HBM
      - Middle / filler tokens:              score 0.5  → evict first

    Args:
        token_idx:     Position of the token in the sequence.
        current_token: Current decode step (total tokens generated so far).

    Returns:
        Float attention score in {0.5, 2.0, 3.0}.
    """
    age = current_token - token_idx
    if age <= current_token * 0.20:                  # recent 20%
        return 3.0
    if token_idx <= current_token * 0.05:            # attention sink 5%
        return 2.0
    return 0.5                                        # middle filler


def asep_evict(manager: TieredMemoryManager, current_token: int) -> None:
    """
    ASEP eviction: evict blocks with the lowest attention score first.
    Ties broken by LRU (oldest last-access time).

    This is the core contribution vs. LRU:
    LRU evicts by age; ASEP evicts by importance — keeping attention
    sinks and recent tokens in HBM regardless of insertion order.
    """
    while manager.hbm_occupancy > HBM_EVICT_THRESHOLD:
        hbm_blocks = manager.get_hbm_blocks()
        if not hbm_blocks:
            break
        for block in hbm_blocks:
            block.attention_score = get_attention_score(block.token_idx, current_token)
        # Sort: evict lowest attention score first; break ties by age (oldest)
        hbm_blocks.sort(key=lambda b: (b.attention_score, b.last_access))
        manager.evict_to_dram(hbm_blocks[0])


def lru_evict(manager: TieredMemoryManager) -> None:
    """
    Standard LRU eviction: evict the block with the oldest last-access time.
    Used as the baseline policy against ASEP.
    """
    while manager.hbm_occupancy > HBM_EVICT_THRESHOLD:
        hbm_blocks = manager.get_hbm_blocks()
        if not hbm_blocks:
            break
        hbm_blocks.sort(key=lambda b: b.last_access)
        manager.evict_to_dram(hbm_blocks[0])


# ── Simulation runners ────────────────────────────────────────────────────────

def simulate_lru(n_tokens: int = 1000) -> tuple:
    """
    Simulate a full conversation using LRU eviction.

    Args:
        n_tokens: Number of decode steps to simulate.

    Returns:
        (history, final_stats) where history is a list of per-step snapshots.
    """
    manager = TieredMemoryManager()
    history = []

    for token_idx in range(1, n_tokens + 1):
        # Add new KV blocks for this token (one per layer)
        for layer in range(LAYERS):
            block = KVBlock(token_idx, layer, KV_BYTES_PER_BLOCK)
            block.last_access = token_idx
            manager.add_block(block)

        # Evict if HBM is over threshold
        lru_evict(manager)

        # Access cost: read all HBM blocks (decode attention)
        latency = sum(
            manager.access_block(b, token_idx)
            for b in manager.get_hbm_blocks()
        )
        snap = manager.snapshot(token_idx)
        snap["latency_ns"] = latency
        history.append(snap)

    return history, history[-1]


def simulate_asep(n_tokens: int = 1000) -> tuple:
    """
    Simulate a full conversation using ASEP eviction.

    Args:
        n_tokens: Number of decode steps to simulate.

    Returns:
        (history, final_stats) where history is a list of per-step snapshots.
    """
    manager = TieredMemoryManager()
    history = []

    for token_idx in range(1, n_tokens + 1):
        for layer in range(LAYERS):
            block = KVBlock(token_idx, layer, KV_BYTES_PER_BLOCK)
            block.attention_score = 3.0   # new tokens start as high-attention
            block.last_access = token_idx
            manager.add_block(block)

        asep_evict(manager, token_idx)

        latency = sum(
            manager.access_block(b, token_idx)
            for b in manager.get_hbm_blocks()
        )
        snap = manager.snapshot(token_idx)
        snap["latency_ns"] = latency
        history.append(snap)

    return history, history[-1]
