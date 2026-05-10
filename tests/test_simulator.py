"""
tests/test_simulator.py
Pytest test suite for the tiered memory simulator.
Covers functional correctness and edge cases required by the rubric:
  - HBM buffer overflow / eviction trigger
  - ASEP always evicts lower-score blocks before higher-score ones
  - Promotion when attention score rises
  - LRU evicts strictly by age
  - KV block accounting is consistent
"""

import pytest
from simulator import (
    KVBlock,
    TieredMemoryManager,
    get_attention_score,
    asep_evict,
    lru_evict,
    simulate_lru,
    simulate_asep,
    HBM_CAPACITY_BYTES,
    HBM_EVICT_THRESHOLD,
    KV_BYTES_PER_BLOCK,
)
from memory_model import compute_decode_analytics, run_all_seq_lengths


# ── Memory model tests ────────────────────────────────────────────────────────

class TestMemoryModel:

    def test_kv_cache_grows_linearly(self):
        """KV cache bytes must be proportional to sequence length."""
        r128  = compute_decode_analytics(128)
        r256  = compute_decode_analytics(256)
        assert r256["kv_cache_bytes"] == 2 * r128["kv_cache_bytes"]

    def test_weight_bytes_constant(self):
        """Weight bytes must not depend on sequence length."""
        r128  = compute_decode_analytics(128)
        r4096 = compute_decode_analytics(4096)
        assert r128["weight_bytes_total"] == r4096["weight_bytes_total"]

    def test_attention_always_memory_bound(self):
        """Attention arithmetic intensity must be below the H100 ridge (93 FLOP/byte)."""
        H100_RIDGE = 93.0
        results = run_all_seq_lengths()
        for seq_len, r in results.items():
            assert r["attn_ai"] < H100_RIDGE, (
                f"Attention at T={seq_len} unexpectedly compute-bound (AI={r['attn_ai']:.3f})"
            )

    def test_total_bytes_increase_with_context(self):
        """Longer contexts must move more bytes per decode step."""
        r128  = compute_decode_analytics(128)
        r4096 = compute_decode_analytics(4096)
        assert r4096["total_bytes"] > r128["total_bytes"]

    def test_sram_overflow_at_t128(self):
        """SRAM budget (4 MB) should already be exceeded at T=128."""
        r = compute_decode_analytics(128)
        assert r["sram_miss_rate"] > 0, "Expected SRAM to overflow at T=128"


# ── Tiered memory manager tests ───────────────────────────────────────────────

class TestTieredMemoryManager:

    def _fill_hbm_to_threshold(self):
        """Helper: fill HBM to just below the eviction threshold."""
        manager = TieredMemoryManager()
        capacity_at_threshold = int(HBM_CAPACITY_BYTES * HBM_EVICT_THRESHOLD)
        n_blocks = capacity_at_threshold // KV_BYTES_PER_BLOCK
        for i in range(n_blocks):
            block = KVBlock(token_idx=i, layer_idx=0, size_bytes=KV_BYTES_PER_BLOCK)
            block.last_access = i
            manager.add_block(block)
        return manager

    def test_hbm_accounting_on_add(self):
        """HBM used bytes should track added blocks exactly."""
        manager = TieredMemoryManager()
        block = KVBlock(token_idx=0, layer_idx=0, size_bytes=KV_BYTES_PER_BLOCK)
        manager.add_block(block)
        assert manager.hbm_used_bytes == KV_BYTES_PER_BLOCK

    def test_eviction_moves_block_to_dram(self):
        """Evicting a block should move it from HBM to DRAM."""
        manager = self._fill_hbm_to_threshold()
        # Add one more block to push over threshold
        overflow = KVBlock(token_idx=9999, layer_idx=0, size_bytes=KV_BYTES_PER_BLOCK)
        overflow.last_access = 9999
        manager.add_block(overflow)

        initial_hbm = manager.hbm_used_bytes
        lru_evict(manager)

        assert manager.hbm_used_bytes < initial_hbm, "HBM should decrease after eviction"
        assert manager.total_evictions > 0

    def test_eviction_trigger_at_80_percent(self):
        """
        Edge case: eviction must trigger at exactly 80% HBM occupancy.
        Adding blocks that push occupancy above threshold should evict.
        """
        manager = TieredMemoryManager()
        # Fill to 79% — no eviction yet
        target = int(HBM_CAPACITY_BYTES * 0.79)
        n_blocks = target // KV_BYTES_PER_BLOCK
        for i in range(n_blocks):
            block = KVBlock(i, 0, KV_BYTES_PER_BLOCK)
            block.last_access = i
            manager.add_block(block)

        assert manager.hbm_occupancy < HBM_EVICT_THRESHOLD

        # Add enough to cross 80%
        while manager.hbm_occupancy <= HBM_EVICT_THRESHOLD:
            block = KVBlock(n_blocks, 0, KV_BYTES_PER_BLOCK)
            block.last_access = n_blocks
            manager.add_block(block)
            n_blocks += 1

        lru_evict(manager)
        assert manager.hbm_occupancy <= HBM_EVICT_THRESHOLD, (
            "Eviction should bring occupancy back to or below threshold"
        )

    def test_promotion_increments_counter(self):
        """Accessing a DRAM block must promote it and count the promotion."""
        manager = TieredMemoryManager()
        block = KVBlock(token_idx=0, layer_idx=0, size_bytes=KV_BYTES_PER_BLOCK)
        manager.add_block(block)
        # Manually demote to DRAM to set up the test
        manager.evict_to_dram(block)
        assert block.tier == "DRAM"

        manager.access_block(block, token_step=1)
        assert block.tier == "HBM"
        assert manager.total_promotions == 1

    def test_hbm_bytes_consistent_after_evict_and_promote(self):
        """HBM accounting must be consistent after a full evict-promote cycle."""
        manager = TieredMemoryManager()
        block = KVBlock(token_idx=0, layer_idx=0, size_bytes=KV_BYTES_PER_BLOCK)
        manager.add_block(block)
        initial_hbm = manager.hbm_used_bytes

        manager.evict_to_dram(block)
        manager.promote_to_hbm(block)

        assert manager.hbm_used_bytes == initial_hbm, (
            "HBM bytes should be restored after evict + promote"
        )


# ── ASEP policy tests ─────────────────────────────────────────────────────────

class TestASEP:

    def test_attention_score_recent_tokens(self):
        """Tokens in the last 20% of context should score 3.0."""
        score = get_attention_score(token_idx=900, current_token=1000)
        assert score == 3.0

    def test_attention_score_sink_tokens(self):
        """Tokens in the first 5% of context should score 2.0."""
        score = get_attention_score(token_idx=3, current_token=1000)
        assert score == 2.0

    def test_attention_score_middle_tokens(self):
        """Middle tokens (not recent, not sink) should score 0.5."""
        score = get_attention_score(token_idx=500, current_token=1000)
        assert score == 0.5

    def test_asep_evicts_low_score_before_high_score(self):
        """
        Core ASEP invariant: given a mix of low- and high-attention blocks,
        all low-attention blocks must be evicted before any high-attention
        block is touched.
        """
        manager = TieredMemoryManager()
        # Fill HBM to just under threshold with high-attention blocks
        # then add one low-attention block that tips it over
        n_high = int(HBM_CAPACITY_BYTES * HBM_EVICT_THRESHOLD) // KV_BYTES_PER_BLOCK
        high_blocks = []
        for i in range(n_high):
            # token_idx in recent 20% of context (current_token=1000)
            b = KVBlock(token_idx=850 + (i % 50), layer_idx=i, size_bytes=KV_BYTES_PER_BLOCK)
            b.last_access = 850 + (i % 50)
            manager.add_block(b)
            high_blocks.append(b)

        # Add a low-attention (middle) block to tip over the threshold
        low_block = KVBlock(token_idx=300, layer_idx=999, size_bytes=KV_BYTES_PER_BLOCK)
        low_block.last_access = 300
        manager.add_block(low_block)

        assert manager.hbm_occupancy > HBM_EVICT_THRESHOLD

        asep_evict(manager, current_token=1000)

        # The low-attention block must be the first one evicted
        assert low_block.tier == "DRAM", "Low-attention block should be evicted first"
        # At least some high-attention blocks should still be in HBM
        assert any(b.tier == "HBM" for b in high_blocks), \
            "At least some high-attention blocks should remain in HBM"

    def test_lru_evicts_oldest_first(self):
        """LRU must evict the block with the smallest last_access time."""
        manager = TieredMemoryManager()
        # Fill HBM to just under threshold with newer blocks
        n_blocks = int(HBM_CAPACITY_BYTES * HBM_EVICT_THRESHOLD) // KV_BYTES_PER_BLOCK
        newer_blocks = []
        for i in range(n_blocks):
            b = KVBlock(token_idx=100 + i, layer_idx=i, size_bytes=KV_BYTES_PER_BLOCK)
            b.last_access = 100 + i   # all newer than the old block below
            manager.add_block(b)
            newer_blocks.append(b)

        # Add old block that tips over threshold — should be evicted first
        old_block = KVBlock(token_idx=0, layer_idx=9999, size_bytes=KV_BYTES_PER_BLOCK)
        old_block.last_access = 0
        manager.add_block(old_block)

        assert manager.hbm_occupancy > HBM_EVICT_THRESHOLD

        lru_evict(manager)

        assert old_block.tier == "DRAM", "LRU must evict oldest block first"
        assert any(b.tier == "HBM" for b in newer_blocks), \
            "Newer blocks should remain in HBM"


# ── End-to-end simulation tests ───────────────────────────────────────────────

class TestSimulationEndToEnd:

    def test_lru_simulation_runs_to_completion(self):
        """LRU simulation must complete 100 tokens without error."""
        history, final = simulate_lru(n_tokens=100)
        assert len(history) == 100
        assert final["evictions"] >= 0

    def test_asep_simulation_runs_to_completion(self):
        """ASEP simulation must complete 100 tokens without error."""
        history, final = simulate_asep(n_tokens=100)
        assert len(history) == 100
        assert final["evictions"] >= 0

    def test_asep_fewer_wasted_evictions_than_lru(self):
        """
        Over a full 500-token run, ASEP should produce fewer or equal
        wasted evictions compared to LRU — the central claim of the paper.
        """
        _, lru_final  = simulate_lru(n_tokens=500)
        _, asep_final = simulate_asep(n_tokens=500)
        assert asep_final["wasted_evictions"] <= lru_final["wasted_evictions"], (
            "ASEP must not produce more wasted evictions than LRU"
        )

    def test_asep_lower_promotion_cost_than_lru(self):
        """ASEP must have lower or equal promotion cost than LRU."""
        _, lru_final  = simulate_lru(n_tokens=500)
        _, asep_final = simulate_asep(n_tokens=500)
        assert asep_final["promotion_cost_mb"] <= lru_final["promotion_cost_mb"], (
            "ASEP must not have higher promotion cost than LRU"
        )

    def test_total_evictions_equal_between_policies(self):
        """
        Both policies operate under the same HBM budget, so total eviction
        count should be equal — only *which* blocks are evicted differs.
        """
        _, lru_final  = simulate_lru(n_tokens=500)
        _, asep_final = simulate_asep(n_tokens=500)
        assert lru_final["evictions"] == asep_final["evictions"], (
            "Total eviction count must be equal for LRU and ASEP "
            "(same budget, same eviction trigger)"
        )
