"""
memory_model.py
Analytical memory traffic model for GPT-2 medium decode phase.
Computes bytes moved, FLOPs, arithmetic intensity, and KV cache size
for a single decode step across arbitrary sequence lengths.
"""

# ── Model constants ──────────────────────────────────────────────────────────
LAYERS    = 24      # transformer layers
HEADS     = 16      # attention heads
HEAD_DIM  = 64      # dimension per head
EMBED_DIM = 1024    # total embedding dimension (HEADS * HEAD_DIM)
BYTES_FP16 = 2      # bytes per FP16 element
BYTES_INT4 = 0.5    # bytes per INT4 element

SRAM_BUDGET_BYTES = 4 * 1024 * 1024   # 4 MB SRAM budget
HBM_BUDGET_BYTES  = 48 * 1024 * 1024  # 48 MB per-user HBM allocation
DRAM_BUDGET_BYTES = 512 * 1024**3     # 512 GB host DRAM

# ── Core analytical function ─────────────────────────────────────────────────

def compute_decode_analytics(context_seq_len: int) -> dict:
    """
    Analytically compute all memory traffic for one GPT-2 medium decode step.

    Args:
        context_seq_len: Number of past tokens in the KV cache (T).

    Returns:
        Dictionary with bytes moved, FLOPs, arithmetic intensity,
        KV cache size, and tier assignment.
    """
    L, H, D_h, d, B = LAYERS, HEADS, HEAD_DIM, EMBED_DIM, BYTES_FP16
    T = context_seq_len

    # ── Weight bytes (constant — independent of T) ────────────────────────
    # Attention: Q, K, V, O projections — each (d × d), 4 per layer
    attn_weight_bytes = L * 4 * d * d * B
    # FFN: fc_in (d → 4d) and fc_out (4d → d)
    ffn_weight_bytes  = L * (d * 4*d + 4*d * d) * B
    weight_bytes_total = attn_weight_bytes + ffn_weight_bytes

    # ── KV cache bytes (grows linearly with T) ────────────────────────────
    kv_read_bytes  = L * 2 * T * H * D_h * B   # read all T past K and V
    kv_write_bytes = L * 2 * H * D_h * B        # write 1 new token's K and V
    kv_cache_bytes = L * 2 * T * H * D_h * B    # total KV stored for T tokens

    total_bytes = weight_bytes_total + kv_read_bytes + kv_write_bytes

    # ── FLOPs ─────────────────────────────────────────────────────────────
    # Attention: QK^T + AV across all layers (factor 2 for multiply-add)
    attn_flops = L * 2 * 2 * 1 * T * d
    # FFN: two matmuls per layer, one token
    ffn_flops  = L * 2 * (d * 4*d + 4*d * d)
    total_flops = attn_flops + ffn_flops

    # ── Arithmetic intensity ──────────────────────────────────────────────
    overall_ai = total_flops / total_bytes
    attn_ai    = attn_flops / (attn_weight_bytes + kv_read_bytes)
    ffn_ai     = ffn_flops  / ffn_weight_bytes

    # ── Tier assignment based on KV cache size ────────────────────────────
    if kv_cache_bytes <= SRAM_BUDGET_BYTES:
        sram_miss_rate = 0.0
        kv_tier = "SRAM"
    else:
        sram_miss_rate = (kv_cache_bytes - SRAM_BUDGET_BYTES) / kv_cache_bytes
        kv_tier = "HBM"

    return {
        "seq_len":             T,
        "attn_weight_bytes":   attn_weight_bytes,
        "ffn_weight_bytes":    ffn_weight_bytes,
        "weight_bytes_total":  weight_bytes_total,
        "kv_read_bytes":       kv_read_bytes,
        "kv_write_bytes":      kv_write_bytes,
        "kv_cache_bytes":      kv_cache_bytes,
        "total_bytes":         total_bytes,
        "attn_flops":          attn_flops,
        "ffn_flops":           ffn_flops,
        "total_flops":         total_flops,
        "overall_ai":          overall_ai,
        "attn_ai":             attn_ai,
        "ffn_ai":              ffn_ai,
        "kv_tier":             kv_tier,
        "sram_miss_rate":      round(sram_miss_rate, 4),
        "kv_cache_mb":         round(kv_cache_bytes / 1e6, 3),
        "total_bytes_mb":      round(total_bytes / 1e6, 2),
        "bandwidth_limited_stage": "attention" if attn_ai < ffn_ai else "ffn",
    }


def run_all_seq_lengths(seq_lengths=None) -> dict:
    """
    Run compute_decode_analytics across a list of sequence lengths.

    Args:
        seq_lengths: List of ints. Defaults to [128, 256, 512, 1024, 2048, 4096].

    Returns:
        Dict mapping str(seq_len) → analytics dict.
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    return {str(T): compute_decode_analytics(T) for T in seq_lengths}


# ── Optimisation calculators ─────────────────────────────────────────────────

def kv_int4_savings(seq_len: int) -> dict:
    """Compute KV cache size reduction from FP16 → INT4 quantization."""
    L, H, D_h = LAYERS, HEADS, HEAD_DIM
    kv_fp16 = L * 2 * seq_len * H * D_h * BYTES_FP16
    kv_int4 = L * 2 * seq_len * H * D_h * BYTES_INT4
    return {
        "seq_len":        seq_len,
        "kv_fp16_mb":     kv_fp16 / 1e6,
        "kv_int4_mb":     kv_int4 / 1e6,
        "hbm_freed_mb":   (kv_fp16 - kv_int4) / 1e6,
        "reduction_pct":  75.0,
        "tokens_with_int4": seq_len * 4,
    }


def attention_windowing_savings(seq_len: int, window: int = 512) -> dict:
    """Compute KV read traffic reduction from attention windowing."""
    L, H, D_h, B = LAYERS, HEADS, HEAD_DIM, BYTES_FP16
    full_traffic   = 2 * seq_len * L * H * D_h * B
    eff_window     = min(seq_len, window)
    window_traffic = 2 * eff_window * L * H * D_h * B
    saved_mb  = (full_traffic - window_traffic) / 1e6
    saved_pct = (full_traffic - window_traffic) / full_traffic * 100 if full_traffic > 0 else 0
    return {
        "seq_len":    seq_len,
        "window":     window,
        "full_mb":    full_traffic / 1e6,
        "window_mb":  window_traffic / 1e6,
        "saved_mb":   saved_mb,
        "saved_pct":  saved_pct,
    }


def operator_fusion_savings(seq_len: int, total_bytes: int) -> dict:
    """
    Compute activation buffer savings from operator fusion.
    Fusion eliminates 3 intermediate buffers per layer between
    attention output and layer norm.
    """
    BUFFERS_ELIMINATED = 3
    single_buffer = 1 * EMBED_DIM * BYTES_FP16
    total_saved   = LAYERS * BUFFERS_ELIMINATED * single_buffer
    fused_bytes   = total_bytes - total_saved
    return {
        "seq_len":        seq_len,
        "unfused_mb":     total_bytes / 1e6,
        "fused_mb":       fused_bytes / 1e6,
        "saved_mb":       total_saved / 1e6,
        "reduction_pct":  (total_saved / total_bytes) * 100,
    }
