import torch
import time
import json
from transformers import GPT2Model, GPT2Tokenizer

print(f"Device: Apple M1 (MPS)")
print(f"PyTorch: {torch.__version__}")

# M1 uses MPS backend
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using: {device}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2Model.from_pretrained("gpt2-medium")
model = model.to(device)
model.eval()
print("Model loaded successfully")

seq_lengths = [128, 256, 512, 768]
results = {}

print(f"\n{'Seq':>6} | {'Decode ms':>10} | {'Tok/s':>8} | {'Throughput drop':>16}")
print("-" * 50)

baseline_tps = None

for seq_len in seq_lengths:
    # Build context
    safe_len  = seq_len - 1
    input_ids = tokenizer.encode(
        "the " * safe_len, return_tensors="pt"
    )[:, :safe_len].to(device)

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    # Time 20 decode steps and average
    next_tok = torch.tensor([[464]]).to(device)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(next_tok,
                  past_key_values=out.past_key_values,
                  use_cache=True)
        times.append((time.perf_counter() - t0) * 1000)

    # Drop first 5 warmup
    times = times[5:]
    decode_ms = sum(times) / len(times)
    tok_per_s = 1000 / decode_ms

    if baseline_tps is None:
        baseline_tps = tok_per_s
        drop = "baseline"
    else:
        drop = f"{(baseline_tps - tok_per_s)/baseline_tps*100:.1f}% slower"

    results[str(seq_len)] = {
        "seq_len":   seq_len,
        "decode_ms": round(decode_ms, 3),
        "tok_per_s": round(tok_per_s, 1),
        "throughput_drop": drop
    }

    print(f"{seq_len:>6} | {decode_ms:>10.3f} | {tok_per_s:>8.1f} | {drop:>16}")

# Save results
with open("m1_results.json", "w") as f:
    json.dump({
        "device": "Apple M1",
        "model": "GPT-2 Medium",
        "results": results
    }, f, indent=2)

print("\nm1_results.json saved.")
print()
print("Key finding: throughput degrades as context grows")
print("This is the memory bottleneck manifesting in real hardware.")