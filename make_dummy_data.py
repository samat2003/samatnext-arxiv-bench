#!/usr/bin/env python3
"""
make_dummy_data.py
Generate a synthetic tokenized dataset that passes every validation check
inside production_train_550m.py (MemmapTokenStream + load_and_validate_cooldown_metadata).

Run from the project root:
    cd ~/samatnext-CL/samatnext-CL
    source ../venv/bin/activate
    python make_dummy_data.py

What this creates
-----------------
data/
    shard_0000.bin   -- uint16 token IDs
    shard_0001.bin
    shard_0002.bin
    metadata.json    -- cooldown manifest the trainer parses

Validator chain this must satisfy
----------------------------------
1.  metadata["phases"] is a list containing one dict with phase_id == "1F"
2.  phase["status"] == "complete"
3.  phase has "start_token", "end_token", "tokens_written"
4.  metadata["cooldown_start_token"] == phase["start_token"]
5.  start > 0, end > start, written == end - start
6.  metadata["tokens_written"] (total) > 0, end <= total
7.  sum(shard_*.bin byte sizes // 2) == metadata["tokens_written"]  <-- hard equality
"""
from __future__ import annotations

import hashlib
import json
import os
import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generate a synthetic tokenized dataset that passes validation checks."
)
parser.add_argument(
    "--num-shards",
    type=int,
    default=3,
    help="Number of shards to generate (default: 3)."
)
parser.add_argument(
    "--tokens-per-shard",
    type=int,
    default=None,
    help="Tokens per shard. If not set, total tokens default to 110,067,776 distributed across shards."
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="data",
    help="Output directory for the generated data (default: data)."
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants mirrored exactly from production_train_550m.py
# (do not change these without also changing the trainer constants)
# ---------------------------------------------------------------------------
VOCAB_SIZE: int       = 50304   # must match CONFIG["vocab_size"]
BATCH_SIZE: int       = 32
SEQ_LEN: int          = 512
TOKENS_PER_STEP: int  = BATCH_SIZE * SEQ_LEN          # 16 384

# Stage-2C training budget from the trainer (110 M tokens total)
TOTAL_TRAINING_STEPS: int  = 6_714
PRE_COOLDOWN_STEPS: int    = 6_497
COOLDOWN_STEPS: int        = 217

# Total tokens the training run expects to see across all shards
if args.tokens_per_shard is not None:
    TOTAL_TOKENS: int = args.num_shards * args.tokens_per_shard
    COOLDOWN_START_TOKEN: int = int(TOTAL_TOKENS * (PRE_COOLDOWN_STEPS / TOTAL_TRAINING_STEPS))
    COOLDOWN_END_TOKEN: int   = TOTAL_TOKENS
else:
    TOTAL_TOKENS: int = TOTAL_TRAINING_STEPS * TOKENS_PER_STEP   # 110 067 776
    COOLDOWN_START_TOKEN: int = PRE_COOLDOWN_STEPS * TOKENS_PER_STEP   # 106 497 024
    COOLDOWN_END_TOKEN: int   = COOLDOWN_START_TOKEN + COOLDOWN_STEPS * TOKENS_PER_STEP  # 110 049 280

# ---------------------------------------------------------------------------
# Derived check: the trainer asserts
#   sum(shard_token_counts) == cooldown.total_tokens
# cooldown.total_tokens is metadata["tokens_written"] (the top-level key).
# The phase["end_token"] must be <= that total too.
# ---------------------------------------------------------------------------
assert COOLDOWN_END_TOKEN <= TOTAL_TOKENS, (
    f"Cooldown end ({COOLDOWN_END_TOKEN}) exceeds total tokens ({TOTAL_TOKENS})"
)
assert COOLDOWN_START_TOKEN > 0
assert COOLDOWN_END_TOKEN > COOLDOWN_START_TOKEN

# ---------------------------------------------------------------------------
# Shard layout
# ---------------------------------------------------------------------------
N_SHARDS: int = args.num_shards
# Divide total tokens evenly; last shard absorbs the remainder so that
# sum(shard_token_counts) == TOTAL_TOKENS exactly.
base_tokens_per_shard: int = TOTAL_TOKENS // N_SHARDS
shard_sizes: list[int] = [base_tokens_per_shard] * N_SHARDS
shard_sizes[-1] += TOTAL_TOKENS - sum(shard_sizes)   # absorb remainder

assert sum(shard_sizes) == TOTAL_TOKENS, (
    f"Shard size mismatch: {sum(shard_sizes)} != {TOTAL_TOKENS}"
)

# ---------------------------------------------------------------------------
# Build the metadata document
# ---------------------------------------------------------------------------
metadata: dict = {
    # Top-level total token count -- must equal sum of all shard byte sizes // 2
    "tokens_written": TOTAL_TOKENS,
    # Required by load_and_validate_cooldown_metadata
    "cooldown_start_token": COOLDOWN_START_TOKEN,
    # Human-readable extras (not validated, but good to have)
    "vocab_size": VOCAB_SIZE,
    "seq_len": SEQ_LEN,
    "batch_size": BATCH_SIZE,
    "n_shards": N_SHARDS,
    "shard_sizes_tokens": shard_sizes,
    "note": "Synthetic dummy data generated by make_dummy_data.py",
    # The trainer searches for phase_id == "1F" in this list
    "phases": [
        {
            "phase_id": "1F",
            "status": "complete",          # must be "complete"
            "start_token": COOLDOWN_START_TOKEN,
            "end_token": COOLDOWN_END_TOKEN,
            # tokens_written must equal end_token - start_token exactly
            "tokens_written": COOLDOWN_END_TOKEN - COOLDOWN_START_TOKEN,
        }
    ],
}

# ---------------------------------------------------------------------------
# Write everything
# ---------------------------------------------------------------------------
DATA_DIR = Path(args.out_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed=42)

total_written = 0
for i, n_tokens in enumerate(shard_sizes):
    path = DATA_DIR / f"shard_{i:04d}.bin"
    if path.exists():
        existing = path.stat().st_size // 2
        if existing == n_tokens:
            print(f"  [skip]  {path.name}  ({n_tokens:,} tokens already present)")
            total_written += n_tokens
            continue
        print(f"  [overwrite]  {path.name}  (size mismatch: {existing} vs {n_tokens})")

    print(f"  [write]  {path.name}  ({n_tokens:,} tokens, {n_tokens * 2 / 1e6:.1f} MB) ...",
          end="", flush=True)
    tokens = rng.integers(0, VOCAB_SIZE, size=n_tokens, dtype=np.uint16)
    tokens.tofile(path)
    total_written += n_tokens
    print(" done")

assert total_written == TOTAL_TOKENS, (
    f"Written {total_written} tokens but expected {TOTAL_TOKENS}"
)

metadata_path = DATA_DIR / "metadata.json"
raw = json.dumps(metadata, indent=2).encode("utf-8")
metadata_path.write_bytes(raw)
sha256 = hashlib.sha256(raw).hexdigest()

# ---------------------------------------------------------------------------
# Final validation: run the same checks the trainer will run
# ---------------------------------------------------------------------------
print("\nRunning pre-flight validation...")

phases = metadata["phases"]
assert isinstance(phases, list), "phases must be a list"
phase = next((p for p in phases if p.get("phase_id") == "1F"), None)
assert phase is not None, "phase_id 1F not found"
assert phase.get("status") == "complete", "phase 1F must be complete"
assert phase["tokens_written"] == phase["end_token"] - phase["start_token"]
assert metadata["cooldown_start_token"] == phase["start_token"]
assert phase["start_token"] > 0
assert phase["end_token"] > phase["start_token"]
assert metadata["tokens_written"] > 0
assert phase["end_token"] <= metadata["tokens_written"]

shard_byte_total = sum(
    (DATA_DIR / f"shard_{i:04d}.bin").stat().st_size
    for i in range(N_SHARDS)
)
assert shard_byte_total // 2 == TOTAL_TOKENS, (
    f"Shard byte sum {shard_byte_total // 2} != expected {TOTAL_TOKENS}"
)

print("  metadata.json  OK")
print("  shard byte sum matches metadata['tokens_written']  OK")
print("  cooldown_start_token / phase 1F boundaries  OK")
print()
print("=" * 60)
print("SUCCESS — data/ is ready for production_train_550m.py")
print(f"  Total tokens  : {TOTAL_TOKENS:,}")
print(f"  Shards        : {N_SHARDS} x shard_XXXX.bin")
print(f"  Cooldown start: token {COOLDOWN_START_TOKEN:,}  (step {PRE_COOLDOWN_STEPS})")
print(f"  Cooldown end  : token {COOLDOWN_END_TOKEN:,}  (step {TOTAL_TRAINING_STEPS})")
print(f"  metadata sha256: {sha256[:16]}...")
print()
print("Run the smoke test (no GPU memory committed, synthetic tokens):")
print("  python production_train_550m.py --smoke-test --smoke-full-model \\")
print("    --activation-checkpointing --no-compile \\")
print("    --experimental-lowbit-linears")
print()
print("Or start a real fresh run:")
print("  LD_PRELOAD=/usr/local/cuda/lib64/libcublasLt.so.13:/usr/local/cuda/lib64/libcublas.so.13 \\")
print("  python production_train_550m.py --fresh-run --data-dir data \\")
print("    --activation-checkpointing --experimental-lowbit-linears")
print("=" * 60)
