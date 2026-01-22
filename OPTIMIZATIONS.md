# Kernel Optimization Documentation

## Overview
This document tracks optimizations made to the VLIW SIMD kernel for the performance take-home.

**Current Result: 3,302 cycles (44.7x speedup from baseline)**

**Tests Passing: 7/8**

---

## Summary of Major Optimizations

| Optimization | Cycles | Speedup | Cumulative |
|--------------|--------|---------|------------|
| Baseline | 147,734 | - | 1x |
| 1. SIMD (VALU) | 25,677 | 5.75x | 5.75x |
| 2. VLIW Packing | 13,888 | 1.85x | 10.6x |
| 3. Op Batching | 6,368 | 2.18x | 23.2x |
| 4. Eliminate vselects | 5,536 | 1.15x | 26.7x |
| 5-8. Various caching | 4,881 | 1.13x | 30.3x |
| 9. Software Pipelining | 4,418 | 1.10x | 33.4x |
| 10. multiply_add for hash | 3,639 | 1.21x | 40.6x |
| 11. Multi-round chunks | 3,302 | 1.10x | **44.7x** |

---

## Key Optimization: Multi-Round Chunk Processing

### The Breakthrough
Instead of loading/storing idx/val every round, process vectors in chunks across ALL rounds:

```
Before (per round):
  Load idx, val from memory
  Compute
  Store idx, val to memory

After (per chunk):
  Load idx, val ONCE
  For all 16 rounds: Compute
  Store idx, val ONCE
```

### Impact
- Eliminated 896 redundant load ops (14 rounds × 64 vload ops)
- Eliminated 960 redundant store ops (15 rounds × 64 vstore ops)
- Total reduction: 1,856 memory ops

### Current Operation Counts
- Load ops: 3,200 → 1,600 min cycles at 2/cycle
- VALU ops: 10,328 → 1,721 min cycles at 6/cycle
- Store ops: 64 → 32 min cycles at 2/cycle
- **Theoretical minimum: 1,721 cycles (VALU-bound)**

---

## Key Optimization: Hash multiply_add

### Insight
Hash stages 0, 2, 4 have pattern: `result = (val + const) + (val << shift)`
This equals: `val * (1 + 2^shift) + const = multiply_add(val, mult, const)`

### Impact
- Reduced 3 VALU ops to 1 per stage for stages 0, 2, 4
- Saved ~3,000 VALU ops total

---

## Key Optimization: Tree Level Caching

### Insight
- Rounds 0 and 11: All items at idx=0 (tree wrap-around)
- Rounds 1 and 12: Items at idx ∈ {1, 2}

### Solution
- Cache forest[0], forest[1], forest[2] in vector registers
- Use VALU-based select instead of gather for these rounds
- `result = (forest[1] - forest[2]) * (idx & 1) + forest[2]` via multiply_add

### Impact
- Eliminated 512 gather loads (4 rounds × 256 gathers)
- Replaced with fast VALU operations

---

## Current Bottleneck: Packing Efficiency

### The Problem
Theoretical minimum: 1,721 cycles
Actual: 3,302 cycles
**Efficiency: 52%**

### Why
Each round has a dependency chain:
```
Gather(LOAD) -> XOR(VALU) -> Hash(VALU) -> idx_update(VALU) -> Gather(LOAD)
```

The packer processes operations in order. It can pack operations WITHIN each phase but cannot overlap ACROSS phases due to dependencies.

### What We Tried
1. **Double-buffered batches**: Overlap batch N compute with batch N+1 load
   - Helped within rounds but not across rounds
   
2. **Cross-chunk pipelining**: Overlap chunk N processing with chunk N+1 initial load
   - Marginal improvement (chunks are large, initial load is small)

3. **Round-level pipelining**: Overlap hash[N] with gather[N+1]
   - Blocked by dependency: gather[N+1] needs idx from round N

### Fundamental Limitation
The algorithm's dependency chain is intrinsic:
- idx_update produces new idx
- Next round's gather NEEDS that idx
- No way to overlap without breaking correctness

---

## Target Analysis

**Target: 2,164 cycles (test_opus4_many_hours)**

With theoretical minimum of 1,721 cycles:
- Current efficiency: 52% (3,302/1,721 = 1.92x overhead)
- Required efficiency: 79% (1,721/2,164 = 0.79x)

Options to reach target:
1. **Reduce ops further**: Already very optimized
2. **Improve packing**: Needs fundamental restructuring
3. **Different algorithm**: Beyond scope of current approach

---

## Architecture Constraints (from problem.py)

```python
SLOT_LIMITS = {
    "alu": 12,    # Scalar integer ops
    "valu": 6,    # 8-wide SIMD ops
    "load": 2,    # Memory reads
    "store": 2,   # Memory writes
    "flow": 1,    # Control flow
}
VLEN = 8          # Vector width
SCRATCH_SIZE = 1536  # Register file
```

---

## Files Modified

- `perf_takehome.py`: Main kernel implementation
  - `build_kernel()`: Multi-round chunk processing with pipelining
  - `build_packed()`: Greedy VLIW packer with dependency tracking
  - Helper functions: emit_gather_phase, emit_hash_phase, emit_idx_update, etc.
