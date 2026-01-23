# Kernel Optimization Documentation

## Overview
This document tracks optimizations made to the VLIW SIMD kernel for the performance take-home.

**Current Result: 2,896 cycles (51.0x speedup from baseline)**

**Tests Passing: 3/9**

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
| 11. Multi-round chunks | 3,302 | 1.10x | 44.7x |
| 12. Level 2 caching | 3,144 | 1.05x | 47.0x |
| 13. idx multiply_add | 3,064 | 1.03x | 48.3x |
| 14. Conditional bounds check | 2,896 | 1.06x | **51.0x** |

---

## Current Status

### Operation Counts
- VALU ops: 9,311 → 1,552 min cycles at 6/cycle
- Load ops: 2,694 → 1,347 min cycles at 2/cycle
- Store ops: 64 → 32 min cycles at 2/cycle
- **Theoretical minimum: 1,552 cycles (VALU-bound)**

### Efficiency
- Current: 2,896 cycles
- Theoretical min: 1,552 cycles
- **Efficiency: 53.6%**

---

## Key Optimizations

### 1. Tree Level Caching (Levels 0-2)
**Problem**: Scatter gathers are expensive (8 loads per vector).

**Solution**: Cache nodes 0-6 and use VALU-based select:
- Level 0: Just use cached forest[0]
- Level 1: 2-way select with multiply_add
- Level 2: 4-way select using 3 multiply_adds

**Impact**: Eliminated 1,012 gather loads (levels 0-2, rounds 0-2 and 11-13)

### 2. Hash multiply_add Optimization
**Pattern**: Hash stages 0, 2, 4 have form `(val + const) + (val << shift)`
**Rewrite**: `val * (1 + 2^shift) + const` = single multiply_add

**Impact**: 9 VALU ops → 3 per hash (for 3 optimized stages)

### 3. idx multiply_add Optimization
**Pattern**: `idx = 2*idx + offset` (two ops: multiply, add)
**Rewrite**: `multiply_add(2, idx, offset)` (one op)

**Impact**: Saved 1 VALU op per round per vector

### 4. Conditional Bounds Check
**Insight**: Bounds check (`idx = idx * (idx < n_nodes)`) only needed at tree depth boundary.

**Solution**: Only emit bounds check at round 10 (level 10), not every round.

**Impact**: Saved 2 VALU ops per round for 15 rounds

---

## Current Bottleneck: Dependency Chain

### The Fundamental Problem
Each round has an unavoidable dependency chain:
```
Gather(LOAD) → XOR(VALU) → Hash(VALU) → idx_update(VALU) → next Gather(LOAD)
                                              ↓
                                        depends on new idx
```

This forces LOAD and VALU to run mostly serially (only 3.7% overlap).

### Why 53.6% Efficiency
- Theoretical: All LOAD and VALU could overlap perfectly
- Reality: Dependency chain forces sequential execution
- Gap: ~1,344 cycles lost to waiting

### Attempted Mitigations
1. **Round interleaving**: Process multiple chunks' rounds together - no improvement
2. **Speculative loading**: Pre-compute both children - register pressure issues
3. **Cross-chunk pipelining**: Already implemented, helps marginally

---

## Target Gap Analysis

**Next Target: 2,164 cycles (test_opus4_many_hours)**

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Cycles | 2,896 | 2,164 | 732 |
| Required speedup | - | 1.34x | - |
| Required efficiency | 53.6% | 71.7% | 18.1% |

**Potential approaches to close the gap:**
1. Cache more tree levels (trades LOAD for VALU)
2. Speculative execution (doubles LOADs but enables overlap)
3. Better instruction scheduling
4. Algorithmic restructuring

---

## Architecture Reference (from problem.py)

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
  - `build_kernel()`: Multi-round chunk processing with level caching
  - `build_packed()`: Greedy VLIW packer with dependency tracking
  - Helper functions: emit_gather_phase, emit_hash_phase, emit_idx_update
