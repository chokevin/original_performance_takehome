# Kernel Optimization Documentation

## Overview
This document tracks optimizations made to the VLIW SIMD kernel for the performance take-home.

**Current Result: 5,536 cycles (26.7x speedup from baseline)**

---

## Optimization 1: SIMD Vectorization (VALU)

### Result: 147,734 → 25,677 cycles (5.75x)

### Problem
The baseline kernel processes **1 batch item per cycle** using scalar ALU operations.
- 256 batch items × 16 rounds × ~36 ops/item = 147,734 cycles
- Slot utilization: only 10% (1 slot used per instruction)

### Solution
Use **VALU (8-wide SIMD)** to process 8 batch items simultaneously.

### Changes Made

| Component | Before (Scalar) | After (VALU) |
|-----------|-----------------|--------------|
| Batch loop | `for i in range(256)` | `for i in range(0, 256, 8)` |
| Load idx/val | `load` (1 item) | `vload` (8 contiguous items) |
| Hash arithmetic | `alu +, ^, <<, >>` | `valu +, ^, <<, >>` |
| Index math | `alu *, +, %, ==, <` | `valu *, +, %, ==, <` |
| Select | `flow select` | `flow vselect` |
| Store idx/val | `store` (1 item) | `vstore` (8 items) |
| Load node_val | `load` | **8× scalar load** (gather) |

### Why Gather Can't Be Vectorized
The `node_val` load requires fetching `forest[idx[i]]` for each item. After the first round, each item has a different `idx` value (e.g., items traverse to nodes 13, 8, 14, 8, 12, 12, 11, 12). Since `vload` requires **contiguous** addresses, we must use 8 separate scalar loads.

### Memory Layout (proof of contiguity)
From `build_mem_image()` in `problem.py`:
```python
mem[inp_indices_p:inp_values_p] = inp.indices  # contiguous slice
mem[inp_values_p:] = inp.values                # contiguous slice
```
This guarantees `idx[0..255]` and `val[0..255]` are at consecutive addresses, enabling `vload`/`vstore`.

---

## Optimization 2: VLIW Instruction Packing

### Result: 25,677 → 13,888 cycles (1.85x)

### Problem
After SIMD, each instruction still used only **1 slot** when the architecture supports:
- 12 ALU slots per cycle
- 6 VALU slots per cycle
- 2 load slots per cycle
- 2 store slots per cycle
- 1 flow slot per cycle

Slot utilization was only ~16%.

### Solution
Implement `build_packed()` to automatically combine independent operations into the same cycle.

### How It Works
The packer scans through operations and bundles them together when:
1. **Slot limit not exceeded** - e.g., can't pack 13 ALU ops in one cycle
2. **No data dependencies** - can't read from an address written in same cycle

### Dependency Tracking for Vectors
A key insight: VALU operations on `v_val` (base address 21) actually read/write addresses 21-28 (VLEN=8). The packer must track the **full range** to avoid incorrect packing.

```python
def get_written_addrs(engine, slot):
    if engine == "valu":
        base = slot[1]
        return {base + i for i in range(VLEN)}  # All 8 addresses
```

### Example: Gather Address Calculation
Before packing (8 cycles):
```
Cycle 1: addr0 = forest_p + idx[0]
Cycle 2: addr1 = forest_p + idx[1]
...
Cycle 8: addr7 = forest_p + idx[7]
```

After packing (1 cycle):
```
Cycle 1: addr0 = forest_p + idx[0]
         addr1 = forest_p + idx[1]
         ...
         addr7 = forest_p + idx[7]   (8 ALU ops, limit is 12)
```

### Why Dependencies Limit Hash Packing
Each hash stage has 3 operations with a dependency chain:
```
tmp1 = val + const1  ─┬─ Cycle N   (CAN pack - both read val)
tmp2 = val << const2 ─┘
val  = tmp1 ^ tmp2   ─── Cycle N+1 (MUST wait - reads tmp1, tmp2)
```

So each hash stage needs **2 cycles minimum**. 6 stages × 2 = 12 cycles for hash.

---

## Optimization 3: Operation Batching

### Result: 13,888 → 6,368 cycles (2.18x)

### Problem
Operations were packed within each vector iteration, but iterations ran sequentially:
```
iter0: load → gather → hash → store
iter1: load → gather → hash → store
...
```

This limits packing because each iteration has internal dependencies. But **iterations are independent of each other**!

### Solution
Batch 16 vector iterations together, grouping all operations of the same type:

```
Before (sequential iterations):
  iter0: [load_idx, load_val, 8×gather_addr, 8×gather_load, hash..., store]
  iter1: [load_idx, load_val, 8×gather_addr, 8×gather_load, hash..., store]
  ...

After (batched by operation type):
  Phase 1:  ALL 16 load_idx addresses     (16 ALU → 2 cycles at 12/cycle)
  Phase 2:  ALL 16 vloads for idx         (16 loads → 8 cycles at 2/cycle)
  Phase 3:  ALL 16 load_val addresses     (16 ALU → 2 cycles)
  Phase 4:  ALL 16 vloads for val         (16 loads → 8 cycles)
  Phase 5:  ALL 16×8 gather addresses     (128 ALU → 11 cycles)
  Phase 6:  ALL 16×8 gather loads         (128 loads → 64 cycles)
  Phase 7:  ALL 16 XORs                   (16 VALU → 3 cycles at 6/cycle)
  Phase 8+: ALL hash stages batched
  ...
```

### Memory Requirements
Each iteration needs 6 vector registers × 8 words = 48 words.
- 16 iterations × 48 = 768 words
- Plus constants, address registers ≈ 200 words
- Total: ~968 words < 1536 scratch limit ✓

---

## Optimization 4: Eliminate vselects with ALU

### Result: 6,368 → 5,536 cycles (1.15x)

### Problem
The flow engine can only execute **1 operation per cycle**. We had 1,024 vselects:
- 2 vselects per vector iteration × 32 iterations × 16 rounds = 1,024

This alone consumed 1,024 cycles (16% of total).

### Solution
Replace both vselects with pure VALU arithmetic:

#### 1. Branch Direction Select
```python
# Before (vselect):
cond = (val % 2) == 0
offset = vselect(cond, 1, 2)  # 1 if even, 2 if odd

# After (VALU):
offset = 1 + (val & 1)
# val&1 = 0 if even, 1 if odd
# 1 + 0 = 1 ✓ (even → left child)
# 1 + 1 = 2 ✓ (odd → right child)
```

#### 2. Index Wrap Select
```python
# Before (vselect):
cond = idx < n_nodes
result = vselect(cond, idx, 0)  # idx if in bounds, 0 if out

# After (VALU):
cond = idx < n_nodes  # 0 or 1
result = idx * cond
# If in bounds: idx × 1 = idx ✓
# If out of bounds: idx × 0 = 0 ✓
```

### Results
- Eliminated all 1,024 vselects
- Replaced with VALU ops that pack with other operations
- Saved ~832 cycles

---

## Current Bottleneck: Load Engine

After eliminating vselects, the **load engine** is now the bottleneck:
- 4,103 scalar loads (gather) ÷ 2 per cycle = **2,052 cycles minimum**
- 1,024 vloads (idx/val) ÷ 2 per cycle = 512 cycles

The gather loads cannot be vectorized because each batch item accesses a different tree node.

---

## Optimization 5: Tree Level Caching

### Result: 5,536 → 5,306 cycles (1.04x)

### Insight
At round R, all indices are in range [2^R - 1, 2^(R+1) - 2]:
- Round 0: all items at idx=0
- Round 1: items at idx ∈ {1, 2}
- Round 2: items at idx ∈ {3, 4, 5, 6}

### Solution
Cache first 7 tree nodes in vector scratch. Use cached values instead of gather loads for rounds 0-2:
- Round 0: Direct copy from cache[0]
- Round 1: vselect between cache[1] and cache[2] based on idx&1
- Round 2: Multi-level vselect from cache[3..6]

---

## Optimization 6: Address Precomputation

### Result: 5,306 → 5,219 cycles (1.02x)

### Problem
Address calculations like `inp_indices_p + offset` were recomputed every round.

### Solution
Precompute all 32 idx_addr and val_addr values once before the round loop. Reuse across all 16 rounds.

---

## Optimization 7: VALU Gather Address Computation

### Result: 5,219 → 4,981 cycles (1.05x)

### Problem
Computing `forest_values_p + v_idx[j] + vi` for each of 8 vector elements used 8 scalar ALU operations.

### Solution
1. Broadcast `forest_values_p` to a vector register
2. Use single VALU add: `v_gather_addrs[j] = v_forest_p + v_idx[j]`
3. Extract individual addresses for scalar loads

This computes all 8 gather addresses in one VALU cycle instead of 8 ALU cycles.

---

## Summary

| Optimization | Cycles | Speedup | Cumulative |
|--------------|--------|---------|------------|
| Baseline | 147,734 | - | 1x |
| 1. SIMD (VALU) | 25,677 | 5.75x | 5.75x |
| 2. VLIW Packing | 13,888 | 1.85x | 10.6x |
| 3. Op Batching | 6,368 | 2.18x | 23.2x |
| 4. Eliminate vselects | 5,536 | 1.15x | 26.7x |
| 5. Tree Level Caching | 5,306 | 1.04x | 27.8x |
| 6. Address Precomputation | 5,219 | 1.02x | 28.3x |
| 7. VALU Gather Addresses | 4,981 | 1.05x | 29.7x |
| 8. BATCH_ITERS=8 | 4,881 | 1.02x | **30.3x** |

Tests passing: 3/9
Next target: <2,164 cycles (test_opus4_many_hours) - requires ~2.26x more improvement

---

## Critical Finding: Engine Serialization

### The Problem

Analysis of the execution trace reveals a **critical inefficiency**:

```
Engine utilization at 4,881 cycles:
- Load only:  2,056 cycles (42%)
- VALU only:  2,157 cycles (44%)
- Both:         172 cycles (3.5%)
- Other:        496 cycles (10%)
```

**The load and VALU engines run almost completely serially!** Only 3.5% of cycles use both engines simultaneously.

### Why This Happens

The algorithm creates a **dependency chain** within each batch:

```
1. vload idx, val        <- LOAD engine
2. compute gather addrs  <- VALU engine (needs idx)
3. gather forest[idx]    <- LOAD engine (needs addrs)
4. XOR + hash           <- VALU engine (needs gathered values)
5. compute next idx     <- VALU engine (needs hash result)
6. vstore idx, val      <- STORE engine (needs idx)
```

Each phase **must wait** for the previous phase to complete. The packer can only pack operations **within** each phase, not across phases.

### Theoretical Analysis

If engines could fully overlap:
- Load cycles needed: ~2,228
- VALU cycles needed: ~2,267
- Theoretical minimum: max(2228, 2267) = **~2,267 cycles**

Current: 4,881 cycles = 2.15x worse than theoretical

### The Solution: Software Pipelining

To achieve overlap, we need to **interleave operations from different batches**:

```
Current (serial):
  Batch 0: [load] -> [valu] -> [store]
  Batch 1: [load] -> [valu] -> [store]

Pipelined (overlapped):
  Batch 0: [load]
  Batch 0: [valu] + Batch 1: [load]  <- OVERLAP!
  Batch 0: [store] + Batch 1: [valu]
  Batch 1: [store]
```

This requires:
1. **Double-buffering**: Two sets of registers (one per batch in flight)
2. **Interleaved emission**: Emit load ops from batch N+1 interleaved with hash ops from batch N
3. **Memory constraints**: With BATCH_ITERS=8 and 2 buffer sets, fits in 1536-word scratch

### Expected Improvement

With proper pipelining:
- Current: 4,881 cycles
- Target: ~2,500-3,000 cycles (50-60% reduction)
- Theoretical minimum: ~2,267 cycles

---

## Next Steps: Pipeline Refactoring

Required changes to `build_kernel`:

1. Reduce `BATCH_ITERS` to 8 (already done)
2. Allocate double-buffered registers (2 sets of v_idx, v_val, etc.)
3. Restructure the round loop to:
   - Prologue: Load first batch
   - Steady state: Interleave hash[N] with load[N+1]
   - Epilogue: Complete last batch
4. Emit operations in interleaved order so packer sees both load and valu ops together
