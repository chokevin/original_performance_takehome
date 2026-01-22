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

## Summary

| Optimization | Cycles | Speedup | Cumulative |
|--------------|--------|---------|------------|
| Baseline | 147,734 | - | 1x |
| 1. SIMD (VALU) | 25,677 | 5.75x | 5.75x |
| 2. VLIW Packing | 13,888 | 1.85x | 10.6x |
| 3. Op Batching | 6,368 | 2.18x | 23.2x |
| 4. Eliminate vselects | 5,536 | 1.15x | **26.7x** |

Tests passing: 3/9
Next target: <2,164 cycles (test_opus4_many_hours) - requires ~2.5x more improvement
