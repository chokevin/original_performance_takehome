# Kernel Optimization Documentation

## Overview
This document tracks optimizations made to the VLIW SIMD kernel for the performance take-home.

---

## Optimization 1: SIMD Vectorization (VALU)

### Commit
Initial SIMD implementation

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

### Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cycles | 147,734 | 25,677 | **5.75× faster** |
| Batch iterations/round | 256 | 32 | 8× fewer |
| Tests passing | 1/9 | 2/9 | +1 |

### Remaining Bottlenecks
1. **Gather operations**: 16 cycles per vector iteration (8 ALU + 8 load)
2. **No VLIW packing**: Still 1 slot per instruction, ~10% utilization
3. **Fully unrolled**: No loops - could use jumps to reduce code size

---

## Future Optimizations (TODO)

### Optimization 2: VLIW Instruction Packing
Pack multiple independent operations into the same cycle:
- Up to 12 ALU + 6 VALU + 2 load + 2 store + 1 flow per cycle

### Optimization 3: Loop with Jumps
Replace unrolled code with actual loops using `cond_jump` to reduce instruction count.
