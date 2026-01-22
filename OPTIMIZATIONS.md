# Kernel Optimization Documentation

## Overview
This document tracks optimizations made to the VLIW SIMD kernel for the performance take-home.

**Current Result: 6,368 cycles (23.2x speedup from baseline)**

---

## Optimization 1: SIMD Vectorization (VALU)

### Result: 147,734 → 25,677 cycles (5.75x)

### Problem
The baseline kernel processes **1 batch item per cycle** using scalar ALU operations.

### Solution
Use **VALU (8-wide SIMD)** to process 8 batch items simultaneously.

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
Each batch item traverses to a different tree node, so `forest[idx[i]]` has non-contiguous addresses. `vload` requires consecutive memory addresses.

---

## Optimization 2: VLIW Instruction Packing

### Result: 25,677 → 13,888 cycles (1.85x)

### Problem
Each instruction used only 1 slot when multiple slots are available per engine.

### Solution
Implement `build_packed()` to combine independent operations into same cycle, respecting:
- Slot limits (12 ALU, 6 VALU, 2 load, 2 store, 1 flow)
- Data dependencies (can't read an address written in same cycle)

### Key Insight: Vector Dependency Tracking
A VALU op on `v_val` (address 21) actually reads/writes addresses 21-28 (VLEN=8). The packer must track the full range to avoid false packing.

---

## Optimization 3: Operation Batching

### Result: 13,888 → 6,368 cycles (2.18x)

### Problem
Operations were packed within each vector iteration, but iterations were processed sequentially. This limited packing because:
- Each iteration has internal dependencies (hash stage N needs stage N-1)
- But iterations are **independent** of each other!

### Solution
Batch 16 vector iterations (128 batch items) together, grouping all operations of the same type:

```
Before (per-iteration):
  iter0: load_idx → load_val → gather → hash → store
  iter1: load_idx → load_val → gather → hash → store
  ...

After (batched by operation):
  Phase 1: ALL 16 load_idx addresses (16 ALU → 2 cycles at 12 ALU/cycle)
  Phase 2: ALL 16 vloads for idx (16 loads → 8 cycles at 2 load/cycle)
  Phase 3: ALL 16 load_val addresses
  Phase 4: ALL 16 vloads for val
  Phase 5: ALL 16×8 gather addresses (128 ALU → 11 cycles)
  Phase 6: ALL 16×8 gather loads (128 loads → 64 cycles)
  Phase 7: ALL 16 XORs (16 VALU → 3 cycles at 6 VALU/cycle)
  Phase 8-13: ALL hash stages (each stage's ops batched)
  ...
```

### Memory Requirements
Each iteration needs 6 vector registers (v_idx, v_val, v_node_val, v_tmp1, v_tmp2, v_tmp3).
- 6 vectors × 8 words × 16 iterations = 768 words
- Plus constants, addresses = ~200 words
- Total: ~968 words < 1536 scratch limit ✓

### Results

| Metric | Before | After |
|--------|--------|-------|
| Cycles | 13,888 | 6,368 |
| VALU utilization | 1.33/6 (22%) | **6.00/6 (100%)** |
| ALU utilization | 2.40/12 (20%) | **10.11/12 (84%)** |
| Load utilization | 1.65/2 (83%) | **1.98/2 (99%)** |
| Store utilization | 1.00/2 (50%) | **2.00/2 (100%)** |
| Overall slot utilization | 27% | **95.5%** |

---

## Current Bottleneck: Flow Engine

The flow engine can only execute 1 operation per cycle. We need 1,024 vselects (32 per batch × 32 iterations), which takes 1,024 cycles minimum.

Flow operations: 1,026 cycles (16% of total)

---

## Summary

| Optimization | Cycles | Speedup | Cumulative |
|--------------|--------|---------|------------|
| Baseline | 147,734 | - | 1x |
| SIMD (VALU) | 25,677 | 5.75x | 5.75x |
| VLIW Packing | 13,888 | 1.85x | 10.6x |
| Op Batching | 6,368 | 2.18x | **23.2x** |

Tests passing: 3/9
Next target: <2,164 cycles (test_opus4_many_hours)
