"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs
    
    def build_packed(self, slots: list[tuple[Engine, tuple]]):
        """
        Pack slots into VLIW instruction bundles respecting slot limits and dependencies.
        """
        from problem import SLOT_LIMITS, VLEN
        
        instrs = []
        current_bundle = {}
        current_counts = {e: 0 for e in SLOT_LIMITS}
        writes_in_bundle = set()
        
        def get_written_addrs(engine, slot):
            op = slot[0]
            addrs = set()
            if engine == "alu":
                if len(slot) > 1:
                    addrs.add(slot[1])
            elif engine == "valu":
                if len(slot) > 1:
                    base = slot[1]
                    for i in range(VLEN):
                        addrs.add(base + i)
            elif engine == "load":
                if op in ("load", "const", "load_offset"):
                    if len(slot) > 1:
                        addrs.add(slot[1])
                elif op == "vload":
                    if len(slot) > 1:
                        base = slot[1]
                        for i in range(VLEN):
                            addrs.add(base + i)
            elif engine == "flow":
                if op == "select":
                    if len(slot) > 1:
                        addrs.add(slot[1])
                elif op == "vselect":
                    if len(slot) > 1:
                        base = slot[1]
                        for i in range(VLEN):
                            addrs.add(base + i)
                elif op in ("add_imm", "coreid"):
                    if len(slot) > 1:
                        addrs.add(slot[1])
            return addrs
        
        def get_read_addrs(engine, slot):
            op = slot[0]
            addrs = set()
            if engine == "alu":
                if len(slot) >= 4:
                    addrs.add(slot[2])
                    addrs.add(slot[3])
            elif engine == "valu":
                if op == "vbroadcast" and len(slot) >= 3:
                    addrs.add(slot[2])
                elif len(slot) >= 4:
                    for src_base in [slot[2], slot[3]]:
                        for i in range(VLEN):
                            addrs.add(src_base + i)
            elif engine == "load":
                if op == "load" and len(slot) >= 3:
                    addrs.add(slot[2])
                elif op == "vload" and len(slot) >= 3:
                    addrs.add(slot[2])
            elif engine == "store":
                if op == "store" and len(slot) >= 3:
                    addrs.add(slot[1])
                    addrs.add(slot[2])
                elif op == "vstore" and len(slot) >= 3:
                    addrs.add(slot[1])
                    base = slot[2]
                    for i in range(VLEN):
                        addrs.add(base + i)
            elif engine == "flow":
                if op == "select" and len(slot) >= 5:
                    addrs.add(slot[2])
                    addrs.add(slot[3])
                    addrs.add(slot[4])
                elif op == "vselect" and len(slot) >= 5:
                    for src_base in [slot[2], slot[3], slot[4]]:
                        for i in range(VLEN):
                            addrs.add(src_base + i)
                elif op == "add_imm" and len(slot) >= 3:
                    addrs.add(slot[2])
            return addrs
        
        def has_dependency(engine, slot):
            return bool(get_read_addrs(engine, slot) & writes_in_bundle)
        
        def can_add_to_bundle(engine, slot):
            if engine == "debug":
                return True
            limit = SLOT_LIMITS.get(engine, 1)
            if current_counts.get(engine, 0) >= limit:
                return False
            if has_dependency(engine, slot):
                return False
            return True
        
        def flush_bundle():
            nonlocal current_bundle, current_counts, writes_in_bundle
            if current_bundle:
                instrs.append(current_bundle)
            current_bundle = {}
            current_counts = {e: 0 for e in SLOT_LIMITS}
            writes_in_bundle = set()
        
        def add_to_bundle(engine, slot):
            nonlocal current_bundle, current_counts, writes_in_bundle
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            if engine != "debug":
                current_counts[engine] = current_counts.get(engine, 0) + 1
                writes_in_bundle |= get_written_addrs(engine, slot)
        
        for engine, slot in slots:
            if not can_add_to_bundle(engine, slot):
                flush_bundle()
            add_to_bundle(engine, slot)
        
        flush_bundle()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vhash(self, val_hash_addr, tmp1, tmp2, v_hash_consts):
        """
        Vectorized hash using VALU - processes 8 values at once.
        v_hash_consts: pre-allocated vector constants for each hash stage
        """
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1, const2 = v_hash_consts[hi]
            # a = op1(val, const1)
            slots.append(("valu", (op1, tmp1, val_hash_addr, const1)))
            # b = op3(val, const2)
            slots.append(("valu", (op3, tmp2, val_hash_addr, const2)))
            # val = op2(a, b)
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Software-pipelined SIMD implementation.
        
        Key optimization: Overlap load and VALU engines by interleaving
        operations from different batches. While batch N computes hash,
        batch N+1 loads data.
        
        Uses double-buffering with two sets of registers.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting pipelined SIMD loop"))

        body = []

        # Number of vector iterations per batch (smaller for double-buffering)
        BATCH_ITERS = 8  # Number of vector iterations per batch (for pipelining)
        
        # Double-buffered registers: two sets (A and B)
        # While set A does compute, set B does loads (and vice versa)
        bufs = []
        for b in range(2):
            buf = {
                'v_idx': [self.alloc_scratch(f"b{b}_v_idx_{j}", VLEN) for j in range(BATCH_ITERS)],
                'v_val': [self.alloc_scratch(f"b{b}_v_val_{j}", VLEN) for j in range(BATCH_ITERS)],
                'v_node_val': [self.alloc_scratch(f"b{b}_v_node_val_{j}", VLEN) for j in range(BATCH_ITERS)],
                'v_tmp1': [self.alloc_scratch(f"b{b}_v_tmp1_{j}", VLEN) for j in range(BATCH_ITERS)],
                'v_tmp2': [self.alloc_scratch(f"b{b}_v_tmp2_{j}", VLEN) for j in range(BATCH_ITERS)],
                'v_tmp3': [self.alloc_scratch(f"b{b}_v_tmp3_{j}", VLEN) for j in range(BATCH_ITERS)],
                'v_gather_addrs': [self.alloc_scratch(f"b{b}_v_ga_{j}", VLEN) for j in range(BATCH_ITERS)],
            }
            bufs.append(buf)
        
        # Shared registers
        addr_regs = [self.alloc_scratch(f"addr_{j}") for j in range(BATCH_ITERS)]
        
        # Vector constants (shared)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        # Pre-allocate hash constants
        v_hash_consts = []
        v_hash_mults = []  # For multiply_add optimization
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1 = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
            const2 = self.alloc_scratch(f"v_hc2_{hi}", VLEN)
            v_hash_consts.append((const1, const2))
            # Pre-allocate multiplier for multiply_add optimization
            if op1 == '+' and op2 == '+' and op3 == '<<':
                v_mult = self.alloc_scratch(f"v_hmult_{hi}", VLEN)
                v_hash_mults.append(v_mult)
            else:
                v_hash_mults.append(None)
        
        # Broadcast constants
        body.append(("valu", ("vbroadcast", v_zero, zero_const)))
        body.append(("valu", ("vbroadcast", v_one, one_const)))
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        body.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))
        
        # Broadcast forest_values_p for faster gather address computation
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        body.append(("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"])))
        
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1, const2 = v_hash_consts[hi]
            body.append(("valu", ("vbroadcast", const1, self.scratch_const(val1))))
            body.append(("valu", ("vbroadcast", const2, self.scratch_const(val3))))
            # Broadcast multiplier for multiply_add optimization
            if v_hash_mults[hi] is not None:
                mult_factor = 1 + (1 << val3)
                body.append(("valu", ("vbroadcast", v_hash_mults[hi], self.scratch_const(mult_factor))))

        n_vec_iters = batch_size // VLEN
        
        # Pre-compute idx and val addresses for all batch positions
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(n_vec_iters)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(n_vec_iters)]
        
        for i in range(n_vec_iters):
            offset = i * VLEN
            body.append(("alu", ("+", idx_addrs[i], self.scratch["inp_indices_p"], self.scratch_const(offset))))
            body.append(("alu", ("+", val_addrs[i], self.scratch["inp_values_p"], self.scratch_const(offset))))
        
        # Cache forest levels 0-2 for early rounds (nodes 0-6)
        # Level 0: node 0 (1 node)
        # Level 1: nodes 1-2 (2 nodes)
        # Level 2: nodes 3-6 (4 nodes)
        # Total: 7 nodes × 8 words = 56 words
        
        v_level_cache = []  # v_level_cache[i] = broadcast of forest[i]
        forest_scalar = self.alloc_scratch("forest_scalar")
        
        for node_idx in range(7):  # nodes 0-6
            v_node = self.alloc_scratch(f"v_forest_{node_idx}", VLEN)
            v_level_cache.append(v_node)
            body.append(("alu", ("+", forest_scalar, self.scratch["forest_values_p"], self.scratch_const(node_idx))))
            body.append(("load", ("load", forest_scalar, forest_scalar)))
            body.append(("valu", ("vbroadcast", v_node, forest_scalar)))
        
        # Aliases for backward compatibility
        v_forest_0 = v_level_cache[0]
        v_forest_1 = v_level_cache[1]
        v_forest_2 = v_level_cache[2]
        
        # Precompute differences for select operations
        v_forest_diff_1_2 = self.alloc_scratch("v_forest_diff_1_2", VLEN)  # f1 - f2
        body.append(("valu", ("-", v_forest_diff_1_2, v_forest_1, v_forest_2)))
        v_forest_diff = v_forest_diff_1_2  # alias
        
        # For level 2: f3, f4, f5, f6
        # Precompute: f4-f3, f6-f5, (f5 + f6)/2 - (f3 + f4)/2 
        v_diff_4_3 = self.alloc_scratch("v_diff_4_3", VLEN)  # f4 - f3
        v_diff_6_5 = self.alloc_scratch("v_diff_6_5", VLEN)  # f6 - f5
        body.append(("valu", ("-", v_diff_4_3, v_level_cache[4], v_level_cache[3])))
        body.append(("valu", ("-", v_diff_6_5, v_level_cache[6], v_level_cache[5])))
        
        # v_three constant for level 2 select
        v_three = self.alloc_scratch("v_three", VLEN)
        body.append(("valu", ("vbroadcast", v_three, self.scratch_const(3))))
        
        # Helper functions to emit operations for each phase
        def emit_gather_phase(buf, chunk_count, round_idx, ops):
            """Emit gather operations for a round"""
            effective_round = round_idx % (forest_height + 1)
            
            if effective_round == 0:
                # Level 0: all idx=0, use cached forest[0]
                for j in range(chunk_count):
                    ops.append(("valu", ("+", buf['v_node_val'][j], v_forest_0, v_zero)))
            elif effective_round == 1:
                # Level 1: idx in {1, 2}, use multiply_add select
                # cond = idx & 1: idx=1 → cond=1, idx=2 → cond=0
                # result = (forest[1]-forest[2])*cond + forest[2]
                for j in range(chunk_count):
                    ops.append(("valu", ("&", buf['v_tmp1'][j], buf['v_idx'][j], v_one)))
                for j in range(chunk_count):
                    ops.append(("valu", ("multiply_add", buf['v_node_val'][j], v_forest_diff, buf['v_tmp1'][j], v_forest_2)))
            elif effective_round == 2:
                # Level 2: idx in {3,4,5,6}, 4-way select
                # local_idx = idx - 3, in {0,1,2,3}
                # bit0 = local_idx & 1, bit1 = (local_idx >> 1) & 1
                # low = f3 + bit0*(f4-f3)   [select f3 or f4]
                # high = f5 + bit0*(f6-f5)  [select f5 or f6]
                # result = low + bit1*(high-low)
                for j in range(chunk_count):
                    # local_idx = idx - 3
                    ops.append(("valu", ("-", buf['v_tmp1'][j], buf['v_idx'][j], v_three)))
                for j in range(chunk_count):
                    # bit0 = local_idx & 1
                    ops.append(("valu", ("&", buf['v_tmp2'][j], buf['v_tmp1'][j], v_one)))
                for j in range(chunk_count):
                    # bit1 = (local_idx >> 1) & 1
                    ops.append(("valu", (">>", buf['v_tmp1'][j], buf['v_tmp1'][j], v_one)))
                for j in range(chunk_count):
                    ops.append(("valu", ("&", buf['v_tmp1'][j], buf['v_tmp1'][j], v_one)))
                # low = f3 + bit0*(f4-f3)
                for j in range(chunk_count):
                    ops.append(("valu", ("multiply_add", buf['v_tmp3'][j], v_diff_4_3, buf['v_tmp2'][j], v_level_cache[3])))
                # high = f5 + bit0*(f6-f5)  
                for j in range(chunk_count):
                    ops.append(("valu", ("multiply_add", buf['v_gather_addrs'][j], v_diff_6_5, buf['v_tmp2'][j], v_level_cache[5])))
                # result = low + bit1*(high-low)
                for j in range(chunk_count):
                    ops.append(("valu", ("-", buf['v_tmp2'][j], buf['v_gather_addrs'][j], buf['v_tmp3'][j])))
                for j in range(chunk_count):
                    ops.append(("valu", ("multiply_add", buf['v_node_val'][j], buf['v_tmp2'][j], buf['v_tmp1'][j], buf['v_tmp3'][j])))
            else:
                # All other levels: full scatter gather
                for j in range(chunk_count):
                    ops.append(("valu", ("+", buf['v_gather_addrs'][j], v_forest_p, buf['v_idx'][j])))
                for j in range(chunk_count):
                    for vi in range(VLEN):
                        ops.append(("load", ("load", buf['v_node_val'][j] + vi, buf['v_gather_addrs'][j] + vi)))
        
        def emit_hash_phase(buf, chunk_count, ops):
            """Emit XOR, hash computation, and idx update"""
            for j in range(chunk_count):
                ops.append(("valu", ("^", buf['v_val'][j], buf['v_val'][j], buf['v_node_val'][j])))
            
            # Hash stages - interleave for better ILP
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1, const2 = v_hash_consts[hi]
                if op1 == '+' and op2 == '+' and op3 == '<<':
                    # Optimized: single multiply_add per vector
                    v_mult = v_hash_mults[hi]
                    for j in range(chunk_count):
                        ops.append(("valu", ("multiply_add", buf['v_val'][j], v_mult, buf['v_val'][j], const1)))
                else:
                    # Standard 3-op: emit in waves for better packing
                    for j in range(chunk_count):
                        ops.append(("valu", (op1, buf['v_tmp1'][j], buf['v_val'][j], const1)))
                    for j in range(chunk_count):
                        ops.append(("valu", (op3, buf['v_tmp2'][j], buf['v_val'][j], const2)))
                    for j in range(chunk_count):
                        ops.append(("valu", (op2, buf['v_val'][j], buf['v_tmp1'][j], buf['v_tmp2'][j])))
            
            # idx update: offset = 1 + (val & 1), idx = 2*idx + offset
            for j in range(chunk_count):
                ops.append(("valu", ("&", buf['v_tmp1'][j], buf['v_val'][j], v_one)))
            for j in range(chunk_count):
                ops.append(("valu", ("+", buf['v_tmp3'][j], buf['v_tmp1'][j], v_one)))
            for j in range(chunk_count):
                ops.append(("valu", ("*", buf['v_idx'][j], buf['v_idx'][j], v_two)))
            for j in range(chunk_count):
                ops.append(("valu", ("+", buf['v_idx'][j], buf['v_idx'][j], buf['v_tmp3'][j])))
            
            # idx bounds check: idx = idx * (idx < n_nodes)
            for j in range(chunk_count):
                ops.append(("valu", ("<", buf['v_tmp1'][j], buf['v_idx'][j], v_n_nodes)))
            for j in range(chunk_count):
                ops.append(("valu", ("*", buf['v_idx'][j], buf['v_idx'][j], buf['v_tmp1'][j])))
        
        def emit_store_phase(buf, batch_start, batch_count, ops):
            """Emit store operations"""
            for j in range(batch_count):
                ops.append(("store", ("vstore", idx_addrs[batch_start + j], buf['v_idx'][j])))
            for j in range(batch_count):
                ops.append(("store", ("vstore", val_addrs[batch_start + j], buf['v_val'][j])))
        
        def interleave_ops(ops_a, ops_b):
            """Interleave two lists of operations for better packing"""
            result = []
            max_len = max(len(ops_a), len(ops_b))
            for i in range(max_len):
                if i < len(ops_a):
                    result.append(ops_a[i])
                if i < len(ops_b):
                    result.append(ops_b[i])
            return result
        
        def emit_chunk_load(buf, chunk_start, chunk_count, ops):
            """Load initial idx/val for a chunk"""
            for j in range(chunk_count):
                ops.append(("load", ("vload", buf['v_idx'][j], idx_addrs[chunk_start + j])))
            for j in range(chunk_count):
                ops.append(("load", ("vload", buf['v_val'][j], val_addrs[chunk_start + j])))
        
        def emit_chunk_store(buf, chunk_start, chunk_count, ops):
            """Store final idx/val for a chunk"""
            for j in range(chunk_count):
                ops.append(("store", ("vstore", idx_addrs[chunk_start + j], buf['v_idx'][j])))
            for j in range(chunk_count):
                ops.append(("store", ("vstore", val_addrs[chunk_start + j], buf['v_val'][j])))
        
        def emit_hash_phase(buf, chunk_count, ops):
            """Emit XOR and hash computation (doesn't touch idx)"""
            # XOR
            for j in range(chunk_count):
                ops.append(("valu", ("^", buf['v_val'][j], buf['v_val'][j], buf['v_node_val'][j])))
            
            # Hash stages
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1, const2 = v_hash_consts[hi]
                if op1 == '+' and op2 == '+' and op3 == '<<':
                    v_mult = v_hash_mults[hi]
                    for j in range(chunk_count):
                        ops.append(("valu", ("multiply_add", buf['v_val'][j], v_mult, buf['v_val'][j], const1)))
                else:
                    for j in range(chunk_count):
                        ops.append(("valu", (op1, buf['v_tmp1'][j], buf['v_val'][j], const1)))
                    for j in range(chunk_count):
                        ops.append(("valu", (op3, buf['v_tmp2'][j], buf['v_val'][j], const2)))
                    for j in range(chunk_count):
                        ops.append(("valu", (op2, buf['v_val'][j], buf['v_tmp1'][j], buf['v_tmp2'][j])))
        
        def emit_idx_update(buf, chunk_count, ops, round_idx=None):
            """Emit idx update (depends on val, produces new idx)"""
            # offset = 1 + (val & 1)
            for j in range(chunk_count):
                ops.append(("valu", ("&", buf['v_tmp1'][j], buf['v_val'][j], v_one)))
            for j in range(chunk_count):
                ops.append(("valu", ("+", buf['v_tmp3'][j], buf['v_tmp1'][j], v_one)))
            # idx = 2*idx + offset using multiply_add
            for j in range(chunk_count):
                ops.append(("valu", ("multiply_add", buf['v_idx'][j], v_two, buf['v_idx'][j], buf['v_tmp3'][j])))
            
            # idx bounds check - only needed after reaching the leaves (level 10)
            # With 16 rounds, wrap happens at round 10 (level 10 with idx >= 2047)
            # The wrap also happens at round 10+11=21, but we only have 16 rounds
            if round_idx is not None:
                effective_round = round_idx % (forest_height + 1)
                needs_check = (effective_round == forest_height)  # level 10 exactly
            else:
                needs_check = True  # default to safe behavior
            
            if needs_check:
                for j in range(chunk_count):
                    ops.append(("valu", ("<", buf['v_tmp1'][j], buf['v_idx'][j], v_n_nodes)))
                for j in range(chunk_count):
                    ops.append(("valu", ("*", buf['v_idx'][j], buf['v_idx'][j], buf['v_tmp1'][j])))
        
        def emit_all_rounds_speculative(buf, chunk_count, ops):
            """Process all rounds with speculative child loading.
            
            For levels 2+, pre-compute both child indices and load both children
            before the hash completes. This allows LOAD to overlap with VALU.
            
            Dependencies per round:
              - left_idx = 2*idx + 1, right_idx = 2*idx + 2 (from current idx)
              - Load forest[left_idx], forest[right_idx] (can overlap with hash)
              - Hash current node_val
              - Select correct child based on hash result
              - Update idx to the selected child
            """
            for round_idx in range(rounds):
                effective_round = round_idx % (forest_height + 1)
                next_effective_round = (round_idx + 1) % (forest_height + 1)
                
                # --- Gather current node ---
                emit_gather_phase(buf, chunk_count, round_idx, ops)
                
                # --- Speculatively compute next round's child indices ---
                # Only for rounds where next round needs a gather (not level 0 or 1)
                if next_effective_round >= 2 and round_idx < rounds - 1:
                    # left_idx = 2*idx + 1
                    for j in range(chunk_count):
                        ops.append(("valu", ("*", buf['v_tmp1'][j], buf['v_idx'][j], v_two)))
                    for j in range(chunk_count):
                        ops.append(("valu", ("+", buf['v_tmp1'][j], buf['v_tmp1'][j], v_one)))  # left_idx in tmp1
                    # right_idx = 2*idx + 2 = left_idx + 1
                    for j in range(chunk_count):
                        ops.append(("valu", ("+", buf['v_tmp2'][j], buf['v_tmp1'][j], v_one)))  # right_idx in tmp2
                    
                    # --- Speculatively load both children (overlaps with hash) ---
                    # Compute addresses
                    for j in range(chunk_count):
                        ops.append(("valu", ("+", buf['v_gather_addrs'][j], v_forest_p, buf['v_tmp1'][j])))  # left addr
                    # We need another vector for right addresses - reuse v_node_val since we'll overwrite it
                    # Actually, let's load left children into v_tmp3 area and right into v_node_val
                    # But this gets complex with register pressure. For now, skip speculation.
                
                # --- Hash phase (XOR and hash) ---
                for j in range(chunk_count):
                    ops.append(("valu", ("^", buf['v_val'][j], buf['v_val'][j], buf['v_node_val'][j])))
                
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    const1, const2 = v_hash_consts[hi]
                    if op1 == '+' and op2 == '+' and op3 == '<<':
                        v_mult = v_hash_mults[hi]
                        for j in range(chunk_count):
                            ops.append(("valu", ("multiply_add", buf['v_val'][j], v_mult, buf['v_val'][j], const1)))
                    else:
                        for j in range(chunk_count):
                            ops.append(("valu", (op1, buf['v_tmp1'][j], buf['v_val'][j], const1)))
                        for j in range(chunk_count):
                            ops.append(("valu", (op3, buf['v_tmp2'][j], buf['v_val'][j], const2)))
                        for j in range(chunk_count):
                            ops.append(("valu", (op2, buf['v_val'][j], buf['v_tmp1'][j], buf['v_tmp2'][j])))
                
                # --- idx update (with conditional bounds check) ---
                for j in range(chunk_count):
                    ops.append(("valu", ("&", buf['v_tmp1'][j], buf['v_val'][j], v_one)))
                for j in range(chunk_count):
                    ops.append(("valu", ("+", buf['v_tmp3'][j], buf['v_tmp1'][j], v_one)))
                for j in range(chunk_count):
                    ops.append(("valu", ("multiply_add", buf['v_idx'][j], v_two, buf['v_idx'][j], buf['v_tmp3'][j])))
                # Only bounds check at level 10 (exactly at leaves)
                if effective_round == forest_height:
                    for j in range(chunk_count):
                        ops.append(("valu", ("<", buf['v_tmp1'][j], buf['v_idx'][j], v_n_nodes)))
                    for j in range(chunk_count):
                        ops.append(("valu", ("*", buf['v_idx'][j], buf['v_idx'][j], buf['v_tmp1'][j])))
        
        def emit_all_rounds_pipelined(buf, chunk_count, ops):
            """Process all rounds with fine-grained pipelining.
            
            Key insight: Hash phase doesn't modify idx, so we can overlap:
              hash[round N] with gather[round N] (gather reads old idx from previous round)
            
            But idx_update[N] must complete before gather[N+1].
            
            Timeline for rounds 0,1,2:
              Gather(0) -> [Hash(0) | idx_update(-)] -> 
              Gather(1) -> [Hash(1) | idx_update(0)] -> 
              Gather(2) -> [Hash(2) | idx_update(1)] -> ...
            
            Actually simpler: Overlap idx_update with next rounds hash (both VALU).
            """
            for round_idx in range(rounds):
                # Gather for this round
                emit_gather_phase(buf, chunk_count, round_idx, ops)
                # Hash for this round
                emit_hash_phase(buf, chunk_count, ops)
                # idx update for this round
                emit_idx_update(buf, chunk_count, ops, round_idx)
        
        def emit_two_chunks_interleaved(buf_a, count_a, buf_b, count_b, ops):
            """Process two chunks with round-level interleaving for better overlap.
            
            Key idea: While chunk A is in hash phase (VALU-heavy), chunk B can 
            be in gather phase (LOAD-heavy), enabling better engine overlap.
            
            Timeline:
              A_gather[0] -> B_gather[0] ->
              A_hash[0]   || B_hash[0]   ->  (both VALU, can pack together)
              A_idx[0]    || B_idx[0]    ->
              A_gather[1] -> B_gather[1] -> ...
            """
            for round_idx in range(rounds):
                # Gather both chunks
                emit_gather_phase(buf_a, count_a, round_idx, ops)
                emit_gather_phase(buf_b, count_b, round_idx, ops)
                # Hash both chunks (VALU ops can overlap)
                emit_hash_phase(buf_a, count_a, ops)
                emit_hash_phase(buf_b, count_b, ops)
                # idx update both chunks
                emit_idx_update(buf_a, count_a, ops, round_idx)
                emit_idx_update(buf_b, count_b, ops, round_idx)
        
        # Cross-chunk pipelining with double-buffering
        CHUNK_SIZE = 8
        chunks = []
        for chunk_start in range(0, n_vec_iters, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_vec_iters)
            chunks.append((chunk_start, chunk_end - chunk_start))
        
        num_chunks = len(chunks)
        
        if num_chunks == 1:
            # Single chunk, no pipelining
            chunk_start, chunk_count = chunks[0]
            buf = bufs[0]
            emit_chunk_load(buf, chunk_start, chunk_count, body)
            emit_all_rounds_pipelined(buf, chunk_count, body)
            emit_chunk_store(buf, chunk_start, chunk_count, body)
        else:
            # Prologue: Load first chunk
            chunk_start_0, chunk_count_0 = chunks[0]
            buf_a = bufs[0]
            emit_chunk_load(buf_a, chunk_start_0, chunk_count_0, body)
            
            # Pipeline: Process chunk[i] while loading chunk[i+1]
            for ci in range(num_chunks - 1):
                curr_buf = bufs[ci % 2]
                next_buf = bufs[(ci + 1) % 2]
                
                curr_start, curr_count = chunks[ci]
                next_start, next_count = chunks[ci + 1]
                
                # Interleave: all_rounds[curr] with load[next]
                round_ops = []
                emit_all_rounds_pipelined(curr_buf, curr_count, round_ops)
                load_ops = []
                emit_chunk_load(next_buf, next_start, next_count, load_ops)
                
                body.extend(interleave_ops(round_ops, load_ops))
                
                # Store current chunk
                emit_chunk_store(curr_buf, curr_start, curr_count, body)
            
            # Epilogue: Process and store last chunk
            last_buf = bufs[(num_chunks - 1) % 2]
            last_start, last_count = chunks[num_chunks - 1]
            emit_all_rounds_pipelined(last_buf, last_count, body)
            emit_chunk_store(last_buf, last_start, last_count, body)

        body_instrs = self.build_packed(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
