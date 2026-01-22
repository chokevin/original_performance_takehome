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
        Operation-batched SIMD implementation.
        
        Optimizations:
        1. VALU (8-wide SIMD) to process 8 batch items at once
        2. VLIW packing to combine independent ops into same cycle
        3. Operation batching: batch 16 vector iterations together
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
        self.add("debug", ("comment", "Starting batched SIMD loop"))

        body = []

        # Number of vector iterations to batch together  
        # 16 balances scratch usage with packing efficiency
        BATCH_ITERS = 16
        
        # Allocate vectors for BATCH_ITERS iterations
        v_idx = [self.alloc_scratch(f"v_idx_{j}", VLEN) for j in range(BATCH_ITERS)]
        v_val = [self.alloc_scratch(f"v_val_{j}", VLEN) for j in range(BATCH_ITERS)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{j}", VLEN) for j in range(BATCH_ITERS)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{j}", VLEN) for j in range(BATCH_ITERS)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{j}", VLEN) for j in range(BATCH_ITERS)]
        v_tmp3 = [self.alloc_scratch(f"v_tmp3_{j}", VLEN) for j in range(BATCH_ITERS)]
        
        # Scalar address registers (only need VLEN for extracting from vector addresses)
        gather_addrs = [self.alloc_scratch(f"ga_{vi}") for vi in range(VLEN)]
        # Vector gather addresses (compute all 8 addresses at once with VALU)
        v_gather_addrs = [self.alloc_scratch(f"v_ga_{j}", VLEN) for j in range(BATCH_ITERS)]
        addr_regs = [self.alloc_scratch(f"addr_{j}") for j in range(BATCH_ITERS)]
        
        # Vector constants (shared)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        # Pre-allocate hash constants
        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1 = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
            const2 = self.alloc_scratch(f"v_hc2_{hi}", VLEN)
            v_hash_consts.append((const1, const2))
        
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

        n_vec_iters = batch_size // VLEN
        
        # Pre-compute idx and val addresses for all batch positions (reused every round)
        # This avoids recomputing inp_indices_p + offset and inp_values_p + offset each round
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(n_vec_iters)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(n_vec_iters)]
        
        # Compute all addresses once before the round loop
        for i in range(n_vec_iters):
            offset = i * VLEN
            body.append(("alu", ("+", idx_addrs[i], self.scratch["inp_indices_p"], self.scratch_const(offset))))
            body.append(("alu", ("+", val_addrs[i], self.scratch["inp_values_p"], self.scratch_const(offset))))
        
        # Cache for first tree levels - at round R, all indices are in range [2^R - 1, 2^(R+1) - 2]
        # Round 0: idx=0 for all items
        # Round 1: idx in {1, 2}
        # Round 2: idx in {3, 4, 5, 6}
        CACHE_SIZE = 7  # nodes 0-6 (3 levels)
        v_forest_cache = [self.alloc_scratch(f"v_fcache_{i}", VLEN) for i in range(CACHE_SIZE)]
        forest_cache_scalar = self.alloc_scratch("fcache_scalar")
        
        # Load first CACHE_SIZE forest values into vector scratch (broadcast each)
        for i in range(CACHE_SIZE):
            body.append(("alu", ("+", addr_regs[0], self.scratch["forest_values_p"], self.scratch_const(i))))
            body.append(("load", ("load", forest_cache_scalar, addr_regs[0])))
            body.append(("valu", ("vbroadcast", v_forest_cache[i], forest_cache_scalar)))
        
        # Constants for round 2 index selection
        v_three = self.alloc_scratch("v_three", VLEN)
        body.append(("valu", ("vbroadcast", v_three, self.scratch_const(3))))
        
        for round in range(rounds):
            for batch_start in range(0, n_vec_iters, BATCH_ITERS):
                batch_end = min(batch_start + BATCH_ITERS, n_vec_iters)
                batch_count = batch_end - batch_start
                
                if round == 0:
                    # Round 0: all idx=0, all items use forest[0]
                    # Skip loading idx (use v_zero), skip gather (use cached forest[0])
                    
                    # Phase 1-2: idx is zero for all items
                    for j in range(batch_count):
                        body.append(("valu", ("+", v_idx[j], v_zero, v_zero)))  # v_idx = 0
                    
                    # Phase 4: vload ALL val (use precomputed addresses)
                    for j in range(batch_count):
                        body.append(("load", ("vload", v_val[j], val_addrs[batch_start + j])))
                    
                    # Phase 5-6: Use cached forest[0] instead of gather
                    for j in range(batch_count):
                        body.append(("valu", ("+", v_node_val[j], v_forest_cache[0], v_zero)))  # copy cache
                        
                elif round == 1:
                    # Round 1: idx in {1, 2} - use cache with vselect
                    
                    # Phase 2: vload ALL idx (use precomputed addresses)
                    for j in range(batch_count):
                        body.append(("load", ("vload", v_idx[j], idx_addrs[batch_start + j])))
                    
                    # Phase 4: vload ALL val (use precomputed addresses)
                    for j in range(batch_count):
                        body.append(("load", ("vload", v_val[j], val_addrs[batch_start + j])))
                    
                    # Phase 5-6: idx is 1 or 2, select from cache
                    # vselect: dest = a if cond else b
                    # idx=1: idx&1=1 (nonzero) -> select a=cache[1] ✓
                    # idx=2: idx&1=0 -> select b=cache[2] ✓
                    for j in range(batch_count):
                        body.append(("valu", ("&", v_tmp1[j], v_idx[j], v_one)))  # idx & 1
                    for j in range(batch_count):
                        body.append(("flow", ("vselect", v_node_val[j], v_tmp1[j], v_forest_cache[1], v_forest_cache[2])))
                
                elif round == 2:
                    # Round 2: idx in {3, 4, 5, 6} - use cache with multiple selects
                    
                    # Phase 2: vload ALL idx (use precomputed addresses)
                    for j in range(batch_count):
                        body.append(("load", ("vload", v_idx[j], idx_addrs[batch_start + j])))
                    
                    # Phase 4: vload ALL val (use precomputed addresses)
                    for j in range(batch_count):
                        body.append(("load", ("vload", v_val[j], val_addrs[batch_start + j])))
                    
                    # Phase 5-6: idx in {3,4,5,6}, select from cache[3..6]
                    # idx-3 gives {0,1,2,3}
                    # bit 1: selects between {3,4} and {5,6}
                    # bit 0: selects within pair
                    for j in range(batch_count):
                        body.append(("valu", ("-", v_tmp1[j], v_idx[j], v_three)))  # idx - 3 -> {0,1,2,3}
                    
                    # First level: select between cache[3]/cache[4] vs cache[5]/cache[6]
                    # tmp1 & 2 gives 0 for idx∈{3,4}, 2 for idx∈{5,6}
                    for j in range(batch_count):
                        body.append(("valu", ("&", v_tmp2[j], v_tmp1[j], v_two)))  # (idx-3) & 2
                    
                    # Select first pair based on bit 1
                    for j in range(batch_count):
                        body.append(("flow", ("vselect", v_tmp3[j], v_tmp2[j], v_forest_cache[5], v_forest_cache[3])))  # 5 if bit1, else 3
                    
                    # Select second of pair
                    for j in range(batch_count):
                        body.append(("flow", ("vselect", v_node_val[j], v_tmp2[j], v_forest_cache[6], v_forest_cache[4])))  # 6 if bit1, else 4
                    
                    # Final select based on bit 0
                    for j in range(batch_count):
                        body.append(("valu", ("&", v_tmp2[j], v_tmp1[j], v_one)))  # (idx-3) & 1
                    for j in range(batch_count):
                        body.append(("flow", ("vselect", v_node_val[j], v_tmp2[j], v_node_val[j], v_tmp3[j])))
                        
                else:
                    # Round 2+: full gather loads
                    
                    # Phase 2: vload ALL idx (use precomputed addresses)
                    for j in range(batch_count):
                        body.append(("load", ("vload", v_idx[j], idx_addrs[batch_start + j])))
                    
                    # Phase 4: vload ALL val (use precomputed addresses)
                    for j in range(batch_count):
                        body.append(("load", ("vload", v_val[j], val_addrs[batch_start + j])))
                    
                    # Phase 5: Compute ALL gather addresses using VALU (8 at a time!)
                    # v_gather_addrs[j] = v_forest_p + v_idx[j]
                    for j in range(batch_count):
                        body.append(("valu", ("+", v_gather_addrs[j], v_forest_p, v_idx[j])))
                    
                    # Phase 6: ALL gather loads (extract individual addresses from vector)
                    for j in range(batch_count):
                        for vi in range(VLEN):
                            body.append(("load", ("load", v_node_val[j] + vi, v_gather_addrs[j] + vi)))
                
                # Phase 7: ALL XORs
                for j in range(batch_count):
                    body.append(("valu", ("^", v_val[j], v_val[j], v_node_val[j])))
                
                # Phase 8: Hash - batch each stage
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    const1, const2 = v_hash_consts[hi]
                    for j in range(batch_count):
                        body.append(("valu", (op1, v_tmp1[j], v_val[j], const1)))
                    for j in range(batch_count):
                        body.append(("valu", (op3, v_tmp2[j], v_val[j], const2)))
                    for j in range(batch_count):
                        body.append(("valu", (op2, v_val[j], v_tmp1[j], v_tmp2[j])))
                
                # Phase 9: offset = 1 + (val & 1)
                # Replaces: val%2==0 ? 1 : 2 with pure ALU
                # val&1 = 0 if even, 1 if odd
                # 1 + (val&1) = 1 if even, 2 if odd ✓
                for j in range(batch_count):
                    body.append(("valu", ("&", v_tmp1[j], v_val[j], v_one)))  # val & 1
                for j in range(batch_count):
                    body.append(("valu", ("+", v_tmp3[j], v_tmp1[j], v_one)))  # 1 + (val & 1)
                
                # Phase 10: idx = 2*idx + offset
                for j in range(batch_count):
                    body.append(("valu", ("*", v_idx[j], v_idx[j], v_two)))
                for j in range(batch_count):
                    body.append(("valu", ("+", v_idx[j], v_idx[j], v_tmp3[j])))
                
                # Phase 11: idx < n_nodes
                for j in range(batch_count):
                    body.append(("valu", ("<", v_tmp1[j], v_idx[j], v_n_nodes)))
                
                # Phase 12: idx = idx * (idx < n_nodes)
                # Replaces: idx >= n_nodes ? 0 : idx with pure ALU
                # If idx < n_nodes: tmp1=1, idx*1=idx ✓
                # If idx >= n_nodes: tmp1=0, idx*0=0 ✓
                for j in range(batch_count):
                    body.append(("valu", ("*", v_idx[j], v_idx[j], v_tmp1[j])))
                
                # Phase 13-14: Store idx (use precomputed addresses)
                for j in range(batch_count):
                    body.append(("store", ("vstore", idx_addrs[batch_start + j], v_idx[j])))
                
                # Phase 15-16: Store val (use precomputed addresses)
                for j in range(batch_count):
                    body.append(("store", ("vstore", val_addrs[batch_start + j], v_val[j])))

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
