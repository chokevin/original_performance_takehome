"""
Cycle profiler for the VLIW simulator.
Shows where cycles are being spent in the kernel, broken down by phase.
"""

from collections import defaultdict
from problem import Machine, DebugInfo, SLOT_LIMITS


class CycleProfiler:
    """
    Wraps the Machine to profile cycle usage by instruction type, engine, and phase.
    """
    
    def __init__(self):
        self.engine_cycles = defaultdict(int)  # cycles spent per engine
        self.op_cycles = defaultdict(int)      # cycles spent per operation
        self.slot_usage = defaultdict(int)     # how many slots used per engine
        self.total_slots_available = 0
        self.total_slots_used = 0
        self.instruction_count = 0
        self.cycle_log = []  # detailed log: [(cycle, engine, op, slots_used), ...]
        
        # Phase tracking
        self.current_phase = "init"
        self.phase_cycles = defaultdict(int)
        self.phase_ops = defaultdict(lambda: defaultdict(int))
        self.phase_start_cycle = 0
        self.phase_engine_cycles = defaultdict(lambda: defaultdict(int))
        self.phase_slot_usage = defaultdict(lambda: defaultdict(int))
        self.phase_history = []  # [(phase_name, start_cycle, end_cycle), ...]
        
    def set_phase(self, phase_name, cycle):
        """Mark the start of a new phase."""
        if self.current_phase:
            self.phase_history.append((self.current_phase, self.phase_start_cycle, cycle))
        self.current_phase = phase_name
        self.phase_start_cycle = cycle
        
    def profile_instruction(self, instr, cycle):
        """Profile a single instruction bundle."""
        self.instruction_count += 1
        phase = self.current_phase
        
        for engine, slots in instr.items():
            if engine == "debug":
                continue
                
            n_slots = len(slots)
            self.engine_cycles[engine] += 1
            self.slot_usage[engine] += n_slots
            self.total_slots_used += n_slots
            self.total_slots_available += SLOT_LIMITS.get(engine, 1)
            
            # Phase tracking
            self.phase_engine_cycles[phase][engine] += 1
            self.phase_slot_usage[phase][engine] += n_slots
            
            for slot in slots:
                op = slot[0] if slot else "unknown"
                self.op_cycles[(engine, op)] += 1
                self.phase_ops[phase][(engine, op)] += 1
                self.cycle_log.append((cycle, engine, op, slot, phase))
        
        self.phase_cycles[phase] += 1
    
    def print_summary(self):
        """Print a summary of cycle usage."""
        print("\n" + "="*70)
        print("CYCLE PROFILER SUMMARY")
        print("="*70)
        
        print(f"\nTotal instructions: {self.instruction_count}")
        print(f"Total slots used: {self.total_slots_used}")
        print(f"Total slots available: {self.total_slots_available}")
        if self.total_slots_available > 0:
            print(f"Slot utilization: {100*self.total_slots_used/self.total_slots_available:.1f}%")
        
        print("\n--- Cycles per Engine ---")
        for engine, cycles in sorted(self.engine_cycles.items(), key=lambda x: -x[1]):
            limit = SLOT_LIMITS.get(engine, 1)
            avg_slots = self.slot_usage[engine] / cycles if cycles > 0 else 0
            print(f"  {engine:10s}: {cycles:6d} cycles, {self.slot_usage[engine]:6d} slots, "
                  f"avg {avg_slots:.2f}/{limit} slots/cycle")
        
        print("\n--- Top Operations by Count ---")
        sorted_ops = sorted(self.op_cycles.items(), key=lambda x: -x[1])[:20]
        for (engine, op), count in sorted_ops:
            print(f"  {engine:6s}.{op:15s}: {count:6d}")
        
        print("="*70 + "\n")
    
    def print_phase_summary(self):
        """Print breakdown by phase."""
        print("\n" + "="*70)
        print("PHASE BREAKDOWN")
        print("="*70)
        
        total = sum(self.phase_cycles.values())
        
        for phase in sorted(self.phase_cycles.keys(), key=lambda p: -self.phase_cycles[p]):
            cycles = self.phase_cycles[phase]
            pct = 100 * cycles / total if total > 0 else 0
            print(f"\n[{phase}] {cycles:,} cycles ({pct:.1f}%)")
            print("-" * 50)
            
            # Engine breakdown for this phase
            phase_engines = self.phase_engine_cycles[phase]
            for engine, eng_cycles in sorted(phase_engines.items(), key=lambda x: -x[1]):
                limit = SLOT_LIMITS.get(engine, 1)
                slots = self.phase_slot_usage[phase][engine]
                avg_slots = slots / eng_cycles if eng_cycles > 0 else 0
                print(f"  {engine:10s}: {eng_cycles:6d} cycles, avg {avg_slots:.2f}/{limit} slots")
            
            # Top ops for this phase
            phase_op_counts = self.phase_ops[phase]
            if phase_op_counts:
                print(f"  Top ops:")
                sorted_ops = sorted(phase_op_counts.items(), key=lambda x: -x[1])[:5]
                for (engine, op), count in sorted_ops:
                    print(f"    {engine}.{op}: {count:,}")
        
        print("\n" + "="*70)
    
    def print_cycle_breakdown(self, start=0, end=None):
        """Print detailed cycle-by-cycle breakdown."""
        if end is None:
            end = len(self.cycle_log)
        
        print(f"\n--- Cycle Breakdown (cycles {start}-{end}) ---")
        current_cycle = -1
        for cycle, engine, op, slot, phase in self.cycle_log[start:end]:
            if cycle != current_cycle:
                if current_cycle >= 0:
                    print()
                print(f"Cycle {cycle} [{phase}]:")
                current_cycle = cycle
            print(f"  {engine:6s}: {op:15s} {slot}")


class ProfiledMachine(Machine):
    """
    Machine subclass that profiles cycle usage.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = CycleProfiler()
    
    def step(self, instr, core):
        """Override step to add profiling."""
        # Profile before executing
        has_non_debug = any(name != "debug" for name in instr.keys())
        if has_non_debug:
            self.profiler.profile_instruction(instr, self.cycle)
        
        # Call parent step
        super().step(instr, core)
    
    def set_phase(self, phase_name):
        """Set the current profiling phase."""
        self.profiler.set_phase(phase_name, self.cycle)
    
    def print_profile(self):
        """Print the profiling summary."""
        self.profiler.print_summary()
        self.profiler.print_phase_summary()


class PhaseTrackingKernelBuilder:
    """
    KernelBuilder that inserts phase markers for profiling.
    """
    def __init__(self):
        from perf_takehome import KernelBuilder
        self.kb = KernelBuilder()
        self.phase_markers = []  # [(pc, phase_name), ...]
        
    def __getattr__(self, name):
        return getattr(self.kb, name)
    
    def mark_phase(self, phase_name):
        """Mark current instruction as start of a phase."""
        self.phase_markers.append((len(self.kb.instrs), phase_name))
    
    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        """Build kernel with phase markers."""
        from problem import HASH_STAGES
        
        self.mark_phase("init")
        
        tmp1 = self.kb.alloc_scratch("tmp1")
        tmp2 = self.kb.alloc_scratch("tmp2")
        tmp3 = self.kb.alloc_scratch("tmp3")
        
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.kb.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.kb.add("load", ("const", tmp1, i))
            self.kb.add("load", ("load", self.kb.scratch[v], tmp1))

        zero_const = self.kb.scratch_const(0)
        one_const = self.kb.scratch_const(1)
        two_const = self.kb.scratch_const(2)

        self.kb.add("flow", ("pause",))
        self.kb.add("debug", ("comment", "Starting loop"))

        tmp_idx = self.kb.alloc_scratch("tmp_idx")
        tmp_val = self.kb.alloc_scratch("tmp_val")
        tmp_node_val = self.kb.alloc_scratch("tmp_node_val")
        tmp_addr = self.kb.alloc_scratch("tmp_addr")

        for round in range(rounds):
            self.mark_phase(f"round_{round}")
            for i in range(batch_size):
                if i == 0:
                    self.mark_phase(f"round_{round}_batch")
                
                i_const = self.kb.scratch_const(i)
                
                # Load idx and val
                body = []
                body.append(("alu", ("+", tmp_addr, self.kb.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                body.append(("alu", ("+", tmp_addr, self.kb.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                
                # Load node_val
                body.append(("alu", ("+", tmp_addr, self.kb.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                
                # Hash
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    body.append(("alu", (op1, tmp1, tmp_val, self.kb.scratch_const(val1))))
                    body.append(("alu", (op3, tmp2, tmp_val, self.kb.scratch_const(val3))))
                    body.append(("alu", (op2, tmp_val, tmp1, tmp2)))
                    body.append(("debug", ("compare", tmp_val, (round, i, "hash_stage", hi))))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                
                # Next index calculation
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                
                # Wrap index
                body.append(("alu", ("<", tmp1, tmp_idx, self.kb.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                
                # Store results
                body.append(("alu", ("+", tmp_addr, self.kb.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                body.append(("alu", ("+", tmp_addr, self.kb.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

                body_instrs = self.kb.build(body)
                self.kb.instrs.extend(body_instrs)

        self.kb.instrs.append({"flow": [("pause",)]})
        
        return self.phase_markers


def profile_kernel(forest_height=10, rounds=16, batch_size=256, seed=123):
    """
    Run the kernel with profiling enabled.
    Uses the actual KernelBuilder from perf_takehome.py.
    """
    import random
    from problem import Tree, Input, build_mem_image, reference_kernel2, N_CORES
    from perf_takehome import KernelBuilder
    
    print(f"Profiling kernel: {forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    value_trace = {}
    machine = ProfiledMachine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
    )
    
    # Run through the kernel
    for ref_mem in reference_kernel2(mem, value_trace):
        machine.run()
    
    print(f"\nTotal cycles: {machine.cycle}")
    machine.print_profile()
    
    return machine


if __name__ == "__main__":
    profile_kernel()
