"""
Read the top of perf_takehome.py for more introduction.

This file is separate mostly for ease of copying it to freeze the machine and
reference kernel for testing.
"""

from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
import json
import random

Engine = Literal["alu", "load", "store", "flow"]
Instruction = dict[Engine, list[tuple]]


class CoreState(Enum):
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


@dataclass
class Core:
    id: int
    scratch: list[int]
    trace_buf: list[int]
    pc: int = 0
    state: CoreState = CoreState.RUNNING


@dataclass
class DebugInfo:
    """
    We give you some debug info but it's up to you to use it in Machine if you
    want to. You're also welcome to add more.
    """

    # Maps scratch variable addr to (name, len) pair
    scratch_map: dict[int, (str, int)]
    # Optional profile-only metadata: pc -> (engine, slot_index) -> phase label.
    profile_slot_phases: dict | None = None


def cdiv(a, b):
    return (a + b - 1) // b


SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
    "debug": 64,
}

VLEN = 8
# Older versions of the take-home used multiple cores, but this version only uses 1
N_CORES = 1
SCRATCH_SIZE = 1536
BASE_ADDR_TID = 100000


class Machine:
    """
    Simulator for a custom VLIW SIMD architecture.

    VLIW (Very Large Instruction Word): Cores are composed of different
    "engines" each of which can execute multiple "slots" per cycle in parallel.
    How many slots each engine can execute per cycle is limited by SLOT_LIMITS.
    Effects of instructions don't take effect until the end of cycle. Each
    cycle, all engines execute all of their filled slots for that instruction.
    Effects like writes to memory take place after all the inputs are read.

    SIMD: There are instructions for acting on vectors of VLEN elements in a
    single slot. You can use vload and vstore to load multiple contiguous
    elements but not non-contiguous elements. Use vbroadcast to broadcast a
    scalar to a vector and then operate on vectors with valu instructions.

    The memory and scratch space are composed of 32-bit words. The solution is
    plucked out of the memory at the end of the program. You can think of the
    scratch space as serving the purpose of registers, constant memory, and a
    manually-managed cache.

    Here's an example of what an instruction might look like:

    {"valu": [("*", 4, 0, 0), ("+", 8, 4, 0)], "load": [("load", 16, 17)]}

    In general every number in an instruction is a scratch address except for
    const and jump, and except for store and some flow instructions the first
    operand is the destination.

    This comment is not meant to be full ISA documentation though, for the rest
    you should look through the simulator code.
    """

    def __init__(
        self,
        mem_dump: list[int],
        program: list[Instruction],
        debug_info: DebugInfo,
        n_cores: int = 1,
        scratch_size: int = SCRATCH_SIZE,
        trace: bool = False,
        profile: bool = False,
        value_trace: dict[Any, int] = {},
    ):
        self.cores = [
            Core(id=i, scratch=[0] * scratch_size, trace_buf=[]) for i in range(n_cores)
        ]
        self.mem = copy(mem_dump)
        self.program = program
        self.debug_info = debug_info
        self.value_trace = value_trace
        self.prints = False
        self.cycle = 0
        self.enable_pause = True
        self.enable_debug = True
        self.enable_hazard_assert = False
        self.profile = None
        self.profile_scratch_names = {}
        self.profile_written = False
        self.trace = None
        if trace or profile:
            self.setup_profile()
        if trace:
            self.setup_trace()

    def rewrite_instr(self, instr):
        """
        Rewrite an instruction to use scratch addresses instead of names
        """
        res = {}
        for name, slots in instr.items():
            res[name] = []
            for slot in slots:
                res[name].append(self.rewrite_slot(slot))
        return res

    def print_step(self, instr, core):
        # print(core.id)
        # print(core.trace_buf)
        print(self.scratch_map(core))
        print(core.pc, instr, self.rewrite_instr(instr))

    def scratch_map(self, core):
        res = {}
        for addr, (name, length) in self.debug_info.scratch_map.items():
            res[name] = core.scratch[addr : addr + length]
        return res

    def rewrite_slot(self, slot):
        return tuple(
            self.debug_info.scratch_map.get(s, (None, None))[0] or s for s in slot
        )

    def setup_profile(self):
        self.profile_scratch_names = {}
        for base, (name, length) in self.debug_info.scratch_map.items():
            for offset in range(length):
                label = name if length == 1 else f"{name}[{offset}]"
                self.profile_scratch_names[base + offset] = label

        slot_capacity = sum(
            limit for name, limit in SLOT_LIMITS.items() if name != "debug"
        )
        self.profile = {
            "slot_capacity": slot_capacity,
            "bundle_count": 0,
            "used_slot_hist": Counter(),
            "engine_slots": Counter(),
            "op_counts": Counter(),
            "pc_counts": Counter(),
            "pc_slots": Counter(),
            "pc_engine_slots": defaultdict(Counter),
            "active_engine_combos": Counter(),
            "scratch_reads": Counter(),
            "scratch_writes": Counter(),
            "touched_scratch": set(),
            "scratch_first_touch": {},
            "scratch_last_touch": {},
            "bundle_scratch_touches": Counter(),
            "last_writer": {},
            "dependency_distance_hist": Counter(),
            "immediate_dependencies": Counter(),
            "memory_ops": Counter(),
            "memory_regions": Counter(),
            "memory_strides": Counter(),
            "last_memory_addr": {},
            "memory_samples": [],
            "select_results": defaultdict(Counter),
            "slot_shapes": Counter(),
            "scalar_slot_shapes": Counter(),
            "vector_slot_shapes": Counter(),
            "phase_markers": Counter(),
            "phase_bundles": Counter(),
            "phase_bundle_slots": Counter(),
            "phase_bundle_load_slots": Counter(),
            "phase_load_unused_slots": Counter(),
            "phase_load_idle_cycles": Counter(),
            "phase_engine_slots": defaultdict(Counter),
            "phase_op_counts": defaultdict(Counter),
            "phase_memory_ops": defaultdict(Counter),
            "phase_memory_regions": defaultdict(Counter),
            "hazard_checks": Counter(),
            "hazard_counts": Counter(),
            "hazard_samples": [],
        }

    def scratch_label(self, addr):
        return self.profile_scratch_names.get(addr, f"scratch[{addr}]")

    def memory_region(self, addr):
        n_nodes = self.mem[1]
        batch_size = self.mem[2]
        forest_values_p = self.mem[4]
        inp_indices_p = self.mem[5]
        inp_values_p = self.mem[6]
        inp_values_end = inp_values_p + batch_size
        if addr < 0 or addr >= len(self.mem):
            return "out_of_bounds"
        if addr < forest_values_p:
            return "header"
        if addr < inp_indices_p:
            return "forest_values"
        if addr < inp_values_p:
            return "inp_indices"
        if addr < inp_values_end:
            return "inp_values"
        return "extra"

    def stride_bucket(self, stride):
        if -16 <= stride <= 16:
            return str(stride)
        return "<-16" if stride < -16 else ">16"

    def dependency_distance_bucket(self, distance):
        if distance <= 16:
            return str(distance)
        if distance <= 64:
            return "17-64"
        if distance <= 256:
            return "65-256"
        if distance <= 1024:
            return "257-1024"
        return ">1024"

    def profile_slot_shape(self, engine, slot):
        parts = [engine, str(slot[0])]
        for i, item in enumerate(slot[1:], start=1):
            if not isinstance(item, int):
                parts.append(str(item))
            elif engine == "load" and slot[0] == "const" and i == 2:
                parts.append("imm")
            elif engine == "flow" and slot[0] in {
                "jump",
                "cond_jump_rel",
                "cond_jump",
            } and i == len(slot) - 1:
                parts.append("target")
            else:
                parts.append(self.scratch_label(item))
        return " ".join(parts)

    def slot_reads_writes(self, engine, slot):
        def span(start, length):
            return list(range(start, start + length))

        if engine == "alu":
            _, dest, a1, a2 = slot
            return [a1, a2], [dest]
        if engine == "valu":
            match slot:
                case ("vbroadcast", dest, src):
                    return [src], span(dest, VLEN)
                case ("multiply_add", dest, a, b, c):
                    return span(a, VLEN) + span(b, VLEN) + span(c, VLEN), span(
                        dest, VLEN
                    )
                case (_, dest, a1, a2):
                    return span(a1, VLEN) + span(a2, VLEN), span(dest, VLEN)
        if engine == "load":
            match slot:
                case ("load", dest, addr):
                    return [addr], [dest]
                case ("load_offset", dest, addr, offset):
                    return [addr + offset], [dest + offset]
                case ("vload", dest, addr):
                    return [addr], span(dest, VLEN)
                case ("const", dest, _):
                    return [], [dest]
        if engine == "store":
            match slot:
                case ("store", addr, src):
                    return [addr, src], []
                case ("vstore", addr, src):
                    return [addr] + span(src, VLEN), []
        if engine == "flow":
            match slot:
                case ("select", dest, cond, a, b):
                    return [cond, a, b], [dest]
                case ("add_imm", dest, a, _):
                    return [a], [dest]
                case ("vselect", dest, cond, a, b):
                    return span(cond, VLEN) + span(a, VLEN) + span(b, VLEN), span(
                        dest, VLEN
                    )
                case ("trace_write", val):
                    return [val], []
                case ("cond_jump", cond, _) | ("cond_jump_rel", cond, _):
                    return [cond], []
                case ("jump_indirect", addr):
                    return [addr], []
                case ("coreid", dest):
                    return [], [dest]
                case ("halt",) | ("pause",) | ("jump", _):
                    return [], []
        return [], []

    def slot_memory_accesses(self, core, engine, slot):
        def span(start, length):
            return list(range(start, start + length))

        if engine == "load":
            match slot:
                case ("load", _, addr):
                    return [core.scratch[addr]], []
                case ("load_offset", _, addr, offset):
                    return [core.scratch[addr + offset]], []
                case ("vload", _, addr):
                    return span(core.scratch[addr], VLEN), []
                case ("const", _, _):
                    return [], []
        if engine == "store":
            match slot:
                case ("store", addr, _):
                    return [], [core.scratch[addr]]
                case ("vstore", addr, _):
                    return [], span(core.scratch[addr], VLEN)
        return [], []

    def bundle_slot_descriptors(self, core, instr):
        descs = {}
        for engine, slots in instr.items():
            if engine == "debug":
                continue
            for slot_index, slot in enumerate(slots):
                scratch_reads, scratch_writes = self.slot_reads_writes(engine, slot)
                memory_reads, memory_writes = self.slot_memory_accesses(
                    core, engine, slot
                )
                descs[(engine, slot_index)] = {
                    "engine": engine,
                    "slot_index": slot_index,
                    "slot": slot,
                    "scratch_reads": scratch_reads,
                    "scratch_writes": scratch_writes,
                    "memory_reads": memory_reads,
                    "memory_writes": memory_writes,
                }
        return descs

    def slot_descriptor_label(self, desc):
        shape = self.profile_slot_shape(desc["engine"], desc["slot"])
        return f"{desc['engine']}[{desc['slot_index']}] {shape}"

    def record_bundle_hazard(self, kind, pc, core, addr, first, second):
        if kind.startswith("scratch"):
            location = self.scratch_label(addr)
        else:
            location = f"mem[{addr}] ({self.memory_region(addr)})"
        message = (
            f"{kind} at pc {pc}, cycle {self.cycle}, {location}: "
            f"{self.slot_descriptor_label(first)} <-> "
            f"{self.slot_descriptor_label(second)}"
        )

        if self.profile is not None:
            self.profile["hazard_counts"][kind] += 1
            if len(self.profile["hazard_samples"]) < 64:
                self.profile["hazard_samples"].append(
                    {
                        "kind": kind,
                        "cycle": self.cycle,
                        "pc": pc,
                        "core": core.id,
                        "location": location,
                        "first": self.slot_descriptor_label(first),
                        "second": self.slot_descriptor_label(second),
                        "message": message,
                    }
                )

        if self.enable_hazard_assert:
            raise AssertionError(message)

    def check_bundle_hazards(self, core, pc, descs):
        descs = list(descs)
        if self.profile is not None:
            self.profile["hazard_checks"]["bundles_checked"] += 1
            if len(descs) > 1:
                self.profile["hazard_checks"]["multi_slot_bundles_checked"] += 1
        if len(descs) <= 1:
            return

        for i, first in enumerate(descs):
            for second in descs[i + 1 :]:
                first_writes = set(first["scratch_writes"])
                second_writes = set(second["scratch_writes"])
                first_reads = set(first["scratch_reads"])
                second_reads = set(second["scratch_reads"])

                for addr in sorted(first_writes & second_writes):
                    self.record_bundle_hazard(
                        "scratch_write_write", pc, core, addr, first, second
                    )
                for addr in sorted((first_writes & second_reads) | (second_writes & first_reads)):
                    self.record_bundle_hazard(
                        "scratch_read_write", pc, core, addr, first, second
                    )

                first_mem_writes = set(first["memory_writes"])
                second_mem_writes = set(second["memory_writes"])
                first_mem_reads = set(first["memory_reads"])
                second_mem_reads = set(second["memory_reads"])

                for addr in sorted(first_mem_writes & second_mem_writes):
                    self.record_bundle_hazard(
                        "memory_write_write", pc, core, addr, first, second
                    )
                for addr in sorted(
                    (first_mem_writes & second_mem_reads)
                    | (second_mem_writes & first_mem_reads)
                ):
                    self.record_bundle_hazard(
                        "memory_read_write", pc, core, addr, first, second
                    )

    def profile_record_bundle(self, core, pc, instr):
        if self.profile is None:
            return
        used_by_engine = {
            name: len(instr.get(name, [])) for name in SLOT_LIMITS if name != "debug"
        }
        used_slots = sum(used_by_engine.values())
        if used_slots == 0:
            return
        active_engines = tuple(name for name, count in used_by_engine.items() if count)
        p = self.profile
        p["bundle_count"] += 1
        p["used_slot_hist"][used_slots] += 1
        p["pc_counts"][pc] += 1
        p["pc_slots"][pc] += used_slots
        p["active_engine_combos"]["+".join(active_engines)] += 1
        for name, count in used_by_engine.items():
            p["engine_slots"][name] += count
            p["pc_engine_slots"][pc][name] += count
        for phase in self.profile_bundle_phases(pc, instr):
            p["phase_bundles"][phase] += 1
            p["phase_bundle_slots"][phase] += used_slots
            p["phase_bundle_load_slots"][phase] += used_by_engine["load"]
            unused_load_slots = SLOT_LIMITS["load"] - used_by_engine["load"]
            p["phase_load_unused_slots"][phase] += unused_load_slots
            if unused_load_slots:
                p["phase_load_idle_cycles"][phase] += 1

    def profile_slot_phase(self, pc, engine, slot_index):
        slot_phases = getattr(self.debug_info, "profile_slot_phases", None) or {}
        return slot_phases.get(pc, {}).get((engine, slot_index), "unattributed")

    def profile_bundle_phases(self, pc, instr):
        phases = set()
        for engine, slots in instr.items():
            if engine == "debug":
                continue
            for slot_index, _ in enumerate(slots):
                phases.add(self.profile_slot_phase(pc, engine, slot_index))
        return sorted(phases)

    def profile_record_slot(self, pc, engine, slot, reads, writes, phase):
        if self.profile is None:
            return []
        p = self.profile
        op = slot[0]
        shape = self.profile_slot_shape(engine, slot)
        p["op_counts"][f"{engine}:{op}"] += 1
        p["phase_engine_slots"][phase][engine] += 1
        p["phase_op_counts"][phase][f"{engine}:{op}"] += 1
        p["slot_shapes"][shape] += 1
        if engine == "valu":
            p["vector_slot_shapes"][shape] += 1
        else:
            p["scalar_slot_shapes"][shape] += 1

        for addr in reads:
            label = self.scratch_label(addr)
            p["scratch_reads"][label] += 1
            p["touched_scratch"].add(addr)
            p["scratch_first_touch"].setdefault(addr, self.cycle)
            p["scratch_last_touch"][addr] = self.cycle
            self.profile_current_scratch_touches.add(addr)
            writer = p["last_writer"].get(addr)
            if writer is None:
                continue
            distance = self.cycle - writer["cycle"]
            p["dependency_distance_hist"][self.dependency_distance_bucket(distance)] += 1
            if distance <= 2:
                dep = (
                    f"pc{writer['pc']} {writer['engine']}:{writer['op']} -> "
                    f"pc{pc} {engine}:{op} via {label}"
                )
                p["immediate_dependencies"][dep] += 1

        pending_writes = []
        for addr in writes:
            label = self.scratch_label(addr)
            p["scratch_writes"][label] += 1
            p["touched_scratch"].add(addr)
            p["scratch_first_touch"].setdefault(addr, self.cycle)
            p["scratch_last_touch"][addr] = self.cycle
            self.profile_current_scratch_touches.add(addr)
            pending_writes.append(
                {
                    "addr": addr,
                    "cycle": self.cycle,
                    "pc": pc,
                    "engine": engine,
                    "op": op,
                }
            )
        return pending_writes

    def profile_finish_bundle(self):
        if self.profile is None:
            return
        self.profile["bundle_scratch_touches"][len(self.profile_current_scratch_touches)] += 1

    def profile_record_marker(self, message):
        if self.profile is not None:
            self.profile["phase_markers"][message] += 1

    def profile_commit_writes(self, pending_writes):
        if self.profile is None:
            return
        for write in pending_writes:
            self.profile["last_writer"][write["addr"]] = write

    def profile_record_memory(self, kind, op, addr, width=1):
        if self.profile is None:
            return
        p = self.profile
        phase = getattr(self, "profile_current_slot_phase", "unattributed")
        p["memory_ops"][f"{kind}:{op}:accesses"] += 1
        p["memory_ops"][f"{kind}:{op}:words"] += width
        p["phase_memory_ops"][phase][f"{kind}:{op}:accesses"] += 1
        p["phase_memory_ops"][phase][f"{kind}:{op}:words"] += width
        first_region = self.memory_region(addr)
        if len(p["memory_samples"]) < 64:
            p["memory_samples"].append(
                {
                    "cycle": self.cycle,
                    "pc": getattr(self, "profile_current_pc", None),
                    "phase": phase,
                    "kind": kind,
                    "op": op,
                    "addr": addr,
                    "width": width,
                    "region": first_region,
                }
            )

        for offset in range(width):
            actual_addr = addr + offset
            region = self.memory_region(actual_addr)
            p["memory_regions"][f"{kind}:{region}"] += 1
            p["phase_memory_regions"][phase][f"{kind}:{region}"] += 1
            stream = (kind, op, region)
            prev_addr = p["last_memory_addr"].get(stream)
            if prev_addr is not None:
                stride = self.stride_bucket(actual_addr - prev_addr)
                p["memory_strides"][f"{kind}:{op}:{region}:stride {stride}"] += 1
            p["last_memory_addr"][stream] = actual_addr

    def profile_record_select(self, op, dest, values):
        if self.profile is None:
            return
        dest_label = self.scratch_label(dest)
        for value in values:
            self.profile["select_results"][f"{op}:{dest_label}"][str(value)] += 1

    def profile_top(self, counter, limit=20):
        return [
            {"name": str(name), "count": count}
            for name, count in counter.most_common(limit)
        ]

    def profile_phase_fields(self, label):
        fields = {}
        for part in label.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if value.isdigit():
                fields[key] = int(value)
            else:
                fields[key] = value
        return fields

    def profile_phase_report(self, top_n=20):
        p = self.profile
        labels = set(p["phase_bundles"])
        labels.update(p["phase_engine_slots"])
        labels.update(p["phase_memory_ops"])
        labels.update(p["phase_memory_regions"])

        phases = []
        total_forest_load_words = 0
        total_forest_load_floor_cycles = 0
        for label in sorted(labels):
            cycles = p["phase_bundles"][label]
            engine_slots = dict(p["phase_engine_slots"][label].most_common())
            memory_ops = dict(p["phase_memory_ops"][label].most_common())
            memory_regions = dict(p["phase_memory_regions"][label].most_common())
            forest_load_words = memory_regions.get("load:forest_values", 0)
            forest_load_floor_cycles = cdiv(forest_load_words, SLOT_LIMITS["load"])
            total_forest_load_words += forest_load_words
            total_forest_load_floor_cycles += forest_load_floor_cycles

            row = {
                "label": label,
                **self.profile_phase_fields(label),
                "cycles": cycles,
                "own_slots": sum(engine_slots.values()),
                "bundle_slots": p["phase_bundle_slots"][label],
                "engine_slots": engine_slots,
                "memory_ops": memory_ops,
                "memory_regions": memory_regions,
                "load_slots_used_in_phase_bundles": p["phase_bundle_load_slots"][label],
                "load_unused_slots_in_phase_bundles": p["phase_load_unused_slots"][label],
                "load_idle_cycles_in_phase_bundles": p["phase_load_idle_cycles"][label],
                "forest_load_words": forest_load_words,
                "forest_load_floor_cycles": forest_load_floor_cycles,
                "cycles_over_forest_load_floor": cycles - forest_load_floor_cycles
                if forest_load_words
                else None,
            }
            phases.append(row)

        return {
            "by_phase": phases,
            "forest_phases": [
                row
                for row in phases
                if row.get("phase") == "forest" or row["forest_load_words"]
            ],
            "top_cycle_phases": sorted(
                phases, key=lambda row: row["cycles"], reverse=True
            )[:top_n],
            "top_load_slack_phases": sorted(
                phases,
                key=lambda row: row["load_unused_slots_in_phase_bundles"],
                reverse=True,
            )[:top_n],
            "forest_load_floor": {
                "words": total_forest_load_words,
                "cycles_at_two_load_slots": total_forest_load_floor_cycles,
            },
        }

    def profile_report(self, top_n=20):
        if self.profile is None:
            return {}
        p = self.profile
        bundle_count = p["bundle_count"]
        used_slots = sum(slots * count for slots, count in p["used_slot_hist"].items())
        capacity = p["slot_capacity"] * bundle_count
        pc_hotness = []
        for pc, count in p["pc_counts"].most_common(top_n):
            pc_hotness.append(
                {
                    "pc": pc,
                    "count": count,
                    "avg_slots": round(p["pc_slots"][pc] / count, 3),
                    "engine_slots": dict(p["pc_engine_slots"][pc]),
                    "instruction": str(self.rewrite_instr(self.program[pc])),
                }
            )
        touch_total = sum(
            touch_count * bundles
            for touch_count, bundles in p["bundle_scratch_touches"].items()
        )
        scratch_ranges = []
        for addr, first in p["scratch_first_touch"].items():
            last = p["scratch_last_touch"][addr]
            scratch_ranges.append(
                {
                    "name": self.scratch_label(addr),
                    "first_cycle": first,
                    "last_cycle": last,
                    "span_cycles": last - first + 1,
                }
            )
        scratch_ranges.sort(key=lambda item: item["span_cycles"], reverse=True)

        return {
            "cycles": self.cycle,
            "program_length": len(self.program),
            "bundle_hazards": {
                "total": sum(p["hazard_counts"].values()),
                "by_kind": dict(p["hazard_counts"].most_common()),
                "checks": dict(p["hazard_checks"].most_common()),
                "samples": p["hazard_samples"],
            },
            "slot_utilization": {
                "bundle_count": bundle_count,
                "slot_capacity_per_bundle": p["slot_capacity"],
                "used_slots": used_slots,
                "available_slots": capacity,
                "avg_slots_per_bundle": round(used_slots / bundle_count, 3)
                if bundle_count
                else 0,
                "avg_utilization_pct": round(100 * used_slots / capacity, 3)
                if capacity
                else 0,
                "histogram": {
                    str(slots): count
                    for slots, count in sorted(p["used_slot_hist"].items())
                },
                "engine_utilization_pct": {
                    name: round(
                        100 * p["engine_slots"][name] / (SLOT_LIMITS[name] * bundle_count),
                        3,
                    )
                    if bundle_count
                    else 0
                    for name in SLOT_LIMITS
                    if name != "debug"
                },
                "active_engine_combos": dict(p["active_engine_combos"].most_common()),
            },
            "instruction_mix": {
                "by_engine": dict(p["engine_slots"].most_common()),
                "by_op": dict(p["op_counts"].most_common()),
            },
            "pc_hotness": pc_hotness,
            "dependencies": {
                "distance_histogram": dict(p["dependency_distance_hist"].most_common()),
                "top_immediate": self.profile_top(p["immediate_dependencies"], top_n),
            },
            "scratch": {
                "touched_words": len(p["touched_scratch"]),
                "operand_pressure": {
                    "max_touched_per_bundle": max(p["bundle_scratch_touches"] or {0: 0}),
                    "avg_touched_per_bundle": round(touch_total / bundle_count, 3)
                    if bundle_count
                    else 0,
                    "histogram": {
                        str(touch_count): bundles
                        for touch_count, bundles in sorted(
                            p["bundle_scratch_touches"].items()
                        )
                    },
                },
                "approx_live_ranges": scratch_ranges[:top_n],
                "top_reads": self.profile_top(p["scratch_reads"], top_n),
                "top_writes": self.profile_top(p["scratch_writes"], top_n),
            },
            "memory": {
                "by_op": dict(p["memory_ops"].most_common()),
                "by_region": dict(p["memory_regions"].most_common()),
                "stride_histogram": dict(p["memory_strides"].most_common()),
                "samples": p["memory_samples"],
            },
            "targeted_phases": self.profile_phase_report(top_n),
            "selects": {
                dest: {
                    "total": sum(values.values()),
                    "distinct_results": len(values),
                    "zero_count": values.get("0", 0),
                    "top_results": self.profile_top(values, top_n),
                }
                for dest, values in p["select_results"].items()
            },
            "vectorization": {
                "valu_slots": p["engine_slots"]["valu"],
                "top_repeated_scalar_shapes": self.profile_top(
                    p["scalar_slot_shapes"], top_n
                ),
                "top_repeated_vector_shapes": self.profile_top(
                    p["vector_slot_shapes"], top_n
                ),
            },
            "phase_markers": dict(p["phase_markers"].most_common()),
        }

    def write_profile(self, path="profile.json"):
        if self.profile is None:
            return None
        report = self.profile_report()
        with open(path, "w", encoding="utf-8") as profile_file:
            json.dump(report, profile_file, indent=2)
            profile_file.write("\n")
        self.profile_written = True
        return report

    def print_profile_summary(self):
        report = self.profile_report()
        if not report:
            return
        slot = report["slot_utilization"]
        print("PROFILE: wrote profile.json")
        print(
            "  Slot utilization: "
            f"{slot['avg_slots_per_bundle']}/{slot['slot_capacity_per_bundle']} "
            f"({slot['avg_utilization_pct']}%)"
        )
        print("  Slot histogram:", slot["histogram"])
        print("  Top ops:", report["instruction_mix"]["by_op"])
        print("  Top PCs:")
        for row in report["pc_hotness"][:5]:
            print(
                f"    pc {row['pc']}: {row['count']}x, "
                f"avg_slots={row['avg_slots']}, {row['instruction']}"
            )
        print("  Memory regions:", report["memory"]["by_region"])
        targeted = report.get("targeted_phases", {})
        forest_floor = targeted.get("forest_load_floor", {})
        if forest_floor:
            print(
                "  Targeted forest load floor: "
                f"{forest_floor['words']} words, "
                f"{forest_floor['cycles_at_two_load_slots']} cycles"
            )
        print("  Select results:", report["selects"])
        hazards = report["bundle_hazards"]
        print(
            "  Bundle hazards: "
            f"{hazards['total']} {hazards['by_kind']} "
            f"(checked {hazards['checks'].get('bundles_checked', 0)} bundles)"
        )

    def write_trace_event(self, event):
        if self.trace_event_count:
            self.trace.write(",\n")
        json.dump(event, self.trace, separators=(",", ":"))
        self.trace_event_count += 1

    def setup_trace(self):
        """
        The simulator generates traces in Chrome's Trace Event Format for
        visualization in Perfetto (or chrome://tracing if you prefer it). See
        the bottom of the file for info about how to use this.

        See the format docs in case you want to add more info to the trace:
        https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        """
        self.trace = open("trace.json", "w", encoding="utf-8")
        self.trace.write("[\n")
        self.trace_event_count = 0
        tid_counter = 0
        self.tids = {}
        self.bundle_tids = {}
        for ci, core in enumerate(self.cores):
            self.write_trace_event(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": ci,
                    "tid": 0,
                    "args": {"name": f"Core {ci}"},
                }
            )
            tid_counter += 1
            self.write_trace_event(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": ci,
                    "tid": tid_counter,
                    "args": {"name": "bundle"},
                }
            )
            self.bundle_tids[ci] = tid_counter
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    tid_counter += 1
                    self.write_trace_event(
                        {
                            "name": "thread_name",
                            "ph": "M",
                            "pid": ci,
                            "tid": tid_counter,
                            "args": {"name": f"{name}-{i}"},
                        }
                    )
                    self.tids[(ci, name, i)] = tid_counter

        # Add zero-length events at the start so all slots show up in Perfetto
        for ci, core in enumerate(self.cores):
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    tid = self.tids[(ci, name, i)]
                    self.write_trace_event(
                        {
                            "name": "init",
                            "cat": "op",
                            "ph": "X",
                            "pid": ci,
                            "tid": tid,
                            "ts": 0,
                            "dur": 0,
                        }
                    )
        for ci, core in enumerate(self.cores):
            self.write_trace_event(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": len(self.cores) + ci,
                    "tid": 0,
                    "args": {"name": f"Core {ci} Scratch"},
                }
            )
            for addr, (name, length) in self.debug_info.scratch_map.items():
                self.write_trace_event(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": len(self.cores) + ci,
                        "tid": BASE_ADDR_TID + addr,
                        "args": {"name": f"{name}-{length}"},
                    }
                )

    def run(self):
        for core in self.cores:
            if core.state == CoreState.PAUSED:
                core.state = CoreState.RUNNING
        while any(c.state == CoreState.RUNNING for c in self.cores):
            has_non_debug = False
            for core in self.cores:
                if core.state != CoreState.RUNNING:
                    continue
                if core.pc >= len(self.program):
                    core.state = CoreState.STOPPED
                    continue
                instr = self.program[core.pc]
                if self.prints:
                    self.print_step(instr, core)
                core.pc += 1
                self.step(instr, core)
                if any(name != "debug" for name in instr.keys()):
                    has_non_debug = True
            if has_non_debug:
                self.cycle += 1

    def alu(self, core, op, dest, a1, a2):
        a1 = core.scratch[a1]
        a2 = core.scratch[a2]
        match op:
            case "+":
                res = a1 + a2
            case "-":
                res = a1 - a2
            case "*":
                res = a1 * a2
            case "//":
                res = a1 // a2
            case "cdiv":
                res = cdiv(a1, a2)
            case "^":
                res = a1 ^ a2
            case "&":
                res = a1 & a2
            case "|":
                res = a1 | a2
            case "<<":
                res = a1 << a2
            case ">>":
                res = a1 >> a2
            case "%":
                res = a1 % a2
            case "<":
                res = int(a1 < a2)
            case "==":
                res = int(a1 == a2)
            case _:
                raise NotImplementedError(f"Unknown alu op {op}")
        res = res % (2**32)
        self.scratch_write[dest] = res

    def valu(self, core, *slot):
        match slot:
            case ("vbroadcast", dest, src):
                for i in range(VLEN):
                    self.scratch_write[dest + i] = core.scratch[src]
            case ("multiply_add", dest, a, b, c):
                for i in range(VLEN):
                    mul = (core.scratch[a + i] * core.scratch[b + i]) % (2**32)
                    self.scratch_write[dest + i] = (mul + core.scratch[c + i]) % (2**32)
            case (op, dest, a1, a2):
                for i in range(VLEN):
                    self.alu(core, op, dest + i, a1 + i, a2 + i)
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")

    def load(self, core, *slot):
        match slot:
            case ("load", dest, addr):
                # print(dest, addr, core.scratch[addr])
                mem_addr = core.scratch[addr]
                self.profile_record_memory("load", "load", mem_addr)
                self.scratch_write[dest] = self.mem[mem_addr]
            case ("load_offset", dest, addr, offset):
                # Handy for treating vector dest and addr as a full block in the mini-compiler if you want
                mem_addr = core.scratch[addr + offset]
                self.profile_record_memory("load", "load_offset", mem_addr)
                self.scratch_write[dest + offset] = self.mem[mem_addr]
            case ("vload", dest, addr):  # addr is a scalar
                addr = core.scratch[addr]
                self.profile_record_memory("load", "vload", addr, VLEN)
                for vi in range(VLEN):
                    self.scratch_write[dest + vi] = self.mem[addr + vi]
            case ("const", dest, val):
                self.scratch_write[dest] = (val) % (2**32)
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")

    def store(self, core, *slot):
        match slot:
            case ("store", addr, src):
                addr = core.scratch[addr]
                self.profile_record_memory("store", "store", addr)
                self.mem_write[addr] = core.scratch[src]
            case ("vstore", addr, src):  # addr is a scalar
                addr = core.scratch[addr]
                self.profile_record_memory("store", "vstore", addr, VLEN)
                for vi in range(VLEN):
                    self.mem_write[addr + vi] = core.scratch[src + vi]
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")

    def flow(self, core, *slot):
        match slot:
            case ("select", dest, cond, a, b):
                res = core.scratch[a] if core.scratch[cond] != 0 else core.scratch[b]
                self.profile_record_select("select", dest, [res])
                self.scratch_write[dest] = res
            case ("add_imm", dest, a, imm):
                self.scratch_write[dest] = (core.scratch[a] + imm) % (2**32)
            case ("vselect", dest, cond, a, b):
                results = []
                for vi in range(VLEN):
                    res = (
                        core.scratch[a + vi]
                        if core.scratch[cond + vi] != 0
                        else core.scratch[b + vi]
                    )
                    results.append(res)
                    self.scratch_write[dest + vi] = res
                self.profile_record_select("vselect", dest, results)
            case ("halt",):
                core.state = CoreState.STOPPED
            case ("pause",):
                if self.enable_pause:
                    core.state = CoreState.PAUSED
            case ("trace_write", val):
                core.trace_buf.append(core.scratch[val])
            case ("cond_jump", cond, addr):
                if core.scratch[cond] != 0:
                    core.pc = addr
            case ("cond_jump_rel", cond, offset):
                if core.scratch[cond] != 0:
                    core.pc += offset
            case ("jump", addr):
                core.pc = addr
            case ("jump_indirect", addr):
                core.pc = core.scratch[addr]
            case ("coreid", dest):
                self.scratch_write[dest] = core.id
            case _:
                raise NotImplementedError(f"Unknown flow op {slot}")

    def trace_post_step(self, instr, core):
        # You can add extra stuff to the trace if you want!
        for addr, (name, length) in self.debug_info.scratch_map.items():
            if any((addr + vi) in self.scratch_write for vi in range(length)):
                val = str(core.scratch[addr : addr + length])
                val = val.replace("[", "").replace("]", "")
                self.write_trace_event(
                    {
                        "name": val,
                        "cat": "op",
                        "ph": "X",
                        "pid": len(self.cores) + core.id,
                        "tid": BASE_ADDR_TID + addr,
                        "ts": self.cycle,
                        "dur": 1,
                    }
                )

    def trace_slot(self, core, slot, name, i, phase="unattributed"):
        self.write_trace_event(
            {
                "name": slot[0],
                "cat": "op",
                "ph": "X",
                "pid": core.id,
                "tid": self.tids[(core.id, name, i)],
                "ts": self.cycle,
                "dur": 1,
                "args": {
                    "slot": str(slot),
                    "named": str(self.rewrite_slot(slot)),
                    "phase": phase,
                },
            }
        )

    def trace_marker(self, core, message):
        self.write_trace_event(
            {
                "name": message,
                "cat": "phase",
                "ph": "i",
                "s": "t",
                "pid": core.id,
                "tid": self.bundle_tids[core.id],
                "ts": self.cycle,
            }
        )

    def trace_bundle(self, core, instr):
        used_by_engine = {
            name: len(instr.get(name, [])) for name in SLOT_LIMITS if name != "debug"
        }
        used_slots = sum(used_by_engine.values())
        slot_capacity = sum(limit for name, limit in SLOT_LIMITS.items() if name != "debug")
        active_engines = [name for name, count in used_by_engine.items() if count]
        pc = core.pc - 1
        self.write_trace_event(
            {
                "name": f"pc {pc}: {used_slots}/{slot_capacity} slots",
                "cat": "bundle",
                "ph": "X",
                "pid": core.id,
                "tid": self.bundle_tids[core.id],
                "ts": self.cycle,
                "dur": 1,
                "args": {
                    "pc": pc,
                    "active_engines": ",".join(active_engines),
                    "used_slots": used_slots,
                    "slot_capacity": slot_capacity,
                    "utilization_pct": round(100 * used_slots / slot_capacity, 2),
                    "phases": ", ".join(
                        self.profile_bundle_phases(core.pc - 1, instr)
                    ),
                    **{f"{name}_slots": count for name, count in used_by_engine.items()},
                },
            }
        )

    def step(self, instr: Instruction, core):
        """
        Execute all the slots in each engine for a single instruction bundle
        """
        ENGINE_FNS = {
            "alu": self.alu,
            "valu": self.valu,
            "load": self.load,
            "store": self.store,
            "flow": self.flow,
        }
        self.scratch_write = {}
        self.mem_write = {}
        pc = core.pc - 1
        self.profile_current_pc = pc
        self.profile_current_scratch_touches = set()
        pending_profile_writes = []
        has_non_debug = any(name != "debug" for name in instr)
        slot_descs = {}
        if has_non_debug and (self.profile is not None or self.enable_hazard_assert):
            slot_descs = self.bundle_slot_descriptors(core, instr)
            self.check_bundle_hazards(core, pc, slot_descs.values())
        if self.profile is not None and has_non_debug:
            self.profile_record_bundle(core, pc, instr)
        if self.trace is not None and has_non_debug:
            self.trace_bundle(core, instr)
        for name, slots in instr.items():
            if name == "debug":
                if not self.enable_debug:
                    continue
                for slot in slots:
                    if slot[0] == "compare":
                        loc, key = slot[1], slot[2]
                        ref = self.value_trace[key]
                        res = core.scratch[loc]
                        assert res == ref, f"{res} != {ref} for {key} at pc={core.pc}"
                    elif slot[0] == "vcompare":
                        loc, keys = slot[1], slot[2]
                        ref = [self.value_trace[key] for key in keys]
                        res = core.scratch[loc : loc + VLEN]
                        assert res == ref, (
                            f"{res} != {ref} for {keys} at pc={core.pc} loc={loc}"
                        )
                    elif slot[0] == "comment":
                        self.profile_record_marker(slot[1])
                        if self.trace is not None:
                            self.trace_marker(core, slot[1])
                continue
            assert len(slots) <= SLOT_LIMITS[name]
            for i, slot in enumerate(slots):
                phase = self.profile_slot_phase(pc, name, i)
                if self.profile is not None:
                    desc = slot_descs[(name, i)]
                    pending_profile_writes.extend(
                        self.profile_record_slot(
                            pc,
                            name,
                            slot,
                            desc["scratch_reads"],
                            desc["scratch_writes"],
                            phase,
                        )
                    )
                self.profile_current_slot_phase = phase
                if self.trace is not None:
                    self.trace_slot(core, slot, name, i, phase)
                ENGINE_FNS[name](core, *slot)
                self.profile_current_slot_phase = "unattributed"
        for addr, val in self.scratch_write.items():
            core.scratch[addr] = val
        for addr, val in self.mem_write.items():
            self.mem[addr] = val
        if has_non_debug:
            self.profile_finish_bundle()
        self.profile_commit_writes(pending_profile_writes)

        if self.trace:
            self.trace_post_step(instr, core)

        del self.scratch_write
        del self.mem_write
        del self.profile_current_pc
        del self.profile_current_scratch_touches

    def close_trace(self):
        if self.profile is not None and not self.profile_written:
            self.write_profile()
        if self.trace is not None:
            self.trace.write("\n]\n")
            self.trace.close()
            self.trace = None

    def __del__(self):
        self.close_trace()


@dataclass
class Tree:
    """
    An implicit perfect balanced binary tree with values on the nodes.
    """

    height: int
    values: list[int]

    @staticmethod
    def generate(height: int):
        n_nodes = 2 ** (height + 1) - 1
        values = [random.randint(0, 2**30 - 1) for _ in range(n_nodes)]
        return Tree(height, values)


@dataclass
class Input:
    """
    A batch of inputs, indices to nodes (starting as 0) and initial input
    values. We then iterate these for a specified number of rounds.
    """

    indices: list[int]
    values: list[int]
    rounds: int

    @staticmethod
    def generate(forest: Tree, batch_size: int, rounds: int):
        indices = [0 for _ in range(batch_size)]
        values = [random.randint(0, 2**30 - 1) for _ in range(batch_size)]
        return Input(indices, values, rounds)


HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]


def myhash(a: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for op1, val1, op2, op3, val3 in HASH_STAGES:
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))

    return a


def reference_kernel(t: Tree, inp: Input):
    """
    Reference implementation of the kernel.

    A parallel tree traversal where at each node we set
    cur_inp_val = myhash(cur_inp_val ^ node_val)
    and then choose the left branch if cur_inp_val is even.
    If we reach the bottom of the tree we wrap around to the top.
    """
    for h in range(inp.rounds):
        for i in range(len(inp.indices)):
            idx = inp.indices[i]
            val = inp.values[i]
            val = myhash(val ^ t.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= len(t.values) else idx
            inp.values[i] = val
            inp.indices[i] = idx


def build_mem_image(t: Tree, inp: Input) -> list[int]:
    """
    Build a flat memory image of the problem.
    """
    header = 7
    extra_room = len(t.values) + len(inp.indices) * 2 + VLEN * 2 + 32
    mem = [0] * (
        header + len(t.values) + len(inp.indices) + len(inp.values) + extra_room
    )
    forest_values_p = header
    inp_indices_p = forest_values_p + len(t.values)
    inp_values_p = inp_indices_p + len(inp.values)
    extra_room = inp_values_p + len(inp.values)

    mem[0] = inp.rounds
    mem[1] = len(t.values)
    mem[2] = len(inp.indices)
    mem[3] = t.height
    mem[4] = forest_values_p
    mem[5] = inp_indices_p
    mem[6] = inp_values_p
    mem[7] = extra_room

    mem[header:inp_indices_p] = t.values
    mem[inp_indices_p:inp_values_p] = inp.indices
    mem[inp_values_p:] = inp.values
    return mem


def myhash_traced(a: int, trace: dict[Any, int], round: int, batch_i: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
        trace[(round, batch_i, "hash_stage", i)] = a

    return a


def reference_kernel2(mem: list[int], trace: dict[Any, int] = {}):
    """
    Reference implementation of the kernel on a flat memory.
    """
    # This is the initial memory layout
    rounds = mem[0]
    n_nodes = mem[1]
    batch_size = mem[2]
    forest_height = mem[3]
    # Offsets into the memory which indices get added to
    forest_values_p = mem[4]
    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    yield mem
    for h in range(rounds):
        for i in range(batch_size):
            idx = mem[inp_indices_p + i]
            trace[(h, i, "idx")] = idx
            val = mem[inp_values_p + i]
            trace[(h, i, "val")] = val
            node_val = mem[forest_values_p + idx]
            trace[(h, i, "node_val")] = node_val
            val = myhash_traced(val ^ node_val, trace, h, i)
            trace[(h, i, "hashed_val")] = val
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            trace[(h, i, "next_idx")] = idx
            idx = 0 if idx >= n_nodes else idx
            trace[(h, i, "wrapped_idx")] = idx
            mem[inp_values_p + i] = val
            mem[inp_indices_p + i] = idx
    # You can add new yields or move this around for debugging
    # as long as it's matched by pause instructions.
    # The submission tests evaluate only on final memory.
    yield mem
