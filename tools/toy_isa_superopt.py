#!/usr/bin/env python3
"""Bounded superoptimization spike for the take-home toy ISA hash stages."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random


MASK32 = 2**32 - 1
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]


def u32(x: int) -> int:
    return x & MASK32


def op2(op: str, a: int, b: int) -> int:
    if op == "+":
        return u32(a + b)
    if op == "^":
        return u32(a ^ b)
    if op == "<<":
        return u32(a << b)
    if op == ">>":
        return u32(a >> b)
    if op == "*":
        return u32(a * b)
    raise ValueError(op)


def stage_eval(stage: tuple[str, int, str, str, int], x: int) -> int:
    op1, val1, op_mid, op3, val3 = stage
    return op2(op_mid, op2(op1, x, val1), op2(op3, x, val3))


@dataclass(frozen=True)
class Expr:
    text: str
    cost: int
    fn: object

    def eval(self, x: int) -> int:
        return self.fn(x)


def check_equiv(expr: Expr, target, samples: list[int], width: int = 32) -> bool:
    mask = (1 << width) - 1
    for x in samples:
        xx = x & mask
        if (expr.eval(xx) & mask) != (target(xx) & mask):
            return False
    return True


def exhaustive_width(expr: Expr, target, width: int) -> bool:
    return check_equiv(expr, target, list(range(1 << width)), width=width)


def candidate_one_stage(stage: tuple[str, int, str, str, int]) -> list[Expr]:
    op1, val1, op_mid, op3, val3 = stage
    candidates: list[Expr] = []
    # Baseline: two independent inputs plus combine. Vector cost is 3, except
    # scalar ALU lane lowering would be costlier; the expression search uses
    # toy op count only.
    candidates.append(
        Expr(
            f"({op1} x {hex(val1)}) {op_mid} ({op3} x {val3})",
            3,
            lambda x, st=stage: stage_eval(st, x),
        )
    )
    # Known legal fusion: (x + c) + (x << s) == x * (2**s + 1) + c.
    if op1 == "+" and op_mid == "+" and op3 == "<<":
        factor = (1 << val3) + 1
        candidates.append(
            Expr(
                f"multiply_add(x, {hex(factor)}, {hex(val1)})",
                1,
                lambda x, factor=factor, val1=val1: u32(x * factor + val1),
            )
        )
    # Generic two-op forms worth checking. These catch simple shift/add/mul
    # identities but intentionally avoid a huge expression grammar.
    constants = [val1, val3, (1 << val3) + 1, u32(val1 ^ val3), u32(val1 + val3)]
    shifts = sorted({val3, 32 - val3 if 0 < val3 < 32 else val3})
    for first_op, second_op in product(["+", "^", "*"], ["+", "^"]):
        for c1 in constants:
            for c2 in constants:
                candidates.append(
                    Expr(
                        f"({first_op} x {hex(c1)}) {second_op} {hex(c2)}",
                        2,
                        lambda x, first_op=first_op, second_op=second_op, c1=c1, c2=c2: op2(
                            second_op, op2(first_op, x, c1), c2
                        ),
                    )
                )
    for shift in shifts:
        for first_op, second_op in product(["+", "^"], ["+", "^"]):
            for c in constants:
                candidates.append(
                    Expr(
                        f"({first_op} x {hex(c)}) {second_op} (x >> {shift})",
                        3,
                        lambda x, first_op=first_op, second_op=second_op, c=c, shift=shift: op2(
                            second_op, op2(first_op, x, c), u32(x >> shift)
                        ),
                    )
                )
                candidates.append(
                    Expr(
                        f"({first_op} x {hex(c)}) {second_op} (x << {shift})",
                        3,
                        lambda x, first_op=first_op, second_op=second_op, c=c, shift=shift: op2(
                            second_op, op2(first_op, x, c), u32(x << shift)
                        ),
                    )
                )
    return candidates


def stage_best_cost(stage: tuple[str, int, str, str, int]) -> int:
    op1, _, op_mid, op3, _ = stage
    if op1 == "+" and op_mid == "+" and op3 == "<<":
        return 1
    return 3


def two_stage_baseline_cost(a: tuple[str, int, str, str, int], b: tuple[str, int, str, str, int]) -> int:
    return stage_best_cost(a) + stage_best_cost(b)


def candidate_two_stage(
    a: tuple[str, int, str, str, int],
    b: tuple[str, int, str, str, int],
) -> list[Expr]:
    candidates: list[Expr] = []
    constants = sorted(
        {
            a[1],
            a[4],
            b[1],
            b[4],
            (1 << a[4]) + 1,
            (1 << b[4]) + 1,
            u32(a[1] + b[1]),
            u32(a[1] ^ b[1]),
        }
    )
    shifts = sorted({s for s in (a[4], b[4], 32 - a[4], 32 - b[4]) if 0 < s < 32})

    # Very small grammar for up to 3 real ops. This is intentionally
    # conservative: if it finds nothing, we know simple two-stage collapsing is
    # not enough, but it won't prove deeper expressions impossible.
    for op_a, op_b in product(["+", "^", "*"], ["+", "^"]):
        for c1 in constants:
            for c2 in constants:
                candidates.append(
                    Expr(
                        f"({op_a} x {hex(c1)}) {op_b} {hex(c2)}",
                        2,
                        lambda x, op_a=op_a, op_b=op_b, c1=c1, c2=c2: op2(
                            op_b, op2(op_a, x, c1), c2
                        ),
                    )
                )
    for shift in shifts:
        for op_a, op_b in product(["+", "^", "*"], ["+", "^"]):
            for c in constants:
                candidates.append(
                    Expr(
                        f"({op_a} x {hex(c)}) {op_b} (x >> {shift})",
                        3,
                        lambda x, op_a=op_a, op_b=op_b, c=c, shift=shift: op2(
                            op_b, op2(op_a, x, c), u32(x >> shift)
                        ),
                    )
                )
                candidates.append(
                    Expr(
                        f"({op_a} x {hex(c)}) {op_b} (x << {shift})",
                        3,
                        lambda x, op_a=op_a, op_b=op_b, c=c, shift=shift: op2(
                            op_b, op2(op_a, x, c), u32(x << shift)
                        ),
                    )
                )
    return candidates


def main() -> None:
    rng = random.Random(0)
    samples = [
        0,
        1,
        2,
        3,
        MASK32,
        MASK32 - 1,
        0x80000000,
        0x7FFFFFFF,
        *[rng.randrange(2**32) for _ in range(4096)],
    ]
    print("one-stage search")
    for i, stage in enumerate(HASH_STAGES):
        target = lambda x, stage=stage: stage_eval(stage, x)
        baseline = min(candidate_one_stage(stage), key=lambda e: e.cost)
        matches = [
            expr
            for expr in candidate_one_stage(stage)
            if expr.cost < 3 and check_equiv(expr, target, samples)
        ]
        verified = []
        for expr in matches:
            if exhaustive_width(expr, target, width=12):
                verified.append(expr)
        best = min(verified, key=lambda e: e.cost, default=None)
        if best:
            print(f"stage {i}: cost {best.cost} {best.text}")
        else:
            print(f"stage {i}: no sampled/exhaustive <=2-op replacement; baseline cost 3")
    print("\ntwo-stage simple-collapse search")
    for i in range(len(HASH_STAGES) - 1):
        first = HASH_STAGES[i]
        second = HASH_STAGES[i + 1]
        target = lambda x, first=first, second=second: stage_eval(second, stage_eval(first, x))
        baseline_cost = two_stage_baseline_cost(first, second)
        matches = [
            expr
            for expr in candidate_two_stage(first, second)
            if expr.cost < baseline_cost and check_equiv(expr, target, samples)
        ]
        verified = []
        for expr in matches:
            if exhaustive_width(expr, target, width=12):
                verified.append(expr)
        best = min(verified, key=lambda e: e.cost, default=None)
        if best:
            print(f"stages {i}-{i+1}: baseline {baseline_cost}, candidate cost {best.cost} {best.text}")
        else:
            print(f"stages {i}-{i+1}: no simple <{baseline_cost}-op collapse found")
    print("\ninterpretation: only stage costs are counted; VLIW engine balance still needs separate measurement.")


if __name__ == "__main__":
    main()
