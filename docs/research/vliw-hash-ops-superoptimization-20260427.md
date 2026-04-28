# Research: Can recent ops/superoptimization work unlock a 1001-cycle VLIW kernel?

**Date:** 2026-04-27
**Asker:** user
**Decision:** ADAPT

## Question

Are there known bit-manipulation, superoptimization, SIMD hashing, or irregular-tree traversal techniques that can reduce this toy ISA's hash/forest operation count enough to make a 1001-cycle VLIW kernel plausible?

## TL;DR

**ADAPT:** use superoptimization as a targeted search method for this toy ISA, but do not expect a published hash peephole to directly cut the current kernel to 1001. The current hash codegen already uses the only obvious ISA-level fusion (`multiply_add` for `(+,+,<<)` stages); the evidence says the next chance is a custom enumerator/search over our exact toy ops plus a forest/index representation change.

## What I read

| Source | Type | Date | What it said (1 line) |
|---|---|---:|---|
| [Minotaur: A SIMD-Oriented Synthesizing Superoptimizer](https://arxiv.org/html/2306.00229v3) | paper | 2024 | SIMD superoptimization finds verified missed compiler optimizations, but works on bounded loop-free cuts and reports modest average speedups. |
| [SuperCoder: Assembly Program Superoptimization with LLMs](https://arxiv.org/html/2505.11480v3) | paper | 2026 | LLM/RL superoptimization can beat `gcc -O3` on assembly, but correctness is test-based and the benchmark is assembly-level, not custom ISA proofs. |
| [Souper README](https://github.com/google/souper) | code-repo | current | Souper extracts LLVM IR bitvector expressions and uses SMT solvers to find shorter equivalent expressions. |
| [STOKE README](https://github.com/StanfordPL/stoke) | code-repo | current | STOKE uses random search over x86-64 transformations; powerful but target-specific and search-heavy. |
| [xxHash README](https://github.com/Cyan4973/xxHash) | code-repo | current | Fast hashes win through vectorized arithmetic, constants, inlining, and memory behavior; auto-vectorization is often not enough. |
| [HighwayHash](https://arxiv.org/abs/1612.06257) | paper/code | 2017 | HighwayHash gets speed from SIMD multiply and permute instructions; those primitives are stronger than this toy ISA's ops. |
| [Automatic Vectorization of Tree Traversals](https://engineering.purdue.edu/~milind/docs/pact13.pdf) | paper | 2013 | Traversal vectorization depends on grouping similar traversals and reducing divergence; it is a layout/scheduling idea, not a hash shortcut. |
| [Fine-Grained Parallel Traversals of Irregular Data Structures](https://www.cs.wm.edu/~bren/files/papers/PACT12.pdf) | paper | 2012 | Irregular traversal speedups come from data layout and parallel traversal organization to improve locality. |
| HN / practitioner summaries on superoptimizers | practitioner-post | mixed | Practitioners consistently frame superoptimizers as useful for hot short bitvector regions, not whole-program magic. |

(Read budget: 9 sources across paper, code-repo, and practitioner-post/source-summary types. Stopped because the answer converged: available techniques point to building our own bounded search, not importing a known hash trick.)

## Findings

1. **Superoptimization is the right tool shape, but it must target this exact ISA.** Minotaur extracts loop-free cuts and synthesizes cheaper replacements, then verifies them; Souper similarly uses SMT over bitvectors. Our hash stages are small fixed-width expressions, so a toy-ISA enumerator/SMT checker is a good fit.

2. **Expected gains from generic superoptimization are bounded unless the operation set changes.** Minotaur reports 7.3% average speedup on GMP and 1.5% on SPEC CPU2017, while our 1001 target needs roughly 20-25% total-cycle reduction and thousands of hash slots removed. That scale likely requires a new expression/layout, not just better scheduling.

3. **Published fast-hash work relies on primitives we do not have.** HighwayHash emphasizes SIMD multiply and permute; xxHash/XXH3 uses vectorized arithmetic and platform-specific vector paths. Our toy ISA has `multiply_add`, no rotate, no permute/shuffle, no carry-less multiply, no lane-crossing op. We already fuse the legal `(+,+,<<)` stages with `multiply_add`.

4. **Tree traversal literature points to grouping/layout, not a direct replacement for gathers.** Traversal vectorization papers win by grouping similar traversals, regularizing layout, and reducing divergence. That supports revisiting forest/index representation, but it does not provide a drop-in way to compute 16+ divergent level-4/5 nodes cheaply.

5. **The strongest actionable idea is a custom bounded superoptimizer.** Enumerate expressions over the toy ops for each hash-stage window and update/index transform, prove equivalence by exhaustive 32-bit sampling plus SMT where feasible, then cost candidates under toy VLIW slot limits. This is more likely to find challenge-specific tricks than importing LLVM/x86 tools.

## Counter-evidence

The strongest counter-case is SuperCoder: recent LLM/RL work claims large assembly-level gains over `gcc -O3` on programs averaging 130 lines. That suggests learned search may find transformations that human heuristics miss. But its correctness is benchmark/test based, the target is real assembly with rich instructions, and the reported gains start from compiler output, not from a hand-shaped toy VLIW DAG whose resource floors we already measured. It argues for search, not for assuming a known paper trick directly applies.

Another counterpoint is that traversal papers show large speedups on irregular data structures when layout is transformed. That keeps the 1001 report credible: a leaderboard solution may have found a representation change that our cache/select model missed. But those papers still require data/layout transformations and do not remove the need to run the hash recurrence.

## Decision: ADAPT

Adapt the research direction into a **toy-ISA superoptimization and representation-search spike**. Do not spend more time looking for a pre-existing hash peephole; the literature does not show one that maps onto our limited ops. The next experiment should synthesize and cost equivalent hash/update subgraphs under exactly our `alu`, `valu`, `load`, and `flow` slots.

## What this means in practice

- **First concrete move:** build a small enumerator for one hash stage/window over `{+, ^, <<, >>, *, multiply_add}` and constants, then test whether it can rediscover our `(+,+,<<)` fusion and find any legal replacements for XOR/shift stages.
- **Watch-fors:** if the enumerator cannot beat the current per-stage op counts, shift effort to forest/index representation; if it finds even one stage cut, scale it to two-stage windows.
- **Out of scope for this research:** cryptographic hash quality, CPU-specific AVX/NEON instruction tuning, modifying the frozen ISA, or relying on leaderboard loopholes.

## Open questions / what I'd read next

1. Whether a two-stage hash window has algebraic equivalences that are invisible at one-stage granularity.
2. Whether an exact bitvector solver can handle 32-bit two-stage windows quickly enough, or whether we need randomized CEGIS.
3. Whether the 1001-class solutions reduced hash work, forest work, or both; our operation census says both floors need cuts.
