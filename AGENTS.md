# AGENTS.md

This file provides guidance to AI agents (Claude Code, Codex etc.) when working with code in this repository.

## What this is

`github.com/cwbudde/algo-approx` — fast, allocation-free scalar math approximations
(`log`, `exp`, trig/arctrig, `tanh`, `logcosh`, `power`),
generic over `float32`/`float64`, with a `Precision` knob. Ported from
`reference_approx.pas` (a read-only Pascal source kept for reference; see the
translation guide at the end of `PLAN.md`). `PLAN.md` is the phase roadmap and
records which decisions were already settled by measurement.

## Commands

`just` is the entry point (`justfile`). The important ones:

```bash
just build              # go build -v ./...
just test               # go test -v -race -count=1 ./...
just test-consumer      # build+vet+test the nested consumerbench module (see below)
just lint               # golangci-lint, root AND consumerbench
just fix                # lint-fix + treefmt
just check              # test + test-consumer + lint + cover  <- run before claiming done
just bench-published    # GOMAXPROCS=1 -benchtime=400ms -count=4 .   (README's method)
just bench-consumer     # same, from consumerbench/ — the numbers that matter
just test-arm64         # cross-arch run under qemu-aarch64-static
```

Single test / package:

```bash
go test -run TestFastTanh_SaturatesExactly ./...
go test ./internal/approx/ -run TestLog
go test ./internal/simd/ -run TestExpAVX2 -v
go test -tags purego ./internal/simd/     # compile the assembly out
go test -run=^$ -fuzz=FuzzFastLog -fuzztime=30s .
```

## Architecture

Three layers, and the boundaries exist for measured reasons:

1. **`approx.go` (root)** — public API only. Every function is a one-line forward
   to `internal/approx` or `internal/simd`, in three flavours per scalar
   operation: generic `FastX[T]`, precision-taking `FastXPrec[T]`, and concrete
   `FastX32`/`FastX64`. `PrecisionAuto` is resolved to `PrecisionBalanced` here
   by `normalizePrecision` (`options.go`). The four **batch** entry points
   (`FastExpBatch32/64`, `FastTanhLogCoshBatch32/64`) follow the same one-line
   rule — no loop bodies in `approx.go`. Their shared contract (lengths,
   aliasing, no `Precision` argument) is documented once in `doc.go` under
   `# Batch functions`; the per-function comments refer to it rather than
   repeating it.
2. **`internal/approx/`** — the algorithms. One file per operation, plus
   `batch.go` for the float64 slice loops.
3. **`internal/simd/`** — float32-native _batch_ (slice) kernels, plus assembly
   for `exp` and for the fused `tanh`/`log(cosh)`: AVX2+FMA on amd64 and NEON on
   arm64. **Reachable from the
   public API** as of the batch entry points above: `FastExpBatch32` forwards to
   `simd.ExpFloat32` and `FastTanhLogCoshBatch32` to `simd.TanhLogCoshFloat32`.
   Two consequences worth knowing before editing here: this package's accuracy
   figures are now public behaviour and live in `ACCURACY.md`, and the root
   module's transitive dependency on `golang.org/x/sys` (via `internal/cpu`) is
   now on the public import path, so `consumerbench/go.sum` carries it too.
   This layer exists because it is where the remaining performance is: the
   scalar kernels are at or below `math`'s cost, while AVX2 `exp` measures
   **11.5–12.3x** over the pure-Go batch loop and **44–49x** over the scalar API,
   and the fused `tanh`/`log(cosh)` measures **10.0–11.8x** and **21–24x**
   respectively (`PLAN.md` §6.0 and §6.0.1 have the tables and the caveats on
   how to read them).
   `dst` and `src` must be **identical or non-overlapping** — an 8-wide
   load/store block is not elementwise, so partial overlap is correct in the Go
   kernel and garbage in the vector one, and two-distinct-slice tests never
   catch it.

Supporting: `internal/cpu` (CPUID feature detection, `sync.Once`-cached,
overridable via `SetForcedFeatures` for tests) and `internal/reference`
(`math`-based references + `MeasureAccuracy`, used by the accuracy tests and by
`ACCURACY.md`).

### SIMD dispatch rules

- **Gate on `HasAVX2 && HasFMA`, never `HasAVX2` alone.** FMA is a separate
  CPUID bit. Every real AVX2 CPU ships FMA3, but hypervisors and emulators can
  mask it, and an FMA opcode then faults with SIGILL. `internal/cpu` was
  vendored before it had `HasFMA`; it does now.
- **Resolve the kernel once, in `init()`.** `DetectFeatures()` takes an
  `RWMutex.RLock` _plus_ a full `Mutex.Lock` on every call. Calling it per batch
  (let alone per element) costs more than a small kernel does, and would make a
  fast kernel look slow at exactly the sizes that matter.
- Keep the **bool-returning contract**: the kernel returns `false` when it
  declines, and the Go side chains to the pure-Go kernel. It costs nothing and
  is the extension point for "this length/alignment isn't mine".
- Tests that call assembly directly bypass dispatch and would SIGILL on a host
  without the features — guard them with a `requireAVX2FMA(tb)` skip.
- **On arm64 the flag is not there to prevent a SIGILL.** Advanced SIMD is
  mandatory in ARMv8-A and every instruction in the NEON kernels is base ASIMD
  with no optional extension behind it, so `expUseNEON` exists to honour
  `ForceGeneric` and nothing else. `requireNEON(tb)` skips rather than fails if a
  host somehow reports no ASIMD, so an emulator produces a skip and not a
  confusing failure.
- `GODEBUG=cpu.avx2=off` exercises the fallback path for free (`x/sys/cpu`
  honours it), as does `-tags purego`.

### Writing the pure-Go batch kernels

Two non-obvious constraints, both discovered by reading the generated code:

- **The kernels do not fit Go's inline budget** (139 and 361 against a budget of
  80). Written as one function per element, the batch loop becomes eight calls
  per block and measures Go's calling convention rather than its arithmetic.
  They are therefore split into small inlinable stages that the loop splices out
  lane by lane. Verify with `-gcflags=-m` that no `CALL` survives in the block body.
- **`if x > HI { x = HI }` is not branchless** — Go emits `UCOMISS`+`JLS`, two
  data-dependent branches per element. The builtin `min`/`max` compile to a
  straight-line `MINSS` sequence _and_ give the NaN propagation the kernel wants.

### Plan 9 operand semantics — verified on this hardware, don't re-derive

Plan 9 reverses Intel's operand order, which makes the three-operand FMA forms
easy to get backwards. These were pinned by an assembled probe, not reasoned out:

| written                           | means                                                      |
| --------------------------------- | ---------------------------------------------------------- |
| `VFMADD213PS Ysrc2, Ysrc1, Ydst`  | `dst = src1*dst + src2`                                    |
| `VFNMADD231PS Ysrc2, Ysrc1, Ydst` | `dst = dst - src1*src2`                                    |
| `VPSUBD Ysrc2, Ysrc1, Ydst`       | `dst = src1 - src2`                                        |
| `VDIVPS Yb, Ya, Ydst`             | `dst = a / b` — the **divisor comes first**                |
| `VSUBPS Yb, Ya, Ydst`             | `dst = a - b`                                              |
| `VANDNPS Yb, Ya, Ydst`            | `dst = ^a & b` — the **first** operand is the negated one  |
| `VBLENDVPS Ymask, Yb, Ya, Ydst`   | picks `b` where mask's sign bit is set, `a` where clear    |
| `VROUNDPS $0x08, Ysrc, Ydst`      | round to nearest, ties to **even**, no precision exception |

Spot-checks that distinguish the right reading from a plausible wrong one:
with `dst=2, src1=3, src2=5`, `VFMADD213PS` gives **11** and `VFNMADD231PS`
gives **-13**; `VROUNDPS $0x08` maps `[2.5 3.5 -2.5 0.5 1.5]` to `[2 4 -2 0 2]`;
`VDIVPS` with `a=10, b=4` gives **2.5** and not 0.4.

### arm64: Go's assembler has almost no vector floating-point arithmetic

This is the single fact that shapes every `*_arm64.s` file here, and it is not
obvious until you try. The complete set of vector mnemonics `cmd/internal/obj/arm64`
accepts is in the header comment of `internal/simd/neon_arm64.h`; the short
version is that `VADD`/`VSUB`/`VUMAX`/`VUMIN` are **integer only**, and
`VFMLA`/`VFMLS` are the only floating-point arithmetic in the list. There is no
vector `FMUL`, `FADD`, `FSUB`, `FDIV`, `FMIN`, `FMAX`, `FABS`, `FCVTZS`, `SCVTF`
or arithmetic right shift. `FMULS` and friends exist but are scalar.

So the kernels reach the hardware through `WORD`, behind named macros in
`neon_arm64.h`. Rules for touching that file:

- **Arguments are register numbers, not names**, ordered `(m, n, d)` so an
  invocation reads in the same Plan 9 order as the native mnemonics beside it.
  Every call site carries a comment naming the registers — `go vet` cannot check
  a `WORD`, and the numbers alone are unreadable.
- **Verify a new encoding by assembling and disassembling it**, never by
  reasoning from the ARM ARM. `go tool objdump` decodes via
  `golang.org/x/arch/arm64`, an implementation independent of the one that
  produced the bytes, so agreement is evidence rather than a tautology. The
  probe loop is: put the candidate in a throwaway `TEXT`, then `go build`,
  `go tool objdump -s`, and read the mnemonic back.
- **`TestNEONWordEncodings` is the standing version of that check.** It builds a
  binary, disassembles the kernel and asserts the expected mnemonics appear. It
  is the only test that can tell "the encoding is right" from "the encoding is
  wrong in a way the test grid does not reach".
- **Constants must live in registers or be loaded.** NEON has no memory
  operands, so the x86 habit of reading a constant straight into an arithmetic
  instruction does not transfer. `FMOVQ off(Rn), Fd` — note `F`, not `V` — is
  the one-instruction immediate-offset 128-bit load, and it costs exactly what
  copying out of a register would, so Horner coefficients are read from a table
  rather than parked in registers.
- **There is no masked load or store.** The 1..3 element tail goes through a
  16-byte scratch buffer in the frame. Do **not** replace it with a final
  full-width vector starting at `n-4`: that is correct for disjoint slices and
  silently computes `f(f(x))` for up to three elements when `dst == src`, which
  no two-distinct-slice test can see.

arm64 Plan 9 semantics pinned by the same probe method:

| written                       | means                                                     |
| ----------------------------- | --------------------------------------------------------- |
| `VFMLA Vm.S4, Vn.S4, Vd.S4`   | `dst = dst + n*m` (accumulates into the dest)             |
| `VFMLS Vm.S4, Vn.S4, Vd.S4`   | `dst = dst - n*m`                                         |
| `VSUB Vm.S4, Vn.S4, Vd.S4`    | `dst = n - m`, **integer**                                |
| `VBIT Vm.B16, Vn.B16, Vd.B16` | takes `n` where `m`'s bits are set, keeps `dst` elsewhere |
| `FMOVQ off(Rn), Fd`           | 128-bit load; the destination is spelled `F`, not `V`     |

Unlike x86 there is no `VMINPS`/`VMAXPS` NaN-asymmetry trap: AArch64's `FMIN`
and `FMAX` are symmetric, so a swapped operand pair is merely wrong about which
number it returns. They do return the **default** quiet NaN rather than the
operand, so tests assert NaN-ness, not a payload.

### arm64 contracts FMAs and amd64 does not — this changes what a differential test means

Go's arm64 backend fuses `x*y + z` into a single `FMADD`; the amd64 backend
never does, because the amd64 baseline has no FMA. The pure-Go kernels are
therefore **not the same computation on the two architectures**, and each
assembly kernel has to match its own host's Go kernel, not the other assembly
kernel.

Concretely: `logCoshLarge32` ends with `a - ln2f + 2*w*(...)`. The AVX2 kernel
computes that as a separate multiply and add, which is right for amd64. The NEON
kernel must use one `VFMLA`, because that is what the arm64 compiler emits. The
first draft of `hyperbolic32_arm64.s` transliterated the AVX2 sequence and
disagreed with the Go kernel by 1 ulp of `log(cosh)` at `x = 0.81` — one element
in twenty-five, caught by `TestBlockAndTailAgree`, invisible in `tanh`.

**So "transliterate the AVX2 kernel" is the right instruction only down to the
point where contraction differs.** When a differential test fails by exactly one
ulp on one branch, suspect this before suspecting the encoding.

The payoff is that on arm64 the two kernels agree far more closely than on
amd64. Swept over all 2³² bit patterns on both architectures:

| kernel      | amd64 drift | amd64 identical | arm64 drift |   arm64 identical |
| ----------- | ----------: | --------------: | ----------: | ----------------: |
| `exp`       |       1 ulp |          99.95% |       1 ulp | all but 22 inputs |
| `tanh`      |       2 ulp |          99.98% |       1 ulp |  all but 2 inputs |
| `log(cosh)` |       4 ulp |          99.99% |       0 ulp |   **all of them** |

Both full sweeps are opt-in (`ALGO_APPROX_EXHAUSTIVE=1`). On a laptop, run them
under `caffeinate -dimsu`: the fused sweep takes 10 minutes of solid CPU, and on
a machine that idle-sleeps it will otherwise take hours of wall time and look
hung.
If a new kernel disagrees with its Go twin by a wild margin, re-run these before
suspecting the maths.

**The first Plan 9 operand may be a memory reference**, and for the FMA forms
this is load-bearing rather than a convenience: `VFMADD213PS someConst<>(SB),
Y1, Y0` means `Y0 = Y1*Y0 + const`. A polynomial evaluated this way needs no
register for its coefficients at all, which is the only reason the fused
tanh/log(cosh) kernel fits — it has roughly 35 constants and 16 registers.

**`VMINPS`/`VMAXPS` are the exception and must keep their constants in
registers.** They return SRC2 when either input is NaN, and SRC2 is the same
operand slot a memory reference would have to occupy. Feed the clamp constant
from memory and every NaN silently comes back as the clamp value, which no
accuracy test over a finite grid will ever see.

`go vet`'s `asmdecl` pass checks FP offsets and frame sizes — it requires the
`dst_base+0(FP)` / `dst_len+8(FP)` spelling for slice arguments. (algo-fft's
`dst+0(FP)` style is what asmdecl flags; this repo uses the vet-clean form.)

### Register budget across a shared macro

`EXPBODY` lives in `exp32_amd64.h` and is included by both `exp32_amd64.s` and
`hyperbolic32_amd64.s`, so the shared exponential has exactly one definition and
the two kernels cannot drift into computing subtly different results. `<>`
symbols are file-scoped, so an included header gives each `.s` its own private
copy of the RODATA rather than a duplicate-symbol error.

**A macro's register usage is part of its contract.** `EXPBODY` originally
broadcast ten constants into `Y6..Y15`. The fused kernel's four branch bodies
then overwrote those registers, so every block after the first computed its
exponential from whatever the previous block had left behind. The result: the
first eight elements correct, everything after them NaN. A spot check at a
handful of _x_ values would not have found it — the existing sweep tests did.

The fix was to shrink the macro rather than reload constants per iteration:
`EXPBODY` now holds only its two clamp constants in registers and reads the rest
from RODATA as memory operands. It occupies `Y0..Y4` plus `Y6..Y7`, and
documents that `Y5` and `Y8..Y15` belong to the caller. If you add a third
kernel, respect that split or widen it deliberately — and re-run the exhaustive
differential for **both** existing kernels afterwards, since they share the body.

### The kernel + shim invariant — do not break this

Most algorithms are **a non-generic `float64` kernel plus a tiny generic shim**:

```go
func Log[T Float](x T, prec Precision) T { return T(log64(float64(x), prec)) }
```

Go compiles a generic body once per gcshape and calls it through a runtime
dictionary, which the compiler will not inline across a package boundary. With
the arithmetic inside the generic body, every consumer paid a call frame it
could not remove — the library won its own benchmarks and lost every caller's.
Keeping the body non-generic makes the shim small enough to inline into an
external caller.

- Applies to `Log`, `Exp`, `Tanh`, `LogCosh`.
- The trig family is genuinely generic and was intentionally left alone.
- **The property is gated statically**, not by timings:
  `consumerbench/inline_test.go:TestCrossModuleInlining` parses
  `go build -gcflags=-m` output and fails if the shims stop inlining. If you
  move arithmetic into a generic body, that test tells you.

**Corollary: never write scalar assembly here.** A Plan 9 ABI0 function can never
be inlined, so it is permanently outside the invariant above. For a ~5 ns
operation the call frame alone eats the win — measured, the scalar kernels are
already at or below `math`'s cost (`FastExp64` 0.97x, `FastTanh64` 0.88x), so
there is nothing for scalar asm to win back. Assembly pays **only** on batch
(slice) kernels, where one call amortises over thousands of elements. See
`PLAN.md` §6.0.

### Precision must not be a runtime argument in a loop

`FastExpPrec_{Fast,Balanced,High}` measure 12.2 / 14.7 / 16.8 ns against 5.6 ns
for `FastExp64`. Passing `prec` as a runtime value defeats constant-folding of
the `switch` in `expPoly` — **the dispatch costs more than the polynomial it
selects.** Any batch API must resolve precision once per call and run a
monomorphic inner loop. The library used to carry a `selectImpl` helper that
returned a `func(T) T` per call; it went out with `sqrt`/`invsqrt`/`recip`, and
it is worth not reinventing — an indirect call per element would defeat the
whole point of a batch API.

### `consumerbench/` is a separate Go module

It imports algo-approx by module path (with a `replace ../`) so nothing gets
same-package inlining. Consequence: **`go build ./...`, `go test ./...` and
`golangci-lint run` at the root never descend into it.** Anything touching the
public API surface or the shim structure must also be run through
`just test-consumer`. `consumerbench/callsites.go` holds the call sites the
inlining test inspects — keep a call site there for any new public entry point
that follows the kernel+shim pattern.

**The batch entry points have call sites there but are deliberately absent from
`inline_test.go`'s `[]string{"Log","Exp","Tanh","LogCosh"}` list.** A batch body
is a loop over a whole slice; it is far past the inline budget and will never
inline, and it does not need to — one call amortises over thousands of elements,
which is the entire premise. `TestCrossModuleInlining` must stay green on
exactly those four operations. Adding a batch function to that list converts the
repo's core regression gate into a permanently failing one, which is strictly
worse than having no gate.

## Invariants the tests pin (don't "optimize" these away)

- **Zero allocations** for every public function, float32 and float64,
  scalar and batch — `approx_alloc_test.go`. The batch cases hoist their buffers
  outside the measured closure; allocating inside it would measure `make`.
- **The float64 batch loops are bit-identical to their scalar twins**, and
  `approx_batch_test.go` asserts that with no tolerance at all. They call the
  same kernels, so any difference is a copied-and-drifted branch point rather
  than a rounding artefact, and a tolerance would hide exactly that. The fused
  `tanhLogCosh64` reads the same `tanhBranch` and the same single
  `expNeg2` as `tanh64`/`logCosh64` for this reason. The float32 batch path is a
  different algorithm and is pinned to a derived ulp bound instead — the _sum_
  of the two paths' accuracy bounds, since they are asymmetric (38 and 1 for
  `exp`). See `ACCURACY.md`.
- **`FastTanh`/`FastLogCosh` are a consistent pair.** `tanh` is exactly
  `d/dx log(cosh x)`; both are built from one shared `u = exp(-2|x|)` and share
  one branch point (`0.625`, identical in scalar and SIMD paths) so the identity
  survives approximation. A downstream discrete-gradient energy scheme depends
  on it. Also: `FastTanh`'s odd symmetry is bit-exact and its saturation at
  `|x| >= 19.0625` is exact; `FastLogCosh` never forms `cosh` (no overflow at
  `|x| > 710`).
  **`19.0625` is a float64 constant.** In float32, `tanh` rounds to exactly
  `1.0f` from `|x| >= 9.010914` — carrying 19.0625 into a float32 kernel buys a
  10-wide band of inputs that can only ever produce `1.0f`. The float32 batch
  kernel needs no saturation constant at all: because both branches are always
  evaluated, `1 - 2u/(1+u)` saturates to exactly `1.0f` on its own at that
  crossover. A test pins the crossover so it cannot drift.
- **float32 through the scalar API is only ~5.5 decimal digits, not 7.** Measured
  in ulps at `PrecisionBalanced`: `FastExp32` **38**, `FastLog32` **103**,
  `FastTanh32` 1, `FastLogCosh32` 0. The widening to float64 is not the problem —
  the _kernel's own_ truncation is: Balanced `expPoly` is a degree-5 Taylor whose
  error is `|r|⁶/720 = 2.4e-6` at `|r| <= ln2/2`, i.e. ~20 float32 ulp before any
  rounding. The float32-native **minimax** kernels in `internal/simd` measure
  **1 ulp** — roughly 28x better _and_ faster, because Taylor is simply the wrong
  basis. Going minimax at equal degree buys ~3 ulp for free; degrees 6 and 7 are
  indistinguishable in float32 because evaluation rounding, not truncation,
  dominates past degree 6.
- **Subnormals.** `log64` rescales subnormal inputs before reading the exponent
  field; without it the result is off by up to ~36 nats (Go's own `math.Log` has
  this bug on amd64).
- **SIMD ↔ Go kernel agreement.** The AVX2 `exp` is verified against the pure-Go
  kernel over all 2³² float32 bit patterns: max 1 ulp, 99.95% bit-identical,
  differing only where the assembly contracts FMAs. `decl_text_test.go` guards
  against a body-less Go declaration losing its `TEXT` symbol. Read the header
  comment in `exp32_amd64.h` before editing it — operand order in
  `VMINPS`/`VMAXPS`, the two-step `2^k` reconstruction and VEX purity +
  `VZEROUPPER` are each load-bearing and each fail silently.
- **Agreement bounds are twice the accuracy bound, and that is a derivation
  rather than slack.** If each kernel is independently within _k_ ulp of the
  true value, the two can straddle it and be _2k_ apart while both are correct.
  So the fused kernel's differential test allows 2 ulp on `tanh` and 8 on
  `log(cosh)`. Before widening any such bound, measure which kernel is closer to
  a float64 reference — for both outputs here the **assembly is fractionally
  closer than the pure-Go kernel**, so the drift is two implementations rounding
  to different sides of a genuine cancellation, not the vector kernel losing
  precision. A tolerance raised without that check is just a hidden bug.
- **Assert that dispatch actually chose the assembly.** Every differential test
  calls the AVX2 kernel directly, so the whole suite stays green even if the
  exported wrapper silently runs the Go path on every call. The tell is an
  "optimisation" that measures no change at all, which reads as _the kernel
  wasn't worth it_ rather than _the kernel never ran_.
  `TestDispatchSelectsAVX2OnCapableHost` pins it.

## Benchmarking on this hardware is a trap — read before running one

The dev box is a **12th Gen i7-1255U**: P-cores on CPUs 0–3 (4.7 GHz), E-cores on
4–11 (3.5 GHz, 400 MHz at idle). An unpinned `go test -bench` migrates between
them mid-run. Observed on this repo, unpinned: the _same_ `FastLog` benchmark
reported **364 ns/op and 8 ns/op**, and `BenchmarkOverhead` — a single array
index — reported **9.886 ns/op**. Those swings are the same size as, or larger
than, most effects worth measuring. **This is not irreducible noise; it is
fixable, and until it is fixed no number means anything.**

```bash
uptime                                   # 1. ABORT if load average is not near-idle;
                                         #    this box is shared and other users' QEMU
                                         #    suites have parked it at load 50.
go test -c -o /tmp/x.test ./internal/simd/
for i in $(seq 10); do                   # 2. separate invocations, NOT -count=10
  taskset -c 2 /tmp/x.test -test.run='^$' -test.bench=. \
      -test.benchtime=200ms -test.count=1 >> /tmp/b.txt
done
benchstat /tmp/b.txt                     # 3. and check the CV it prints
```

- **Pin to a P-core.** Pinning collapsed variance from >2x to ~±10%.
- **`-count=10` in one invocation aliases thermal drift with arm ordering** — it
  runs arm A ten times, then B, then C, so later arms run on a hotter chip. Use
  separate `-count=1` runs; benchstat handles interleaved samples.
- **Discard the first sample after a pin change** (frequency ramp — reliably high).
- **Report ns/_element_.** With `b.Run("n=…")` sub-benchmarks `ns/op` is per batch
  call, so N=64 and N=1M are not comparable. Use `b.SetBytes(n*4)`.
- **Pre-touch destination buffers** outside the timed region, or the first
  iterations measure page faults (milliseconds at N=1M).
- Quirk: `taskset -c 0 go test …` fails here with
  `CPU-Liste konnte nicht eingelesen werden: 0--1` (a wrapper artifact around the
  `go` command); `taskset -c 0,1 go …` works, as does taskset on a compiled
  binary. A _silent_ pinning failure invalidates the whole session — check it runs.
- AVX2 frequency licensing is a **non-issue** on Alder Lake client (that was
  Skylake-SP server), but these numbers will not transfer to a Xeon.
- `just bench-published` / `just bench-consumer` do **not** pin. They are the
  documented method for the README tables; add pinning before trusting a delta.

## Documentation is measured, not asserted

`README.md` and `ACCURACY.md` contain real numbers with dates, a stated CPU and
a stated method (harness overhead subtracted, minimum over ≥24 samples,
`GOMAXPROCS=1`). Several sections exist specifically to record that an
optimization was tried and _lost_. The strongest example is now a removal:
`FastSqrt`, `FastInvSqrt` and `FastRecip` measured 6.9x / 4.3x / 9x **slower**
than the hardware instructions they replaced and were deleted outright — see
`docs/removed-kernels.md`, which keeps the numerical analysis that was worth
more than the code. The README's old justification for them ("for targets
without a hardware square root") turned out to be false: Go intrinsifies
`math.sqrt` on every GOARCH it supports. Check a claim like that against the
toolchain before building on it. If you change a kernel's cost or
accuracy, re-measure with `just bench-published` + `just bench-consumer` and
update the tables with the new date and CPU — do not hand-edit a number, and do
not add a performance claim that isn't backed by a run.

## Style

- `gofumpt` + `goimports`/`gci`, run via `just fmt` (treefmt).
- golangci-lint runs with `default = 'all'`; see `.golangci.toml` for the
  disabled set. Prefer a targeted `//nolint:linter // reason` over broadening the
  config.
- Comments here carry the _why_ — a rejected alternative, a measurement, a
  failure mode. Match that density in the numeric code; it is the repository's
  main defence against a plausible-looking edit that quietly costs ulps.

## Releasing, and Not Drifting

This module is part of the `github.com/cwbudde/algo-*` family, which is co-developed
across separate repositories. That arrangement failed once already, and the rules below
exist to stop it failing the same way twice.

**What went wrong (August 2026).** The family had drifted onto three different `algo-fft`
versions simultaneously — `algo-pde` on v0.6.15, `algo-dsp` on v0.7.3, `algo-acoustics` on
v0.6.11 — while `algo-fft`'s own `main` sat 97 commits past its latest tag and its
CHANGELOG documented a `v0.7.5` that had never been tagged. Because `algo-fft`'s generic
`PlanReal2D`/`PlanReal3D` had changed signature between the v0.6 and v0.7 lines, _no single
upgrade anywhere would compile_. Untangling it took a day and four coordinated releases.

Three separate mistakes combined to produce that. Each now has a check.

### 1. Do not let work pile up untagged

Work that only exists on `main` cannot be consumed. If you finish something a sibling repo
needs, tag it — do not wait for a milestone.

```bash
just check-unreleased     # how much is sitting past the latest tag?
```

A scheduled CI job (`.github/workflows/dep-drift.yml`) reports this weekly.

### 2. Do not sit on stale siblings

```bash
just check-deps           # are all github.com/cwbudde/* deps at their latest tags?
```

This is wired into the repo's aggregate check recipe, and the same scheduled job files a
GitHub issue when it starts failing. If a bump is _deliberately_ deferred, write down why in
`PLAN.md` — an undocumented old pin is indistinguishable from a forgotten one.

Renovate (`.github/renovate.json`) opens the bump PRs automatically and groups the whole
`cwbudde` family into a single PR on purpose: an incompatible `algo-fft` can reach a
consumer through two different dependency paths at once, so bumping them one PR at a time
produces intermediate combinations that never build.

### 3. Never remove or change exported API without the version saying so

Always release through the guard rather than by hand:

```bash
just tag-release v0.8.0       # runs every precondition, then tags and pushes
```

It refuses to tag when the tree is dirty, when `HEAD` is not a pushed default branch, when the tag
already exists or does not sort after the current one, when siblings are stale, when
`CHANGELOG.md` has no section for the version, or when the exported API changed
incompatibly without the version reflecting it.

**That last rule is stricter than semver, deliberately.** Semver exempts `v0.x` — "anything
MAY change at any time" — so `gorelease` will happily approve a _patch_ bump across a
removed symbol. Every module in this family is `v0.x`, so that exemption is exactly the hole
we fell through: `KernelEightStep` was removed and `PlanReal2D` became generic, and nothing
in the version numbers said so. The guard therefore requires a **minor** bump for any
incompatible change while on `v0.x`.

When you do break API, say so in the CHANGELOG in the form a consumer needs: the old
signature, the new signature, and the call-site rewrite. "Refactored plans" does not help
anyone; `NewPlanReal2D(rows, cols)` → `NewPlanReal2D64(rows, cols)` does.

### Order of operations for a cross-repo change

Releases must flow up the dependency graph, never sideways:

```
algo-vecmath ─┐
algo-approx  ─┼─→ algo-dsp ─┐
algo-fft ─────┴─→ algo-pde ─┴─→ algo-acoustics
```

Tag the dependency first, then bump and tag its consumers, then the consumers' consumers.
Bumping a consumer before its dependency is tagged is what forces pseudo-versions into
`go.mod`, and those are how a repo quietly ends up pinned to a commit nobody can find later.
