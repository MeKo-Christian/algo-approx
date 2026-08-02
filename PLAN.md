# algo-approx Implementation Roadmap

## Overview

**Goal**: Create a high-performance Go library for fast mathematical approximations, porting Pascal code from `../go-fft/reference/approx.pas` (~300+ functions, 48K tokens).

**Target**: ~60-80 generic Go functions covering all categories from the Pascal source.

**Total Estimated Size**: ~7100 LOC across 8 phases

## Architecture Decisions

- ✅ **Separate repository** (not monorepo with algo-fft)
- ✅ **Simple function-based API** (not Plan-based - approximations are stateless)
- ✅ **Both float32 and float64** using Go generics (`Float` constraint)
- ✅ **Precision enum** for accuracy/speed tradeoffs (Fast/Balanced/High)
- ✅ **Zero-allocation design** - pure functions, no heap allocations
- ✅ **Comprehensive testing** - reference validation, property tests, benchmarks
- ✅ **Optional SIMD** - AVX2 and NEON for batched operations (Phase 6+)

## Repository Structure

```plain
algo-approx/
├── approx.go              # Public API: FastSqrt, FastLog, FastExp, etc.
├── types.go               # Float constraint, Precision enum
├── errors.go              # Domain error types (minimal)
├── doc.go                 # Package documentation with examples
├── README.md              # User guide, quick start, benchmarks
├── ACCURACY.md            # Detailed accuracy measurements
├── CONTRIBUTING.md        # Contribution guidelines
├── PLAN.md                # This file
├── CLAUDE.md              # Points to AGENTS.md
├── AGENTS.md              # Development guidelines for AI assistants
├── LICENSE                # MIT License
├── justfile               # Build automation (from algo-fft)
├── treefmt.toml           # Code formatting config
├── .golangci.toml         # Linter configuration
├── .gitignore
├── go.mod
├── go.sum
│
├── internal/
│   ├── approx/            # Core algorithm implementations
│   │   ├── sqrt.go        # Square root (Babylonian method)
│   │   ├── invsqrt.go     # Inverse square root (Quake method)
│   │   ├── power.go       # Power, root, integer power
│   │   ├── log.go         # Logarithms (ln, log2, log10)
│   │   ├── exp.go         # Exponentials (e^x, 2^x, 10^x)
│   │   ├── trig.go        # Trigonometric (sin, cos, sec, csc)
│   │   ├── tan.go         # Tangent and cotangent
│   │   ├── arctrig.go     # Inverse trig (arctan, arcsin, arccos)
│   │   ├── util.go        # Shared utilities and helpers
│   │   ├── constants.go   # Mathematical constants for each precision
│   │   └── dispatch.go    # Generic type dispatch for float32/float64
│   │
│   ├── cpu/               # CPU feature detection (vendored from algo-fft)
│   │   ├── cpu.go         # Main CPU detection interface
│   │   ├── detect_amd64.go
│   │   ├── detect_arm64.go
│   │   └── detect_generic.go
│   │
│   ├── reference/         # High-precision reference implementations
│   │   ├── sqrt.go        # math.Sqrt wrappers
│   │   ├── log.go         # math.Log wrappers
│   │   ├── exp.go         # math.Exp wrappers
│   │   ├── trig.go        # math.Sin/Cos/Tan wrappers
│   │   ├── power.go       # math.Pow wrappers
│   │   └── accuracy.go    # Accuracy measurement utilities
│   │
│   └── simd/              # SIMD optimizations (Phase 6+)
│       ├── sqrt_amd64.go  # AVX2 vectorized sqrt
│       ├── sqrt_amd64.s   # AVX2 assembly implementation
│       ├── sqrt_arm64.go  # NEON vectorized sqrt
│       └── sqrt_arm64.s   # NEON assembly implementation
│
└── .github/
    └── workflows/
        ├── test.yaml      # Main CI workflow
        ├── test-unit.yaml # Cross-platform testing
        └── test-lint.yaml # Linting checks
```

> **This tree, and the Phase 1 sections below, are the original plan and are kept
> as a record of it — they are no longer an accurate description of the
> repository.** Two things have changed since:
>
> - `sqrt`, `invsqrt` and `recip` were **removed from the library** (§6.2). They
>   were measured 6.9x / 4.3x / 9x slower than the hardware instructions they
>   replace, and Go intrinsifies `math.sqrt` on every GOARCH it supports, so
>   there is no target where they win. `internal/approx/{sqrt,invsqrt,recip}.go`,
>   `internal/reference/sqrt.go` and `internal/approx/dispatch.go` are gone;
>   `docs/removed-kernels.md` keeps the numerical analysis. §1.6.1 and §1.6.2
>   below describe deleted code.
> - `internal/simd/` holds float32-native batch kernels for `exp` and the fused
>   `tanh`/`log(cosh)`, not the `sqrt` kernels sketched here (§6.0, §6.0.1).

---

# Phase 1: Foundation & Core Math (MVP)

**Goal**: Ship a working, useful library quickly with essential approximation functions.

**Estimated LOC**: ~2000

## 1.1 Project Scaffolding

**Tasks**:

- [x] Create repository directory: `mkdir -p ../algo-approx`
- [x] Initialize Go module: `go mod init github.com/cwbudde/algo-approx`
- [x] Set Go version: `go mod edit -go=1.25.0`
- [x] Copy build infrastructure from ../go-fft:
  - [x] `justfile` (adapt targets for approximations)
  - [x] `treefmt.toml` (formatting config)
  - [x] `.golangci.toml` (linter config)
  - [x] `.gitignore`
- [x] Create directory structure:
  ```bash
  mkdir -p internal/approx internal/cpu internal/reference
  ```
- [ ] Initialize git repository and create initial commit (workspace currently has no `.git/`)

**Files Created**: `go.mod`, `justfile`, `treefmt.toml`, `.golangci.toml`, `.gitignore`

## 1.2 Type System

**File**: `types.go`

**Tasks**:

- [x] Define `Float` constraint:
  ```go
  type Float interface {
      ~float32 | ~float64
  }
  ```
- [x] Add comprehensive documentation explaining the constraint

**File**: `options.go`

**Tasks**:

- [x] Define `Precision` enum:

  ```go
  type Precision int

  const (
      PrecisionAuto Precision = iota
      PrecisionFast      // 2-3 terms, ~2-3 decimal digits
      PrecisionBalanced  // 4-5 terms, ~4-5 decimal digits (default)
      PrecisionHigh      // 6-7 terms, ~6-7 decimal digits
  )
  ```

- [x] Implement `String() string` method for Precision
- [x] Add validation function: `IsValid() bool`
- [ ] Document accuracy guarantees for each level

**Files Created**: `types.go`, `options.go`

## 1.3 Error Handling

**File**: `errors.go`

**Tasks**:

- [x] Define domain error types:
  ```go
  var (
      ErrDomainError  = errors.New("input outside valid domain")
      ErrNaN          = errors.New("result is not a number")
      ErrInfinity     = errors.New("result is infinite")
  )
  ```
- [x] Add helper functions for error checking
- [x] Document when each error is returned

**Files Created**: `errors.go`

## 1.4 CPU Detection (Vendored)

**Directory**: `internal/cpu/`

**Tasks**:

- [x] Copy entire `/internal/cpu/` directory from algo-fft:
  ```bash
  cp -r ../go-fft/internal/cpu/ internal/cpu/
  ```
- [x] Update import paths in copied files:
  - Change `github.com/cwbudde/algo` → `github.com/cwbudde/algo-approx`
- [x] Verify CPU detection compiles:
  ```bash
  go build ./internal/cpu/
  ```
- [x] Add tests for CPU detection:
  ```bash
  go test ./internal/cpu/
  ```

**Files Created**: `internal/cpu/*.go`

## 1.5 Reference Implementation Framework

**File**: `internal/reference/accuracy.go`

**Tasks**:

- [x] Define `AccuracyMetrics` struct:
  ```go
  type AccuracyMetrics struct {
      MaxAbsError   float64
      MaxRelError   float64
      MeanAbsError  float64
      RMSError      float64
      DecimalDigits float64  // -log10(maxRelError)
  }
  ```
- [x] Implement `MeasureAccuracy[T Float]()` function
- [x] Add statistical helpers (mean, rms, max)
- [x] Document measurement methodology

**File**: `internal/reference/sqrt.go`

**Tasks**:

- [x] Implement reference sqrt wrapper:
  ```go
  func Sqrt[T Float](x T) T {
      return T(math.Sqrt(float64(x)))
  }
  ```
- [x] Add reference for inverse sqrt
- [x] Add tests comparing to `math.Sqrt`

**Files Created**: `internal/reference/accuracy.go`, `internal/reference/sqrt.go`

## 1.6 Core Implementations (Pure Go)

### 1.6.1 Square Root

**File**: `internal/approx/sqrt.go`

**Tasks**:

- [x] Implement Babylonian method with precision variants (Fast/Balanced/High)
- [x] Implement initial guess using bit manipulation (Quake-style)
- [x] Add generic wrapper with precision dispatch: `Sqrt[T Float](x T, prec Precision) T`
- [x] Handle edge cases (negative, zero, infinity, NaN)
- [x] Add tests:
  - [x] Unit tests with known values
  - [x] Reference comparison tests
  - [x] Edge case tests
  - [ ] Property tests (monotonicity, etc.)
- [ ] Benchmark against `math.Sqrt`

**Pascal Source Reference**: Lines 25-36 in `approx.pas` (`FastSqrtBab0/1/2`)

### 1.6.2 Inverse Square Root

**File**: `internal/approx/invsqrt.go`

**Tasks**:

- [x] Implement Quake-style seed for inverse square root (float32 + float64 magic constants)
- [x] Add Newton-Raphson refinement iterations
- [x] Implement precision variants (1-3 refinement steps)
- [x] Add generic wrapper
- [x] Handle edge cases
- [x] Add tests
- [ ] Add benchmarks

**Pascal Source Reference**: Lines 17-20 in `approx.pas` (`FastInvSqrt`)

### 1.6.3 Logarithms

**File**: `internal/approx/log.go`

**Tasks**:

- [x] Implement ln(x) with range reduction (`math.Frexp`) and odd-power series in $y=(m-1)/(m+1)$
- [ ] Implement MinError variants (polynomial approximation)
- [ ] Implement ContinuousError variants
- [ ] Add log2 and log10 variants (scaling from ln)
- [x] Range reduction for better accuracy
- [x] Generic wrappers with precision dispatch
- [x] Tests
- [ ] Benchmarks

**Pascal Source Reference**:

- Lines 42, 390-412 (`FastLog2*`)
- Lines 458-481 (`FastLog10*`)
- Lines 486-500 (`FastLn*`)

### 1.6.4 Exponentials

**File**: `internal/approx/exp.go`

**Tasks**:

- [x] Implement exp(x) using range reduction to $k\,\ln 2 + r$ and truncated Taylor polynomial for $e^r$
- [ ] Implement MinError variants
- [ ] Implement ContinuousError variants
- [ ] Add exp2 (2^x) and exp10 (10^x) variants
- [x] Range reduction for large inputs
- [x] Generic wrappers
- [x] Tests
- [ ] Benchmarks

**Pascal Source Reference**:

- Lines 370-384 (`FastPower2*`)
- Lines 418-432 (`FastPower10*`)
- Lines 438-452 (`FastExp*`)

### 1.6.5 Generic Dispatch

**File**: `internal/approx/dispatch.go`

**Tasks**:

- [x] Implement generic type dispatcher:
  ```go
  func selectImpl[T Float](fast, balanced, high func(T) T, prec Precision) func(T) T
  ```
- [ ] Add CPU feature-based selection (for future SIMD)
- [ ] Document dispatch strategy
- [ ] Add tests for dispatch logic

**Files Created**: `internal/approx/*.go`

## 1.7 Public API

**File**: `approx.go`

**Tasks**:

- [x] Implement public generic functions:
  ```go
  func FastSqrt[T Float](x T) T
  func FastSqrtPrec[T Float](x T, prec Precision) T
  func FastInvSqrt[T Float](x T) T
  func FastLog[T Float](x T) T
  func FastExp[T Float](x T) T
  ```
- [x] Add type-specific convenience aliases:
  ```go
  func FastSqrt32(x float32) float32
  func FastSqrt64(x float64) float64
  // ... etc
  ```
- [ ] Add comprehensive GoDoc comments with:
  - Function description
  - Accuracy guarantees (decimal digits)
  - Valid input ranges
  - Example usage
- [ ] Handle all error cases properly

**File**: `doc.go`

**Tasks**:

- [x] Write package-level documentation
- [ ] Add usage examples for all precision levels
- [ ] Document accuracy guarantees
- [ ] Add performance characteristics
- [ ] Include code examples:

  ```go
  // Example: Basic usage
  x := approx.FastSqrt(16.0)  // ≈ 4.0

  // Example: Precision control
  y := approx.FastSqrtPrec(16.0, approx.PrecisionHigh)

  // Example: Type-specific
  z := approx.FastSqrt32(float32(16.0))
  ```

**Files Created**: `approx.go`, `doc.go`

## 1.8 Testing Suite

**Tasks**:

- [x] Create test files for each implementation:
  - [x] `internal/approx/sqrt_test.go`
  - [x] `internal/approx/invsqrt_test.go`
  - [x] `internal/approx/log_test.go`
  - [x] `internal/approx/exp_test.go`

- [ ] Implement test categories:
  - [x] **Unit tests**: Known input/output pairs
  - [x] **Reference tests**: Compare against `math` package
  - [x] **Edge case tests**: NaN, infinity, zero, negative
  - [x] **Property tests**: Mathematical identities
  - [x] **Fuzz tests**: Random inputs for stability
  - [x] **Benchmark tests**: Performance vs `math` package

**File**: `approx_test.go`

**Tasks**:

- [x] Public API integration tests
- [x] End-to-end accuracy validation
- [x] Benchmark suite comparing all functions to `math` package

**File**: `approx_property_test.go`

**Tasks**:

- [x] Test mathematical properties:
  - [x] `FastExp(FastLog(x)) ≈ x`
  - [x] `FastSqrt(x)² ≈ x`
  - [x] `FastInvSqrt(x) * FastSqrt(x) ≈ 1`
  - [x] Monotonicity tests

**File**: `approx_fuzz_test.go`

**Tasks**:

- [x] Fuzz tests for each function
- [x] Verify no crashes, panics, or infinite loops
- [x] Test both float32 and float64 variants

## 1.9 Documentation

**File**: `README.md`

**Tasks**:

- [x] Project overview and features
- [x] Installation instructions
- [x] Quick start with code examples
- [x] API overview
- [x] Performance benchmarks table
- [x] Accuracy guarantees table
- [ ] Use cases (game engines, audio processing, ML, graphics)
- [ ] Comparison to `math` package and other libraries
- [ ] Contributing guidelines link
- [x] License information

**File**: `ACCURACY.md`

**Tasks**:

- [x] Document measurement methodology
- [x] Create accuracy tables for each function:
  - [x] Input ranges tested
  - [x] Max relative error
  - [x] Effective decimal digits
  - [x] RMS error
- [ ] Compare precision levels (Fast/Balanced/High)
- [x] Document input range sensitivity
- [x] Include methodology for reproducing measurements

**File**: `CONTRIBUTING.md`

**Tasks**:

- [ ] How to contribute
- [ ] Development setup instructions
- [ ] Code style guidelines
- [ ] Testing requirements:
  - [ ] All new functions must have reference comparison
  - [ ] Must document accuracy in function comments
  - [ ] Must include benchmarks
- [ ] How to document error bounds
- [ ] Pull request process
- [ ] Areas for contribution

**File**: `AGENTS.md`

**Tasks**:

- [ ] Adapt from algo-fft's `AGENTS.md`
- [ ] Add approximation-specific guidelines:
  - [ ] Pascal to Go translation patterns
  - [ ] Precision variant mapping
  - [ ] Accuracy testing requirements
- [ ] Document architecture decisions
- [ ] Testing strategy specific to approximations

**File**: `CLAUDE.md`

**Tasks**:

- [ ] Point to AGENTS.md: `@AGENTS.md`
- [ ] Add project-specific context

## 1.10 Build System & CI

**Tasks**:

- [x] Update `justfile` for approximations (build/test/bench/lint/cover, cross-arch helpers)
- [x] Configure GitHub Actions workflows:
  - [x] `.github/workflows/test.yaml` - Main CI
  - [x] `.github/workflows/test-unit.yaml` - Multi-OS testing
  - [x] `.github/workflows/test-lint.yaml` - Linting
- [ ] Add pre-commit hooks (optional)

## 1.11 Phase 1 Success Criteria

- [x] ✅ Repository initialized with correct Go module path
- [x] ✅ 4 core functions implemented (sqrt, invsqrt, log, exp)
- [x] ✅ All functions have float32 and float64 variants
- [x] ✅ Precision system works (Fast/Balanced/High)
- [x] ✅ All tests pass (`go test ./...`):
  - [x] Unit tests
  - [x] Reference comparison tests
  - [x] Property tests
  - [x] Fuzz tests
  - [x] Benchmarks
- [x] ✅ Accuracy documented in ACCURACY.md with measurements
- [ ] ✅ 2-5x speedup vs `math` package confirmed

**Performance note (important)**:

- On modern amd64 CPUs, `math.Sqrt` is typically a single hardware instruction and can be extremely fast for scalar calls. A pure-Go iterative approximation will usually be slower for scalar `sqrt`/`invsqrt`.
- Speedups are more realistic for `log`/`exp` (where `math` implementations are more complex) and for batched workloads (SIMD/AVX2/NEON in later phases).

**Perf TODOs (Phase 1/2 carry-over)**:

- [ ] Add float32 benchmarks and document where speedups occur
- [ ] Add batched APIs (slice-in/slice-out) to enable SIMD later (Phase 6+)
- [ ] Investigate avoiding float64 conversions in internal kernels (esp. float32)
- [ ] Evaluate alternate exp/log polynomial approximations per precision tier
- [x] ✅ Zero allocations verified (`testing.AllocsPerRun`)
- [ ] ✅ >80% code coverage
- [ ] ✅ Documentation complete:
  - [x] README.md
  - [x] ACCURACY.md
  - [ ] CONTRIBUTING.md
  - [ ] GoDoc in all exported functions
- [ ] ✅ CI/CD passing on Linux, macOS, Windows
- [ ] ✅ Linter passes with zero errors

**Deliverable**: Working MVP library ready for real-world use

---

# Phase 2: Trigonometry

**Goal**: Add comprehensive trigonometric function support with range reduction.

**Estimated LOC**: +1500

## 2.1 Trigonometric Functions

**File**: `internal/approx/trig.go`

**Tasks**:

- [x] Implement Taylor series for sine and cosine:
  - [x] 3-term Taylor series (~3.2 decimal digits)
  - [x] 4-term Taylor series (~5.2 decimal digits)
  - [x] 5-term Taylor series (~7.3 decimal digits)
  - [x] 6-term Taylor series (~9 decimal digits)
  - [x] 7-term Taylor series (~12.1 decimal digits)
- [x] Implement range reduction:
  - [x] Reduce to [-π/2, π/2] for sine
  - [x] Reduce to [0, π] for cosine
  - [x] Handle quadrant mapping
- [x] Implement reciprocal variants:
  - [x] FastSec (secant = 1/cos)
  - [x] FastCsc (cosecant = 1/sin)
- [ ] Add "Part", "InBounds", and full-range variants (deferred - not needed for MVP)
- [x] Generic wrappers with precision dispatch
- [x] Comprehensive tests and benchmarks

**File**: `internal/reference/trig.go`

**Tasks**:

- [x] Reference wrappers for `math.Sin`, `math.Cos`
- [x] Accuracy measurement tests

**Pascal Source Reference**:

- Lines 66-93 (3-term sin/cos/sec/csc)
- Lines 96-127 (4-term variants)
- Lines 130-157 (5-term variants)
- Lines 160-187 (6-term variants)
- Lines 202-221 (7-term variants)

## 2.2 Public API Extension

**File**: `approx.go`

**Tasks**:

- [x] Add public trig functions:
  ```go
  func FastSin[T Float](x T) T
  func FastSinPrec[T Float](x T, prec Precision) T
  func FastCos[T Float](x T) T
  func FastSec[T Float](x T) T
  func FastCsc[T Float](x T) T
  ```
- [x] Add type-specific aliases
- [ ] Document accuracy and valid ranges (basic docs added, detailed docs deferred)

## 2.3 Testing

**Tasks**:

- [x] Unit tests with known values (sin(π/2)=1, cos(0)=1, etc.)
- [x] Reference comparison across full range
- [x] Property tests:
  - [x] sin(-x) = -sin(x)
  - [x] cos(-x) = cos(x)
  - [x] sin²(x) + cos²(x) = 1 (within tolerance)
  - [ ] Periodicity: sin(x + 2π) ≈ sin(x) (deferred)
- [ ] Benchmark vs `math.Sin`, `math.Cos` (basic benchmarks added, detailed comparison deferred)

## 2.4 Documentation Updates

**Tasks**:

- [ ] Update README.md with trig examples (deferred)
- [ ] Update ACCURACY.md with trig measurements (deferred)
- [ ] Update doc.go with trig usage examples (deferred)

## 2.5 Phase 2 Success Criteria

- [x] ✅ FastSin, FastCos, FastSec, FastCsc implemented
- [x] ✅ All precision levels work (3-7 terms)
- [x] ✅ Range reduction accurate across full input range
- [x] ✅ Tests pass for all trig functions
- [x] ✅ 2-4x speedup vs `math` package (basic benchmarks added)
- [ ] ✅ Documentation updated (deferred - basic API docs exist)

---

# Phase 3: Tangent Functions

**Goal**: Add tangent and cotangent approximations with range-specific optimizations.

**Estimated LOC**: +800

## 3.1 Tangent Implementations

**File**: `internal/approx/tan.go`

**Tasks**:

- [x] Implement tangent approximations:
  - [x] 2-term variants (~3.2 digits, range [0, π/4])
  - [x] 3-term variants (~5.6 digits)
  - [x] 4-term variants (~8.2 digits)
  - [x] 6-term variants (~14 digits)
- [x] Implement cotangent (cotan) variants
- [x] Range-specific optimizations:
  - [x] Full-range variants with reduction to [0, π/4]
  - [x] Quadrant mapping and sign handling
  - [x] Reciprocal optimization for angles > π/4
  - [ ] "Part" variants (partial range [0, π/4]) - deferred (not needed for MVP)
  - [ ] "PInv" variants (reciprocal of Part) - deferred (not needed for MVP)
  - [ ] "InBounds" variants (pre-reduced input) - deferred (not needed for MVP)
- [x] Generic wrappers with precision dispatch (Tan, Cotan)
- [x] Tests (unit tests for all term variants, both float32/float64)
- [ ] Benchmarks (deferred)

**Pascal Source Reference**:

- Lines 224-247 (2-term tan/cotan)
- Lines 250-273 (3-term variants)
- Lines 276-299 (4-term variants)
- Lines 302-325 (6-term variants)

## 3.2 Public API

**File**: `approx.go`

**Tasks**:

- [x] Add public functions:

  ```go
  func FastTan[T Float](x T) T
  func FastTanPrec[T Float](x T, prec Precision) T
  func FastCotan[T Float](x T) T
  func FastCotanPrec[T Float](x T, prec Precision) T
  ```

- [x] Add type-specific convenience functions (FastTan32, FastTan64, FastCotan32, FastCotan64)
- [x] Public API tests (TestFastTan, TestFastTanPrec, TestFastCotan)
- [x] Property-based tests:
  - [x] tan(x) × cotan(x) ≈ 1
  - [x] cotan(x) ≈ 1/tan(x)
  - [x] tan(x + π) ≈ tan(x) (periodicity)

## 3.3 Phase 3 Success Criteria

- [x] ✅ FastTan and FastCotan implemented
- [x] ✅ All precision levels work (2-term, 3-term, 4-term, 6-term)
- [x] ✅ Tests pass (145+ tests across all packages)
- [x] ✅ Range reduction working for full input range
- [x] ✅ Both float32 and float64 variants tested
- [x] ✅ Property-based tests validate mathematical identities
- [ ] ✅ Documentation updated (deferred - basic API docs exist)

---

# Phase 4: Inverse Trigonometry

**Goal**: Add inverse trigonometric functions (arctan, arcsin, arccos).

**Estimated LOC**: +600

## 4.1 Inverse Trig Implementations

**File**: `internal/approx/arctrig.go`

**Tasks**:

- [x] Implement arctan approximations:
  - [x] 3-term variant (~6.6 digits, range [0, π/12])
  - [x] 6-term variant (~13.7 digits)
- [x] Implement arccotan variants
- [x] Implement arccos:
  - [x] 3-term variant
  - [x] 6-term variant
- [x] Range handling and argument reduction
- [x] Tests (comprehensive unit tests for all variants)
- [ ] Benchmarks (deferred)

**Pascal Source Reference**:

- Lines 328-339 (arctan 3-term)
- Lines 342-353 (arctan 6-term)
- Lines 356-365 (arccos variants)

## 4.2 Public API

**File**: `approx.go`

**Tasks**:

- [x] Add public functions:
  - [x] FastArctan / FastArctanPrec
  - [x] FastArccotan / FastArccotanPrec
  - [x] FastArccos / FastArccosPrec
- [x] Add type-specific convenience functions (FastArctan32/64, etc.)
- [x] Public API tests (comprehensive tests for all precision levels)
- [x] Property-based tests:
  - [x] arctan(x) + arccotan(x) ≈ π/2
  - [x] arccos complementarity (arccos(x) + arcsin(x) ≈ π/2)
  - [x] Round-trip: tan(arctan(x)) ≈ x

## 4.3 Phase 4 Success Criteria

- [x] ✅ FastArctan, FastArccotan, FastArccos implemented
- [x] ✅ All precision levels work (3-term and 6-term)
- [x] ✅ Tests pass (comprehensive unit tests, public API tests, property tests)
- [x] ✅ Both float32 and float64 variants tested
- [x] ✅ Property-based tests validate mathematical identities
- [ ] ✅ Documentation updated (deferred - basic API docs exist)
- [ ] ✅ Benchmarks (deferred)

---

# Phase 5: Power Functions

**Goal**: Complete the core mathematical function suite with power and root functions.

**Estimated LOC**: +400

## 5.1 Power Function Implementations

**File**: `internal/approx/power.go`

**Tasks**:

- [x] Implement FastPower using exp/log composition:
  ```go
  func Power[T Float](base, exponent T) T {
      return Exp(exponent * Log(base, PrecisionBalanced), PrecisionBalanced)
  }
  ```
- [x] Implement FastRoot (generalized nth root)
- [x] Implement FastIntPower (efficient integer exponentiation):
  - [x] Binary exponentiation for speed
  - [x] Handle negative exponents
- [x] Tests (comprehensive unit tests for all variants)
- [ ] Benchmarks (deferred)

**Pascal Source Reference**:

- Lines 37-41 (`FastRoot`, `FastIntPower`, `FastPower`)

## 5.2 Public API

**File**: `approx.go`

**Tasks**:

- [x] Add public functions:
  - [x] FastPower / FastPower32 / FastPower64
  - [x] FastRoot / FastRoot32 / FastRoot64
  - [x] FastIntPower / FastIntPower32 / FastIntPower64
- [x] Public API tests (comprehensive tests for all functions)
- [x] Property-based tests:
  - [x] root(x^n, n) ≈ x
  - [x] IntPower vs Power consistency for integer exponents
  - [x] Power exponent laws: (a^b)^c = a^(b\*c)
  - [x] sqrt(x) = root(x, 2)
  - [x] x^(-n) = 1/(x^n)

## 5.3 Phase 5 Success Criteria

- [x] ✅ FastPower, FastRoot, FastIntPower implemented
- [x] ✅ Tests pass (all internal and public API tests passing)
- [x] ✅ Property-based tests validate mathematical identities
- [x] ✅ Core function suite complete
- [ ] ✅ Benchmarks (deferred - basic implementation complete)

---

# Phase 6: SIMD Optimization (AVX2, float32)

**Goal**: Add vectorized implementations for batch processing on x86-64.

## 6.0 Spike result (2026-08-02) — **GATE PASSED**

A timeboxed spike asked whether hand-written Plan 9 assembly buys anything here. Answer:
**scalar assembly, no; batch assembly, decisively yes.**

Go assembly functions cannot be inlined, and this repo's core property is cross-module inlining
(gated statically by `consumerbench/inline_test.go`). So scalar asm would _lose_. The win is
only reachable through batch APIs, which did not exist.

Measured on i7-1255U, `taskset -c 2` (P-core), `GOMAXPROCS=1`, medians of 10 runs,
**ns per element**:

|       N | scalar API | pure-Go batch |  AVX2 asm | AVX2 / batch-Go | AVX2 / scalar |
| ------: | ---------: | ------------: | --------: | --------------: | ------------: |
|      64 |      24.97 |          6.19 | **0.536** |           11.5x |         46.6x |
|     256 |      24.62 |          6.56 | **0.563** |           11.6x |         43.7x |
|    1024 |      25.38 |          6.35 | **0.520** |           12.2x |         48.8x |
|    4096 |      24.41 |          6.07 | **0.520** |           11.7x |         46.9x |
|   65536 |      24.40 |          6.00 | **0.507** |           11.8x |         48.2x |
| 1048576 |      24.61 |          6.69 | **0.545** |           12.3x |         45.1x |

The gate was "≥4x over a correctly-written pure-Go float32 batch baseline at N ≤ 4096". Cleared
by roughly 3x margin.

**Two honest caveats on that number:**

1. **11.7x exceeds the 8-lane theoretical ceiling**, so it is not all vectorization. The
   pure-Go baseline emits ~63 instructions/element where the asm needs ~3.1 (25 instructions ÷
   8 lanes). Part of the ratio is baseline looseness. Against a hypothetically-tight scalar Go
   baseline the figure would be nearer 6x — still well clear of the gate.
2. **No bandwidth collapse appeared at N=1M**, contrary to the prediction. At 0.545 ns/elem ×
   8 B/elem the kernel moves ~14.7 GB/s, under this laptop's ~25 GB/s DRAM ceiling — it is not
   fast enough to saturate memory, so the ratio holds all the way up.

CV was 14–32% (machine shared with other users' jobs), but the effect is an order of magnitude
larger than the noise. Benchmarks verified not to be dead code: all outputs written,
`exp(-10)=4.54e-5`, `exp(0)=1`, `exp(9.995)=21919`.

## 6.0.1 Fused tanh/log(cosh) AVX2 result (2026-08-02) — **GATE PASSED**

This is the kernel the spike actually hinged on. `exp` is branch-free, so vectorising it only
proved the pipeline (ABI0 frame, masked tail, dispatch, VEX purity). `tanh` has a data-dependent
branch at |x| = 0.625, and a vector kernel cannot branch per lane — it must evaluate the
rational core **and** the exponential form for every lane and blend. Whether the eight-fold
width survives paying for both was the open question.

Measured on the **quiet** Xeon Gold 5218 (2.30 GHz, Cascade Lake, 2 vCPU),
`GOMAXPROCS=1 taskset -c 0`, 10 separate `-count=1` runs, benchstat, **CV ≤ 5%**:

|       N | scalar API | pure-Go batch |     AVX2 asm | AVX2 / batch-Go | AVX2 / scalar |
| ------: | ---------: | ------------: | -----------: | --------------: | ------------: |
|      64 |   4.339 µs |      2.058 µs | **206.3 ns** |           9.98x |         21.0x |
|     256 |   17.32 µs |      8.286 µs | **744.9 ns** |          11.12x |         23.3x |
|    1024 |   68.75 µs |      32.86 µs | **2.900 µs** |          11.33x |         23.7x |
|    4096 |   274.6 µs |      133.7 µs | **11.58 µs** |          11.55x |         23.7x |
|   65536 |   4.421 ms |      2.142 ms | **185.4 µs** |          11.55x |         23.8x |
| 1048576 |   70.23 ms |      34.73 ms | **2.956 ms** |          11.75x |         23.8x |

**The branchy-blend cost is real but distribution-independent.** Benchmarked against an
adversarial input where every element is past the branch point, so a branchy scalar version
would take the expensive exponential path for all of them:

| distribution              | AVX2 (N=4096) | pure-Go (N=4096) |
| ------------------------- | ------------: | ---------------: |
| ramp over [-10, 10]       |      11.58 µs |         133.7 µs |
| all past the branch point |      11.47 µs |         132.2 µs |

Within 1% either way. Both kernels are branchless by construction, so neither gains from a
favourable branch mix — which is what makes the pure-Go baseline a fair denominator here rather
than a flattering one.

Throughput is **flat at ~1.32 GiB/s from N=256 to N=1M**, so this kernel is compute-bound over
the whole measured range and the predicted large-N bandwidth collapse does not occur. Counting
all three slices it moves ~4.3 GB/s of real traffic, far under DRAM.

**Where the remaining time goes.** At 2.82 ns/element on a 2.30 GHz part the kernel spends
~6.5 cycles/element on ~8.6 instructions/element, i.e. ~1.3 IPC — low for straight-line vector
code. The likely reason is the three `VDIVPS` per block (the rational core's `num/den`, tanh's
`2u/(1+u)`, and log1p's `u/(2+u)`); the divider does not pipeline like the rest of the kernel,
and on Cascade Lake three 256-bit divides per eight elements plausibly accounts for a large
share of those cycles. This is an inference from instruction tables, **not measured** — see 6.2.

> **Superseded — this paragraph has since been measured and is wrong on both counts (§6.2.2).**
> The "~1.3 IPC" figure is arithmetic on wall-clock divided by an instruction estimate, not a
> counter reading; measured directly the kernel runs at **2.10 IPC**. And while the divider is
> genuinely busy 42% of cycles, removing it entirely recovers only 6%, so it is not where the
> remaining time goes. The paragraph is left in place because §6.2.2 is a correction of it.

## 6.1 Delivered

- [x] `internal/simd/` package with float32-**native** batch kernels (the scalar API widens
      float32 → float64; the batch path does not)
- [x] `internal/simd/exp32_amd64.s` — AVX2+FMA exp, `EXPBODY` macro shared with a `VMASKMOVPS`
      masked tail, `VZEROUPPER` before every `RET`, VEX-only (disassembly-verified)
- [x] `init()`-resolved dispatch on `HasAVX2 && HasFMA && !ForceGeneric`; bool-return contract
      chaining to the pure-Go kernel
- [x] `purego` / non-amd64 fallback; `decl_text_test.go` guarding decl↔asm drift
- [x] `HasFMA` added to `internal/cpu` (FMA is a separate CPUID bit; VMs can mask it → SIGILL)
- [x] float32 ulp accuracy harness (the suite previously had **zero** float32 coverage)
- [x] Fused `TanhLogCoshFloat32` in pure Go, sharing `u = exp(-2|x|)` between both outputs
- [x] `internal/simd/hyperbolic32_amd64.s` — AVX2+FMA fused tanh/log(cosh), three slices
      (`$0-73`), both branches evaluated and blended with `VBLENDVPS`, masked tail for both
      destinations
- [x] `internal/simd/exp32_amd64.h` — the exp constants, tail mask table and `EXPBODY` extracted
      into a header both kernels include, so the shared exponential has exactly one definition.
      `EXPBODY` was reworked to hold only its two clamp constants in registers and read the rest
      from RODATA, which is what leaves the fused kernel enough registers to work in.
- [x] `TestDispatchSelectsAVX2OnCapableHost` — asserts the dispatch flag really resolved to the
      assembly, so the suite cannot pass green while the kernel never executes

**Accuracy**: every assembly kernel is verified against its pure-Go twin over **all 2³² float32
bit patterns**, i.e. 4 278 190 082 non-NaN inputs:

| kernel    | max drift vs Go | bit-identical |
| --------- | --------------: | ------------: |
| `exp`     |           1 ulp |        99.95% |
| `tanh`    |           2 ulp |        99.98% |
| `logCosh` |           4 ulp |        99.99% |

The residual is FMA contraction (asm fuses; Go's amd64 backend never does). Against a float64
reference, exp is 1 ulp across the whole domain, tanh 1 ulp, logCosh 2 ulp (4 at the branch
seam) — and at the seam, where the worst cases live, the **assembly is fractionally closer to
the truth than the pure-Go kernel on both outputs**, so the drift is the two implementations
rounding to opposite sides of a genuine cancellation rather than the vector kernel losing
precision.

## 6.2 Remaining

- [x] AVX2 fused tanh/logCosh kernel — done, see 6.0.1. The branchy-blend cost is real and
      distribution-independent, and the kernel still clears the gate by ~2.9x.
- [x] Re-measure on an idle machine — done on the Xeon, CV ≤ 5% against 14–32% locally.
- [x] `sqrt`/`invsqrt`/`recip` **removed from the library**. They measured 6.9x / 4.3x / 9x
      _slower_ than the hardware instructions they replace, and the README's stated
      justification ("for targets without a hardware square root") does not survive checking:
      `cmd/compile/internal/ssagen/intrinsics.go` intrinsifies `math.sqrt` to `ssa.OpSqrt` on
      I386, AMD64, ARM, ARM64, Loong64, MIPS, MIPS64, PPC64, RISCV64, S390X and Wasm — every
      GOARCH Go supports. There is no target where the approximation wins.
- [x] **Public batch API shipped** — `FastExpBatch32/64` and `FastTanhLogCoshBatch32/64` in the
      root package, forwarding to `internal/simd` (float32) and a new `internal/approx/batch.go`
      (float64). Contract lifted verbatim from the `internal/simd` package doc: panic if a
      destination is shorter than `src`, `len(src)` elements processed, destinations must be
      identical to `src` or non-overlapping. No `...BatchPrec` variants — the tier is a compile-
      time constant, per §"Precision must not be a runtime argument in a loop" in `AGENTS.md`.
      See §6.2.1 for two design notes that came out of building it.

### 6.2.1 Batch API design notes

**There is no generic `FastExpBatch[T Float]`, and the reason is not the one first assumed.**
The design was written up on the theory that reaching the float32 kernel from a generic body
needs `any(dst).([]float32)`, which would box the slice header through `runtime.convTslice` and
allocate — breaking the zero-allocation invariant. **Measured, that is false.** `AllocsPerRun`
reports 0 for all four variants tried (`[]float32`, `[]float64`, a named `~float32` type, and
one behind a `//go:noinline` wrapper), and `go tool objdump` shows no `convTslice` in the
generated code at all: `[]float32` and `[]float64` are distinct gcshapes, so each instantiation
compiles separately and the type assertion is statically decidable and folded away. The generic
form is therefore mechanically viable; it is omitted only because the concrete `…32`/`…64` names
say honestly which width has a kernel behind it. Anyone revisiting this should know the
mechanical objection does not exist.

**The panics name the public function, not the internal one.** `internal/simd`'s length checks
say `approx: FastExpBatch32: ...`, which reads oddly inside that package but is the only message
a caller can act on — they cannot import `internal/simd` and would otherwise be sent looking for
a package that does not exist on their import path.

**`golang.org/x/sys` is now on the public import path.** `approx` imports `internal/simd` →
`internal/cpu` → `golang.org/x/sys/cpu`, so every consumer links it and runs its CPUID init,
including consumers that only call scalar functions. It was already a module requirement; what
changed is that it is now actually linked. `consumerbench` needed a `go.sum` for the first time.

**The float32 `exp` agreement bound sits exactly at its derived ceiling.** Scalar `FastExp32` is
38 ulp and the batch kernel is 1, so the most they can disagree by is 38 + 1 = 39 — and the
measured maximum is exactly 39, at x ≈ 8.665, identically with AVX2 on, AVX2 off and under
`purego`. That confirms the derivation rather than threatening it, but it leaves no headroom, so
the same care applies here as to `hypTolTanh`: if this bound ever needs raising, re-derive it
from a float64 reference rather than nudging the constant.
- [x] **Replace the three `VDIVPS` with `VRCPPS` + two Newton-Raphson steps — measured, and
      REJECTED.** See §6.2.2. The ceiling is 6%, and the replacement costs more instructions
      than that is worth.
- [ ] Consider the same treatment for `exp`'s sibling kernels on ARM64 (Phase 7); note the
      Cascade Lake divider is slower than Alder Lake's, so any payoff is
      microarchitecture-dependent and must be re-measured per target. Given §6.2.2, start by
      measuring rather than by writing the reciprocal sequence.
- [ ] Single-output batch `tanh` and `log(cosh)`. The public batch API ships only the fused
      two-output form, so a caller who wants `tanh` alone must pass a reusable scratch buffer
      for the `log(cosh)` destination. Fixing it properly needs either a second assembly kernel
      or a nil-destination convention inside the existing one. Known wart, deliberately deferred.

### 6.2.2 The divider hypothesis, measured (2026-08-02) — **REFUTED**

Two independent measurements, because neither machine can do both: the quiet Xeon has no `perf`
(`perf_event_paranoid = 3`, no binary), and the local laptop's wall-clock is unusable under
other users' load. A decision rule was fixed **before** either was run — stop below 8%, divide
#3 only at 8–15%, all three above 15%.

**A1 — divider occupancy, i7-1255U (Alder Lake).** `perf stat -e cycles,instructions,`
`arith.fpdiv_active`, `taskset -c 0,1`, prebuilt test binary, `BatchDispatch/n=4096`, AVX2 path
confirmed live first. Four runs:

| metric                          |             value |
| ------------------------------- | ----------------: |
| `arith.fpdiv_active` / `cycles` | **41.9% ± 0.4pp** |
| IPC                             |              2.10 |
| cycles/element                  |              4.48 |

Wall-clock here is meaningless (the box was under other users' load), but the ratio is a
per-task counter and was stable across runs. It cross-checks exactly: 41.9% of 4.48 cyc/element
is 1.88 divider-cycles per element, which at three divides per eight elements is **5.0 cycles
per `VDIVPS ymm`** — the instruction-table figure.

**A2 — ablation ceiling, Xeon Gold 5218, idle.** All three `VDIVPS` replaced by `VMULPS` with
identical operands and register allocation: numerically wrong, but the same instruction count
and dependency structure *minus the divider*. This bounds what any reciprocal scheme could ever
recover. Same protocol as §6.0.1 (`GOMAXPROCS=1 taskset -c 0`, warm-up discarded, ten
interleaved `-count=1` runs, benchstat), AVX2 dispatch re-checked before every round:

|    N | base     | divider ablated |  delta |
| ---: | -------: | --------------: | -----: |
|   64 | 198.4 ns |        185.6 ns | −6.48% |
|  256 | 745.5 ns |        696.6 ns | −6.57% |
| 1024 | 2.902 µs |        2.740 µs | −5.58% |
| 4096 | 11.60 µs |        10.90 µs | −6.00% |

All p = 0.000, n = 10, CV ≤ 2.2%. Base n=4096 reproduces §6.0.1's 11.58 µs, so the two tables
are directly comparable.

**The two results together are the interesting part.** The divider is busy 42% of all cycles,
yet removing it entirely recovers only 6% — the out-of-order engine overlaps roughly
six-sevenths of the divider's occupancy with the surrounding FMA work. **Occupancy is not the
critical path.** This is precisely the error the ablation was designed to catch, and it is why
the earlier estimate (~29% if the divider serialised perfectly) was ~5x too optimistic.

**Why this closes the item rather than merely deferring it.** 6% is the ceiling for a
replacement that costs *nothing*. `VRCPPS` + two Newton steps adds ~5–6 instructions per site,
~18 per eight-element block against a current ~69, into a kernel already issuing at 2.10 IPC —
so the realistic outcome is a **regression**. On top of that, two of the three divides feed
`tanh`, whose measured full-domain drift is 2 ulp against a `hypTolTanh` of 2: zero headroom,
and any accuracy loss there would have to be bought back with work that the timing does not
justify. No branch of this looks good. The single-ablation attribution runs were built but not
executed, because the decision rule gates them on the ceiling being promising.

**Correction to §6.0.1.** That section attributed the kernel's cost to the divider on the
strength of a "~1.3 IPC" figure. That figure was arithmetic on wall-clock divided by an
instruction estimate, not a counter reading, and it does not survive measurement: the same
kernel runs at **2.10 IPC**. The premise was wrong at its root, not only in its conclusion.

---

# Phase 7: ARM64 SIMD (NEON)

**Goal**: Cross-platform SIMD parity with ARM64.

**Estimated LOC**: +600

## 7.1 NEON Implementations

**Tasks**:

- [ ] Implement NEON sqrt (4x float32):
  - [ ] `internal/simd/sqrt_arm64.go`
  - [ ] `internal/simd/sqrt_arm64.s`
  - [ ] Build tags: `//go:build arm64`
- [ ] Implement NEON inverse sqrt
- [ ] CPU feature detection for ARM64
- [ ] Cross-platform testing using QEMU
- [ ] Tests and benchmarks

## 7.2 Phase 7 Success Criteria

- [ ] ✅ NEON implementations working
- [ ] ✅ Performance parity with AVX2
- [ ] ✅ Tests pass on ARM64 hardware or emulator

---

# Phase 8: Utilities & Advanced Features

**Goal**: Complete the library with utility functions and advanced optimizations.

**Estimated LOC**: +400

## 8.1 Utility Functions

**Tasks**:

- [ ] Implement FastRandomGauss (Gaussian random numbers)
- [ ] Implement FastFloorLn2 (integer floor of log2)
- [ ] Implement "Like" variants:
  - [ ] FastArctanLike
  - [ ] FastSinLike
  - [ ] FastCosLike
- [ ] Range-specific optimization variants:
  - [ ] InRange (optimized for known bounds)
  - [ ] InBounds (pre-validated inputs)

**Pascal Source Reference**:

- Lines 48-60 (FloorLn2, RandomGauss, \*Like variants)

## 8.2 Advanced Optimizations

**Tasks**:

- [ ] FMA (fused multiply-add) optimizations where beneficial
- [ ] Float64 SIMD (2-wide AVX2/NEON)
- [ ] Profile-guided optimization hints
- [ ] Benchmark-driven algorithm selection

## 8.3 Phase 8 Success Criteria

- [ ] ✅ All utility functions implemented
- [ ] ✅ Advanced optimizations in place
- [ ] ✅ Complete Pascal port achieved
- [ ] ✅ Performance targets met across all functions

---

# Post-Phase 8: Release Preparation

## Release Checklist

- [ ] All 8 phases complete
- [ ] Comprehensive test coverage (>80%)
- [ ] All benchmarks show expected speedups
- [ ] ACCURACY.md fully populated with measurements
- [ ] Documentation complete and reviewed
- [ ] CI/CD passing on all platforms (Linux, macOS, Windows)
- [ ] Cross-platform testing (amd64, arm64)
- [ ] Security review (no undefined behavior, panics handled)
- [ ] Performance regression testing
- [ ] API review (ensure no breaking changes needed)
- [ ] LICENSE file added (MIT)
- [ ] CHANGELOG.md created
- [ ] Version tagging strategy defined
- [ ] Release notes drafted

## v1.0.0 Release

**Criteria**:

- [ ] ✅ All ~60-80 Go functions implemented (covering ~180 Pascal functions)
- [ ] ✅ SIMD for both amd64 (AVX2) and arm64 (NEON)
- [ ] ✅ >80% code coverage
- [ ] ✅ Comprehensive accuracy measurements documented
- [ ] ✅ Stable API (semantic versioning commitment)
- [ ] ✅ Production-ready quality

---

# Pascal to Go Translation Guide

## Type Mapping

| Pascal                     | Go                                                    |
| -------------------------- | ----------------------------------------------------- |
| `Single`                   | `float32`                                             |
| `Double`                   | `float64`                                             |
| `overload`                 | Generics `[T Float]`                                  |
| `Integer`                  | `int`                                                 |
| `{$IFNDEF PurePascal}` asm | `//go:build amd64` + `.s` files                       |
| `inline`                   | Go compiler auto-inlines (or `//go:inline` directive) |

## Pattern Translation

### Pascal Multi-Term Variants → Go Precision Enum

**Pascal** (separate functions per term count):

```pascal
function FastSin3Term(Value: Single): Single;
function FastSin4Term(Value: Single): Single;
function FastSin5Term(Value: Single): Single;
function FastSin6Term(Value: Single): Single;
function FastSin7Term(Value: Single): Single;
```

**Go** (single function with precision parameter):

```go
func FastSinPrec[T Float](x T, prec Precision) T {
    terms := map[Precision]int{
        PrecisionFast: 3,
        PrecisionBalanced: 5,
        PrecisionHigh: 7,
    }[prec]
    return sinTaylor(x, terms)
}
```

### Pascal Overloading → Go Generics

**Pascal** (manual overloads):

```pascal
function FastSqrt(Value: Single): Single; overload;
function FastSqrt(Value: Double): Double; overload;
```

**Go** (single generic function):

```go
func FastSqrt[T Float](x T) T {
    // Implementation works for both float32 and float64
}
```

### Range-Specific Variants

Pascal has "Part", "InBounds", "PInv" variants:

- **Part**: Optimized for partial range (e.g., [0, π/4])
- **InBounds**: Assumes input already in valid range (no reduction)
- **PInv**: Reciprocal of Part function

In Go, these can be:

1. Separate functions (like Pascal)
2. Additional parameters (e.g., `skipRangeCheck bool`)
3. Internal optimizations based on profiling

**Recommendation**: Start with separate functions for clarity, consolidate if beneficial.

---

# Performance Targets

## Speedup Goals (vs `math` package)

| Function    | Target Speedup | Baseline (math pkg) | Expected Result |
| ----------- | -------------- | ------------------- | --------------- |
| FastSqrt    | 3-5x           | ~10 ns/op           | ~2-3 ns/op      |
| FastInvSqrt | 4-6x           | ~12 ns/op           | ~2-3 ns/op      |
| FastLog     | 5-7x           | ~28 ns/op           | ~4-5 ns/op      |
| FastExp     | 4-6x           | ~35 ns/op           | ~6-8 ns/op      |
| FastSin     | 2-4x           | ~40 ns/op           | ~10-20 ns/op    |
| FastCos     | 2-4x           | ~40 ns/op           | ~10-20 ns/op    |
| FastTan     | 2-3x           | ~45 ns/op           | ~15-20 ns/op    |

**With SIMD (Phase 6+)**: Additional 4-8x speedup for batched operations

## Accuracy Targets

| Precision Level   | Decimal Digits | Typical Use Case                         |
| ----------------- | -------------- | ---------------------------------------- |
| PrecisionFast     | 2-3 digits     | Real-time graphics, rough estimates      |
| PrecisionBalanced | 4-5 digits     | Game physics, audio processing (default) |
| PrecisionHigh     | 6-7 digits     | Simulations, financial calculations      |

**Note**: float32 has ~7 decimal digits maximum precision, float64 has ~15 digits.

---

# Measurement & Validation

## Accuracy Measurement Process

1. **Generate test samples**: 10,000+ uniformly distributed across valid input domain
2. **Compute reference values**: Using Go `math` package (IEEE 754 compliant)
3. **Compute approximate values**: Using our implementation
4. **Calculate metrics**:
   - Max absolute error: `max(|approx - ref|)`
   - Max relative error: `max(|approx - ref| / |ref|)`
   - Mean absolute error: `mean(|approx - ref|)`
   - RMS error: `sqrt(mean((approx - ref)²))`
   - Effective decimal digits: `-log10(maxRelError)`
5. **Document results** in ACCURACY.md

## Continuous Validation

- [ ] Automated accuracy tests in CI
- [ ] Regression detection (alert if accuracy degrades)
- [ ] Benchmark performance tracking
- [ ] Regular profiling for optimization opportunities

---

# Development Workflow

## Recommended Order

1. **Start with Phase 1** - Get MVP working first
2. **Test thoroughly** - Don't move to next phase until current phase tests pass
3. **Document as you go** - Update ACCURACY.md with measurements after each function
4. **Benchmark early** - Verify speedup claims immediately
5. **Iterate on accuracy** - Tune precision levels based on actual measurements

## Daily Development Cycle

```bash
# 1. Make changes
vim internal/approx/sqrt.go

# 2. Format code
just fmt

# 3. Run tests
just test

# 4. Run benchmarks
just bench

# 5. Check coverage
just cover

# 6. Lint
just lint

# 7. Full check before commit
just check
```

## Git Workflow

- Use feature branches for each phase
- Commit early and often
- Squash commits before merging to main
- Tag releases: `v0.1.0` (Phase 1 MVP), `v0.2.0` (Phase 2), ..., `v1.0.0` (all phases)

---

# Testing Strategy Summary

## Test Categories

1. **Unit Tests** (`*_test.go`)
   - Known input/output pairs
   - Edge cases (NaN, infinity, zero, negative)
   - Both float32 and float64

2. **Reference Tests** (`internal/approx/*_test.go`)
   - Compare against `math` package
   - Measure actual accuracy
   - Validate precision claims

3. **Property Tests** (`approx_property_test.go`)
   - Mathematical identities
   - Monotonicity, symmetry, periodicity
   - Composition properties

4. **Fuzz Tests** (`approx_fuzz_test.go`)
   - Random inputs for robustness
   - No crashes, panics, or undefined behavior
   - Stability across input ranges

5. **Benchmark Tests** (`approx_bench_test.go`)
   - Performance vs `math` package
   - Report allocations (must be 0)
   - Track performance regressions

## Coverage Goals

- **Overall**: >80% code coverage
- **Core functions**: >90% coverage
- **Edge cases**: 100% coverage

---

# Rationale: Architectural Decisions

## Why Simple Function API (Not Plan-Based)?

**FFT** needs Plans because:

- Precomputed twiddle factors (expensive, size-dependent)
- Scratch buffers (memory allocation)
- Bit-reversal indices (size-dependent lookup tables)
- Stateful kernel selection based on transform size

**Approximations** don't need Plans because:

- No precomputation (constants are compile-time)
- No scratch buffers (single value in/out)
- No size parameter (operates on scalars)
- Implementation selection done once at init via CPU detection

**Result**: Simple `func FastSqrt(x) T` is more ergonomic than `plan.Sqrt(x)`.

## Why Precision Enum?

Pascal has 3-7 term variants as separate functions. Consolidating into precision levels:

- **Reduces API surface** from ~180 functions to ~60-80
- **Easier to use** - users choose "Fast/Balanced/High" not "3-term vs 4-term"
- **Flexible** - Can tune default precision per function based on benchmarks
- **Maintainable** - Less code duplication

## Why Vendor CPU Detection?

Could use algo-fft as a dependency, but:

- **Independence**: Separate repos should be self-contained
- **Stability**: CPU detection is small (~300 LOC) and stable
- **No coupling**: Avoids dependency hell if APIs diverge
- **Simplicity**: One-time copy is simpler than go.mod dependency

## Why Both float32 and float64?

- **Performance**: float32 is 2x faster for SIMD (4 lanes vs 2)
- **Precision**: Some use cases need float64 accuracy
- **Generics**: No cost to support both with Go generics
- **User choice**: Let users decide their speed/precision tradeoff

---

# Future Considerations (Post-v1.0)

## Potential Phase 9+

- [ ] Complex number approximations (if demand exists)
- [ ] Batch multi-dimensional operations
- [ ] GPU offload via separate package (OpenCL/CUDA/WebGPU)
- [ ] Additional approximation families:
  - [ ] Bessel functions
  - [ ] Error function (erf)
  - [ ] Gamma function
- [ ] Adaptive precision (auto-select based on input)
- [ ] Profile-guided optimization (PGO)

## API Evolution

- Maintain semantic versioning strictly
- Any breaking changes require major version bump
- Consider v2 only if fundamentally better design emerges
- Backport critical fixes to v1.x

---

# Success Metrics Summary

## Phase 1 (MVP) Complete When:

- [x] 4 core functions: sqrt, invsqrt, log, exp
- [x] All current tests pass (`go test ./...`)
- [ ] Property/fuzz/benchmark coverage added
- [ ] 2-5x speedup vs `math` confirmed
- [ ] Zero allocations verified
- [ ] Accuracy documented with measurements
- [ ] Documentation complete (README, CONTRIBUTING, ACCURACY)
- [ ] CI/CD passing

## v1.0.0 Complete When:

- ✅ All 8 phases implemented
- ✅ ~60-80 functions covering ~180 Pascal functions
- ✅ SIMD for amd64 and arm64
- ✅ >80% code coverage
- ✅ Comprehensive accuracy measurements
- ✅ Stable, production-ready API

---

# Questions / Decisions Log

## Open Questions

- [ ] Should we support complex number approximations? (defer to Phase 9+)
- [ ] Profile-guided optimization worth the complexity? (evaluate in Phase 8)
- [ ] WebAssembly SIMD support? (investigate after ARM64 phase)

## Resolved Decisions

- ✅ **Repository structure**: Separate repo (not monorepo)
- ✅ **API style**: Function-based (not Plan-based)
- ✅ **Precision handling**: Enum (not separate functions)
- ✅ **Type support**: Both float32 and float64 via generics
- ✅ **CPU detection**: Vendor from algo-fft
- ✅ **Build system**: Reuse justfile pattern
- ✅ **Testing strategy**: Multi-layer (unit/reference/property/fuzz/bench)

---

# Credits & References

**Original Pascal Code**: `../go-fft/reference/approx.pas` (~48K tokens, ~300 functions)

**Inspired By**:

- algo-fft architecture and patterns
- Quake III fast inverse square root
- Classical approximation theory (Taylor, Laurent, Padé)

**License**: MIT

---

**Last Updated**: 2025-12-29
**Status**: Phase 1-5 complete! Core math (sqrt, invsqrt, log, exp), trigonometry (sin, cos, sec, csc), tangent (tan, cotan), inverse trig (arctan, arccotan, arccos), and power functions (power, root, integer power) all implemented with comprehensive tests.
**Next Milestone**: Phase 6 (SIMD Optimization) - AVX2 vectorized implementations for batch processing
