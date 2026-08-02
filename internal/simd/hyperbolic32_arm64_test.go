//go:build arm64 && !purego

package simd

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

// asmHyp runs the assembly kernel over inputs and returns (tanh, logCosh).
func asmHyp(tb testing.TB, inputs []float32) ([]float32, []float32) {
	tb.Helper()

	dstTanh := make([]float32, len(inputs))
	dstLogCosh := make([]float32, len(inputs))

	if !tanhLogCoshBatch32NEON(dstTanh, dstLogCosh, inputs) {
		tb.Fatal("tanhLogCoshBatch32NEON declined")
	}

	return dstTanh, dstLogCosh
}

// goHyp runs the pure-Go kernel over inputs and returns (tanh, logCosh).
func goHyp(inputs []float32) ([]float32, []float32) {
	dstTanh := make([]float32, len(inputs))
	dstLogCosh := make([]float32, len(inputs))
	tanhLogCoshBatch32Go(dstTanh, dstLogCosh, inputs)

	return dstTanh, dstLogCosh
}

// How far the assembly and pure-Go kernels may drift from each other.
//
// These are agreement bounds between two IMPLEMENTATIONS, not accuracy bounds.
// Accuracy is pinned separately by TestTanhAccuracy and TestLogCoshAccuracy in
// simd_test.go, which measure against a float64 reference and run against
// whichever kernel dispatch selected.
//
// The numbers are the same as the amd64 file's, and deliberately so: each is
// twice the corresponding accuracy bound, on the reasoning that two kernels
// independently within k ulp of the truth can sit on opposite sides of it and
// be 2k apart while both are correct. That derivation does not depend on the
// architecture, so neither do the bounds.
//
// What DOES differ is how much of that room gets used. On amd64 the two kernels
// disagree because the assembly contracts its multiply-adds and the compiler
// refuses to; measured over the full domain they reach 2 ulp on tanh and 4 on
// log(cosh). On arm64 the compiler contracts too, so both sides evaluate the
// same fused expressions and the drift is far smaller — see the full-domain
// test at the bottom of this file for the measured figures.
//
// The bounds are kept at the derived values rather than tightened to what this
// host happens to reach. FMA scheduling differs between microarchitectures, and
// a bound pinned to one machine's observed maximum is a bound that fails on
// somebody else's.
const (
	hypTolTanh    int64 = 2
	hypTolLogCosh int64 = 8
)

// TestHypNEONMatchesGo sweeps the domain that matters, densely around the
// branch point where both kernels' worst cases live.
func TestHypNEONMatchesGo(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	var inputs []float32

	for x := float32(-20); x <= 20; x += 0.001 {
		inputs = append(inputs, x)
	}

	// The branch seam at |x| = 0.625, one float32 at a time.
	for x := float32(0.6); x <= 0.7; x = math.Float32frombits(math.Float32bits(x) + 1) {
		inputs = append(inputs, x, -x)
	}

	gotT, gotL := asmHyp(t, inputs)
	wantT, wantL := goHyp(inputs)

	var maxTanh, maxLogCosh int64

	for i, x := range inputs {
		if d := ulpDiff32(gotT[i], wantT[i]); d > hypTolTanh {
			t.Fatalf("tanh(%v): asm %v, go %v, %d ulp apart (tol %d)", x, gotT[i], wantT[i], d, hypTolTanh)
		} else {
			maxTanh = max(maxTanh, d)
		}

		if d := ulpDiff32(gotL[i], wantL[i]); d > hypTolLogCosh {
			t.Fatalf("logCosh(%v): asm %v, go %v, %d ulp apart (tol %d)", x, gotL[i], wantL[i], d, hypTolLogCosh)
		} else {
			maxLogCosh = max(maxLogCosh, d)
		}
	}

	t.Logf("over %d inputs: tanh max %d ulp, logCosh max %d ulp", len(inputs), maxTanh, maxLogCosh)
}

// TestHypNEONMatchesGoRandom covers the ranges a regular sweep steps over,
// including the saturation tail and the flush-to-zero end of the exponential.
func TestHypNEONMatchesGoRandom(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	rng := rand.New(rand.NewSource(31337)) //nolint:gosec // deterministic test input.

	inputs := make([]float32, 1<<15)
	for i := range inputs {
		inputs[i] = float32(rng.Float64()*200 - 100)
	}

	gotT, gotL := asmHyp(t, inputs)
	wantT, wantL := goHyp(inputs)

	for i, x := range inputs {
		if d := ulpDiff32(gotT[i], wantT[i]); d > hypTolTanh {
			t.Fatalf("tanh(%v): asm %v, go %v, %d ulp apart", x, gotT[i], wantT[i], d)
		}

		if d := ulpDiff32(gotL[i], wantL[i]); d > hypTolLogCosh {
			t.Fatalf("logCosh(%v): asm %v, go %v, %d ulp apart", x, gotL[i], wantL[i], d)
		}
	}
}

// TestHypNEONTails walks every length from 0 to 16, so the scratch-buffer tail
// runs at each of its three lengths and at several block counts.
func TestHypNEONTails(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	for n := range 17 {
		inputs := make([]float32, n)
		for i := range inputs {
			inputs[i] = float32(i)*0.37 - 3
		}

		gotT, gotL := asmHyp(t, inputs)
		wantT, wantL := goHyp(inputs)

		for i := range inputs {
			if d := ulpDiff32(gotT[i], wantT[i]); d > hypTolTanh {
				t.Fatalf("n=%d i=%d: tanh asm %v, go %v, %d ulp apart", n, i, gotT[i], wantT[i], d)
			}

			if d := ulpDiff32(gotL[i], wantL[i]); d > hypTolLogCosh {
				t.Fatalf("n=%d i=%d: logCosh asm %v, go %v, %d ulp apart", n, i, gotL[i], wantL[i], d)
			}
		}
	}
}

// TestHypNEONTailDoesNotWritePastEnd pins the tail contract for BOTH
// destinations.
//
// The kernel computes a full four-lane vector for the tail and then copies
// n%4 elements out of a scratch buffer into each destination in turn. Two
// separate copy loops means two chances to write one element too many, and the
// second one — log(cosh) — is the easier to get wrong, because by then the
// element counter has already been consumed once.
func TestHypNEONTailDoesNotWritePastEnd(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	const guard = 8

	for n := 1; n < 8; n++ {
		bufT := make([]float32, n+guard)
		bufL := make([]float32, n+guard)

		for i := range bufT {
			bufT[i] = 12345
			bufL[i] = 54321
		}

		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i) + 0.5
		}

		if !tanhLogCoshBatch32NEON(bufT[:n], bufL[:n], src) {
			t.Fatal("declined")
		}

		for i := n; i < n+guard; i++ {
			if bufT[i] != 12345 {
				t.Fatalf("n=%d: tanh kernel wrote past the end at index %d: %v", n, i, bufT[i])
			}

			if bufL[i] != 54321 {
				t.Fatalf("n=%d: logCosh kernel wrote past the end at index %d: %v", n, i, bufL[i])
			}
		}
	}
}

// TestHypNEONOddSymmetryBitExact pins the structural guarantee that survives
// vectorisation only because the kernel masks off the sign, computes on the
// magnitude and XORs the sign back on at the very end.
//
// A branchless SIMD tanh written as a polynomial in x rather than in |x| would
// pass every accuracy test in this package and still fail this one.
func TestHypNEONOddSymmetryBitExact(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	pos := make([]float32, 0, 50000)
	for i := 1; i <= 50000; i++ {
		pos = append(pos, float32(i)*0.0005)
	}

	neg := make([]float32, len(pos))
	for i, v := range pos {
		neg[i] = -v
	}

	posT, posL := asmHyp(t, pos)
	negT, negL := asmHyp(t, neg)

	for i, x := range pos {
		if math.Float32bits(negT[i]) != math.Float32bits(posT[i])^signMask32 {
			t.Fatalf("tanh(-%v)=%v is not the exact negation of tanh(%v)=%v", x, negT[i], x, posT[i])
		}

		if math.Float32bits(negL[i]) != math.Float32bits(posL[i]) {
			t.Fatalf("logCosh(-%v)=%v differs from logCosh(%v)=%v; log(cosh) is even", x, negL[i], x, posL[i])
		}
	}
}

// TestHypNEONInPlace covers dst == src, the one aliasing mode the package
// documents as supported. Either destination may alias the source.
//
// The lengths deliberately are not multiples of four, so the scratch-buffer
// tail runs in-place too. That is the path where the tempting
// overlapping-vector shortcut would silently read back its own output.
func TestHypNEONInPlace(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	inputs := make([]float32, 133)
	for i := range inputs {
		inputs[i] = float32(i)*0.19 - 12
	}

	wantT, wantL := asmHyp(t, inputs)

	t.Run("tanh aliases src", func(t *testing.T) {
		t.Parallel()

		buf := append([]float32(nil), inputs...)
		other := make([]float32, len(inputs))

		if !tanhLogCoshBatch32NEON(buf, other, buf) {
			t.Fatal("declined")
		}

		for i := range inputs {
			if math.Float32bits(buf[i]) != math.Float32bits(wantT[i]) {
				t.Fatalf("i=%d: in-place tanh %v, out-of-place %v", i, buf[i], wantT[i])
			}

			if math.Float32bits(other[i]) != math.Float32bits(wantL[i]) {
				t.Fatalf("i=%d: logCosh %v, want %v", i, other[i], wantL[i])
			}
		}
	})

	t.Run("logCosh aliases src", func(t *testing.T) {
		t.Parallel()

		buf := append([]float32(nil), inputs...)
		other := make([]float32, len(inputs))

		if !tanhLogCoshBatch32NEON(other, buf, buf) {
			t.Fatal("declined")
		}

		for i := range inputs {
			if math.Float32bits(other[i]) != math.Float32bits(wantT[i]) {
				t.Fatalf("i=%d: tanh %v, want %v", i, other[i], wantT[i])
			}

			if math.Float32bits(buf[i]) != math.Float32bits(wantL[i]) {
				t.Fatalf("i=%d: in-place logCosh %v, out-of-place %v", i, buf[i], wantL[i])
			}
		}
	})
}

// TestHypNEONEmpty checks the n == 0 early exit for both empty and nil slices.
func TestHypNEONEmpty(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	if !tanhLogCoshBatch32NEON(nil, nil, nil) {
		t.Fatal("nil: kernel declined")
	}

	if !tanhLogCoshBatch32NEON([]float32{}, []float32{}, []float32{}) {
		t.Fatal("empty: kernel declined")
	}
}

// TestHypNEONShortestLengthWins pins that the element count is the minimum of
// all three slices, and that neither destination is touched past it.
func TestHypNEONShortestLengthWins(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	const guard = 99

	src := make([]float32, 10)
	for i := range src {
		src[i] = float32(i) * 0.25
	}

	dstT := make([]float32, 10)
	dstL := make([]float32, 6) // the binding constraint

	for i := range dstT {
		dstT[i] = guard
	}

	for i := range dstL {
		dstL[i] = guard
	}

	if !tanhLogCoshBatch32NEON(dstT, dstL, src) {
		t.Fatal("declined")
	}

	wantT, wantL := goHyp(src[:6])

	for i := range 6 {
		if d := ulpDiff32(dstT[i], wantT[i]); d > hypTolTanh {
			t.Fatalf("i=%d: tanh %v, want %v", i, dstT[i], wantT[i])
		}

		if d := ulpDiff32(dstL[i], wantL[i]); d > hypTolLogCosh {
			t.Fatalf("i=%d: logCosh %v, want %v", i, dstL[i], wantL[i])
		}
	}

	for i := 6; i < len(dstT); i++ {
		if dstT[i] != guard {
			t.Fatalf("tanh destination written past the shortest length at index %d: %v", i, dstT[i])
		}
	}
}

// TestTanhLogCoshFloat32DispatchMatchesGo exercises the exported wrapper, so
// the dispatch wiring itself is covered rather than only the raw kernel.
func TestTanhLogCoshFloat32DispatchMatchesGo(t *testing.T) {
	t.Parallel()

	inputs := make([]float32, 1000)
	for i := range inputs {
		inputs[i] = float32(i)*0.031 - 15
	}

	gotT := make([]float32, len(inputs))
	gotL := make([]float32, len(inputs))
	TanhLogCoshFloat32(gotT, gotL, inputs)

	wantT, wantL := goHyp(inputs)

	for i, x := range inputs {
		if d := ulpDiff32(gotT[i], wantT[i]); d > hypTolTanh {
			t.Fatalf("tanh(%v): dispatch %v, go %v, %d ulp apart", x, gotT[i], wantT[i], d)
		}

		if d := ulpDiff32(gotL[i], wantL[i]); d > hypTolLogCosh {
			t.Fatalf("logCosh(%v): dispatch %v, go %v, %d ulp apart", x, gotL[i], wantL[i], d)
		}
	}
}

// TestHypNEONFullDomainDifferential compares the assembly kernel against the
// pure-Go kernel on every one of the 2^32 float32 bit patterns.
//
// It takes a few minutes, so it is opt-in: set ALGO_APPROX_EXHAUSTIVE=1. The
// bounded sweeps above run by default and cover the same shapes; this one is
// the proof that nothing hides between the sampled points.
func TestHypNEONFullDomainDifferential(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	if os.Getenv("ALGO_APPROX_EXHAUSTIVE") != "1" {
		t.Skip("set ALGO_APPROX_EXHAUSTIVE=1 to sweep all 2^32 float32 inputs")
	}

	const block = 1 << 12

	var (
		buf          = make([]float32, block)
		gotT, gotL   = make([]float32, block), make([]float32, block)
		wantT, wantL = make([]float32, block), make([]float32, block)
		tanhStats    = ulpStats{name: "tanh", tol: hypTolTanh, worst: 0, exact: 0}
		logCoshStats = ulpStats{name: "logCosh", tol: hypTolLogCosh, worst: 0, exact: 0}
		total        uint64
	)

	for base := uint64(0); base < 1<<32; base += block {
		for j := range block {
			buf[j] = math.Float32frombits(uint32(base + uint64(j))) //nolint:gosec // base+j < 2^32 by the loop bound.
		}

		tanhLogCoshBatch32NEON(gotT, gotL, buf)
		tanhLogCoshBatch32Go(wantT, wantL, buf)

		for j := range block {
			if math.IsNaN(float64(buf[j])) {
				if !math.IsNaN(float64(gotT[j])) || !math.IsNaN(float64(gotL[j])) {
					t.Fatalf("x=NaN %#08x gave (%v, %v), want (NaN, NaN)",
						math.Float32bits(buf[j]), gotT[j], gotL[j])
				}

				continue
			}

			total++

			tanhStats.observe(t, buf[j], gotT[j], wantT[j])
			logCoshStats.observe(t, buf[j], gotL[j], wantL[j])
		}
	}

	t.Logf("full domain over %d inputs:", total)
	tanhStats.report(t, total)
	logCoshStats.report(t, total)
}

// ulpStats accumulates the asm-vs-Go comparison for one output across the full
// domain sweep, failing the moment any element exceeds its tolerance.
type ulpStats struct {
	name  string
	tol   int64
	worst int64
	exact uint64
}

func (s *ulpStats) observe(tb testing.TB, x, got, want float32) {
	tb.Helper()

	if math.Float32bits(got) == math.Float32bits(want) {
		s.exact++

		return
	}

	d := ulpDiff32(got, want)
	if d > s.tol {
		tb.Fatalf("%s(%v): asm %v, go %v, %d ulp apart (tol %d)", s.name, x, got, want, d, s.tol)
	}

	s.worst = max(s.worst, d)
}

func (s *ulpStats) report(tb testing.TB, total uint64) {
	tb.Helper()

	tb.Logf("  %-8s max %d ulp, %d (%.2f%%) bit-identical",
		s.name+":", s.worst, s.exact, 100*float64(s.exact)/float64(total))
}
