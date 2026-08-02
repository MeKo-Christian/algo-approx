//go:build amd64 && !purego

package simd

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/cwbudde/algo-approx/internal/cpu"
)

// asmHyp runs the assembly kernel over inputs and returns (tanh, logCosh).
func asmHyp(tb testing.TB, inputs []float32) ([]float32, []float32) {
	tb.Helper()

	dstTanh := make([]float32, len(inputs))
	dstLogCosh := make([]float32, len(inputs))

	if !tanhLogCoshBatch32AVX2(dstTanh, dstLogCosh, inputs) {
		tb.Fatal("tanhLogCoshBatch32AVX2 declined")
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

// compareHyp asserts the two kernels agree on every element, and returns the
// largest disagreement it saw for each output.
func compareHyp(tb testing.TB, inputs []float32) (maxTanh, maxLogCosh int64) {
	tb.Helper()

	gotT, gotL := asmHyp(tb, inputs)
	wantT, wantL := goHyp(inputs)

	for i, x := range inputs {
		if math.IsNaN(float64(x)) {
			if !math.IsNaN(float64(gotT[i])) || !math.IsNaN(float64(gotL[i])) {
				tb.Fatalf("x=NaN: asm gave (%v, %v), want (NaN, NaN)", gotT[i], gotL[i])
			}

			continue
		}

		if d := ulpDiff32(gotT[i], wantT[i]); d > hypTolTanh {
			tb.Fatalf("tanh(%v): asm %v, go %v, %d ulp apart (tol %d)", x, gotT[i], wantT[i], d, hypTolTanh)
		} else {
			maxTanh = max(maxTanh, d)
		}

		if d := ulpDiff32(gotL[i], wantL[i]); d > hypTolLogCosh {
			tb.Fatalf("logCosh(%v): asm %v, go %v, %d ulp apart (tol %d)", x, gotL[i], wantL[i], d, hypTolLogCosh)
		} else {
			maxLogCosh = max(maxLogCosh, d)
		}
	}

	return maxTanh, maxLogCosh
}

// How far the assembly and pure-Go kernels may drift from each other.
//
// These are agreement bounds between two IMPLEMENTATIONS, not accuracy bounds.
// Accuracy is pinned separately by TestTanhAccuracy and TestLogCoshAccuracy in
// simd_test.go, which measure against a float64 reference and run against
// whichever kernel dispatch selected.
//
// The two differ at all because the assembly contracts its multiply-adds into
// FMAs and the Go compiler does not: the amd64 baseline has no FMA, so Go
// emits separate MULSS/ADDSS pairs that round twice where the assembly rounds
// once.
//
// Both bounds are exactly twice the corresponding accuracy bound, and that is
// the whole derivation: if each kernel is independently within k ulp of the
// true value, two kernels can sit on opposite sides of it and be 2k apart
// while both are correct. Neither number is slack chosen to make a test pass.
//
// The evidence, measured over 400001 points spanning [0.60, 0.70], which is
// where both worst cases live:
//
//	tanh     asm worst 1 ulp (mean 0.2428) | go worst 1 ulp (mean 0.2471)
//	         asm closer at 4230 points, go closer at 2513, equal at 393258
//	logCosh  asm worst 4 ulp (mean 0.631)  | go worst 4 ulp (mean 0.632)
//	         asm closer at 841 points, go closer at 745
//
// So on both outputs the assembly is fractionally CLOSER to the truth than the
// pure-Go kernel, not further from it. The drift between them is the two
// implementations rounding to different sides, not the vector kernel losing
// precision.
//
// Why log(cosh) needs 4 where tanh needs 1: just above |x| = 0.625 the large
// branch evaluates a - ln2 + log1p(u), where a - ln2 is small and of opposite
// sign to log1p(u). That cancellation amplifies a half-ulp difference in u
// into several ulp of the result, and it is the same cancellation that caps
// logCoshLarge32's own accuracy at 4 ulp there. tanh has no such subtraction.
//
// Tightening either would take an algorithm change on both sides; it is not
// something to fix in the assembly.
const (
	hypTolTanh    int64 = 2
	hypTolLogCosh int64 = 8
)

// TestDispatchSelectsAVX2OnCapableHost is the guard against the whole suite
// passing while the assembly never executes.
//
// Every differential test here calls tanhLogCoshBatch32AVX2 directly, so they
// would keep passing even if expUseAVX2 were false and the exported wrappers
// quietly ran the Go kernel on every call. The symptom of that is an
// "optimisation" that measures no change at all, which is easy to misread as
// the kernel not being worth it rather than the kernel not running.
//
// expUseAVX2 gates both batch kernels; they share it because they share
// EXPBODY and therefore require exactly the same CPU features.
func TestDispatchSelectsAVX2OnCapableHost(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	want := features.HasAVX2 && features.HasFMA && !features.ForceGeneric

	if expUseAVX2 != want {
		t.Fatalf("expUseAVX2 = %v, but HasAVX2=%v HasFMA=%v ForceGeneric=%v imply %v",
			expUseAVX2, features.HasAVX2, features.HasFMA, features.ForceGeneric, want)
	}

	if !want {
		t.Skip("host is not AVX2+FMA capable, so the Go path is correct here")
	}

	t.Log("dispatch resolved to the AVX2 kernel; the assembly in this package is live")
}

// TestHypAVX2MatchesGo is the differential test over a dense sweep that
// straddles the branch point, the saturation region and the subnormal tail.
func TestHypAVX2MatchesGo(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	inputs := make([]float32, 0, 200000)

	// Dense around the 0.625 seam, where both branches are near each other
	// and the blend has to pick correctly.
	for i := range 40000 {
		inputs = append(inputs, 0.6+float32(i)*1e-6)
		inputs = append(inputs, -(0.6 + float32(i)*1e-6))
	}

	// Wide sweep across the whole interesting range.
	for i := range 100000 {
		inputs = append(inputs, float32(i)*0.002-100)
	}

	mt, ml := compareHyp(t, inputs)
	t.Logf("sweep: max %d ulp (tanh), %d ulp (logCosh) over %d inputs", mt, ml, len(inputs))
}

// TestHypAVX2MatchesGoRandom samples the float32 space by bit pattern rather
// than by value, so the exponent range is covered uniformly instead of being
// dominated by the large-magnitude binades a linear sweep favours.
func TestHypAVX2MatchesGoRandom(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	rng := rand.New(rand.NewSource(20260802)) //nolint:gosec // reproducible test vectors.

	inputs := make([]float32, 200000)
	for i := range inputs {
		inputs[i] = math.Float32frombits(rng.Uint32())
	}

	mt, ml := compareHyp(t, inputs)
	t.Logf("random bit patterns: max %d ulp (tanh), %d ulp (logCosh)", mt, ml)
}

// TestHypAVX2Tails checks every tail length 0..16 at every alignment the loop
// can produce. The masked tail is the part of the kernel that no length that
// happens to be a multiple of eight will ever execute.
func TestHypAVX2Tails(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	for n := range 17 {
		inputs := make([]float32, n)
		for i := range inputs {
			inputs[i] = float32(i)*0.37 - 3
		}

		gotT, gotL := asmHyp(t, inputs)
		wantT, wantL := goHyp(inputs)

		for i := range inputs {
			if d := ulpDiff32(gotT[i], wantT[i]); d > hypTolTanh {
				t.Fatalf("n=%d i=%d: tanh asm %v, go %v", n, i, gotT[i], wantT[i])
			}

			if d := ulpDiff32(gotL[i], wantL[i]); d > hypTolLogCosh {
				t.Fatalf("n=%d i=%d: logCosh asm %v, go %v, %d ulp apart", n, i, gotL[i], wantL[i], d)
			}
		}
	}
}

// TestHypAVX2TailDoesNotWritePastEnd pins the masked-store contract. The tail
// block reads and writes a full 32-byte vector's worth of lanes; only the
// first n%8 of them may reach memory.
func TestHypAVX2TailDoesNotWritePastEnd(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

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

		if !tanhLogCoshBatch32AVX2(bufT[:n], bufL[:n], src) {
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

// TestHypAVX2OddSymmetryBitExact pins the structural guarantee that survives
// vectorisation only because the kernel masks off the sign, computes on the
// magnitude and XORs the sign back on at the very end.
//
// A branchless SIMD tanh written as a polynomial in x rather than in |x| would
// pass every accuracy test in this package and still fail this one.
func TestHypAVX2OddSymmetryBitExact(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

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

// TestHypAVX2InPlace covers dst == src, the one aliasing mode the package
// documents as supported. Either destination may alias the source.
func TestHypAVX2InPlace(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	inputs := make([]float32, 133)
	for i := range inputs {
		inputs[i] = float32(i)*0.19 - 12
	}

	wantT, wantL := goHyp(inputs)

	t.Run("tanh aliases src", func(t *testing.T) {
		t.Parallel()

		buf := append([]float32(nil), inputs...)
		other := make([]float32, len(inputs))

		if !tanhLogCoshBatch32AVX2(buf, other, buf) {
			t.Fatal("declined")
		}

		for i := range inputs {
			if ulpDiff32(buf[i], wantT[i]) > hypTolTanh {
				t.Fatalf("i=%d: in-place tanh %v, want %v", i, buf[i], wantT[i])
			}

			if ulpDiff32(other[i], wantL[i]) > hypTolLogCosh {
				t.Fatalf("i=%d: logCosh %v, want %v", i, other[i], wantL[i])
			}
		}
	})

	t.Run("logCosh aliases src", func(t *testing.T) {
		t.Parallel()

		buf := append([]float32(nil), inputs...)
		other := make([]float32, len(inputs))

		if !tanhLogCoshBatch32AVX2(other, buf, buf) {
			t.Fatal("declined")
		}

		for i := range inputs {
			if ulpDiff32(other[i], wantT[i]) > hypTolTanh {
				t.Fatalf("i=%d: tanh %v, want %v", i, other[i], wantT[i])
			}

			if ulpDiff32(buf[i], wantL[i]) > hypTolLogCosh {
				t.Fatalf("i=%d: in-place logCosh %v, want %v", i, buf[i], wantL[i])
			}
		}
	})
}

// TestHypAVX2Empty covers the zero-length early exit, which skips the constant
// broadcasts and jumps straight to VZEROUPPER.
func TestHypAVX2Empty(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	if !tanhLogCoshBatch32AVX2(nil, nil, nil) {
		t.Fatal("declined on empty input")
	}

	if !tanhLogCoshBatch32AVX2([]float32{}, []float32{}, []float32{}) {
		t.Fatal("declined on empty slices")
	}
}

// TestHypAVX2ShortestLengthWins pins that the kernel processes exactly
// min(len(dstTanh), len(dstLogCosh), len(src)) elements. The three-way minimum
// is computed in assembly, so it needs a test that makes each of the three
// arguments the shortest in turn.
func TestHypAVX2ShortestLengthWins(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	const sentinel = -999

	cases := []struct{ nT, nL, nS, want int }{
		{10, 20, 20, 10},
		{20, 10, 20, 10},
		{20, 20, 10, 10},
		{9, 17, 13, 9},
	}

	for _, tc := range cases {
		dstT := make([]float32, tc.nT)
		dstL := make([]float32, tc.nL)

		for i := range dstT {
			dstT[i] = sentinel
		}

		for i := range dstL {
			dstL[i] = sentinel
		}

		src := make([]float32, tc.nS)
		for i := range src {
			src[i] = 1
		}

		tanhLogCoshBatch32AVX2(dstT, dstL, src)

		for i := tc.want; i < len(dstT); i++ {
			if dstT[i] != sentinel {
				t.Fatalf("%+v: tanh index %d was written, expected only %d elements", tc, i, tc.want)
			}
		}

		for i := tc.want; i < len(dstL); i++ {
			if dstL[i] != sentinel {
				t.Fatalf("%+v: logCosh index %d was written, expected only %d elements", tc, i, tc.want)
			}
		}
	}
}

// TestTanhLogCoshFloat32DispatchMatchesGo exercises the exported wrapper, so
// the dispatch wiring is covered and not only the raw kernel.
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

		if d := ulpDiff32(gotL[i], wantL[i]); d > hypTolTanh {
			t.Fatalf("logCosh(%v): dispatch %v, go %v, %d ulp apart", x, gotL[i], wantL[i], d)
		}
	}
}

// TestHypAVX2FullDomainDifferential compares the assembly kernel against the
// pure-Go kernel on every one of the 2^32 float32 bit patterns.
//
// It takes a few minutes, so it is opt-in: set ALGO_APPROX_EXHAUSTIVE=1. The
// bounded sweeps above run by default and cover the same shapes; this one is
// the proof that nothing hides between the sampled points.
func TestHypAVX2FullDomainDifferential(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	if os.Getenv("ALGO_APPROX_EXHAUSTIVE") != "1" {
		t.Skip("set ALGO_APPROX_EXHAUSTIVE=1 to sweep all 2^32 float32 inputs")
	}

	const block = 1 << 12

	var (
		buf                   = make([]float32, block)
		gotT, gotL            = make([]float32, block), make([]float32, block)
		wantT, wantL          = make([]float32, block), make([]float32, block)
		maxT, maxL            int64
		exactT, exactL, total uint64
	)

	for base := uint64(0); base < 1<<32; base += block {
		for j := range block {
			buf[j] = math.Float32frombits(uint32(base + uint64(j))) //nolint:gosec // base+j < 2^32 by the loop bound.
		}

		tanhLogCoshBatch32AVX2(gotT, gotL, buf)
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

			if math.Float32bits(gotT[j]) == math.Float32bits(wantT[j]) {
				exactT++
			} else if d := ulpDiff32(gotT[j], wantT[j]); d > hypTolTanh {
				t.Fatalf("tanh(%v): asm %v, go %v, %d ulp apart", buf[j], gotT[j], wantT[j], d)
			} else {
				maxT = max(maxT, d)
			}

			if math.Float32bits(gotL[j]) == math.Float32bits(wantL[j]) {
				exactL++
			} else if d := ulpDiff32(gotL[j], wantL[j]); d > hypTolTanh {
				t.Fatalf("logCosh(%v): asm %v, go %v, %d ulp apart", buf[j], gotL[j], wantL[j], d)
			} else {
				maxL = max(maxL, d)
			}
		}
	}

	t.Logf("full domain over %d inputs:", total)
	t.Logf("  tanh:    max %d ulp, %d (%.2f%%) bit-identical", maxT, exactT, 100*float64(exactT)/float64(total))
	t.Logf("  logCosh: max %d ulp, %d (%.2f%%) bit-identical", maxL, exactL, 100*float64(exactL)/float64(total))
}
