package approx_test

import (
	"math"
	"testing"

	approx "github.com/cwbudde/algo-approx"
)

// Tests for the public batch (slice) API.
//
// The organising idea is differential: every batch function has a scalar twin
// in this same package, so the batch path is never checked against a hand-typed
// expected value. It is checked against the scalar function, element by
// element, and the only interesting question is how tight that comparison is
// allowed to be.
//
//   - float64: **bit-for-bit**. The batch loops call the identical kernels, so
//     any difference at all is a bug, not a rounding artefact. There is no
//     tolerance here on purpose — a tolerance would hide exactly the class of
//     mistake (a copied-and-drifted branch point, a differently ordered
//     expression) that the fused float64 kernel is most exposed to.
//   - float32: an ulp bound, because the two paths are genuinely different
//     algorithms. The scalar API widens to float64 and rounds once; the batch
//     path is a float32-native minimax kernel, optionally AVX2. Their bounds
//     against the truth are asymmetric (see ACCURACY.md), and the agreement
//     bound is the sum of the two.

// batchRamp returns n samples spread over [lo, hi], deliberately including the
// branch point at 0.625 and its negative.
// The divisor is max(n-1, 1) rather than n-1 so that n == 1 yields lo instead
// of NaN. TestBatchTailLengths sweeps every length from 0 to 32, and at n == 1
// a 0/0 divisor would have turned a tail-length case into a NaN-propagation
// case without anything failing to say so.
func batchRamp(n int, lo, hi float64) []float64 {
	out := make([]float64, n)
	for i := range n {
		out[i] = lo + (hi-lo)*float64(i)/float64(max(n-1, 1))
	}

	return out
}

func batchRamp32(n int, lo, hi float32) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = lo + (hi-lo)*float32(i)/float32(max(n-1, 1))
	}

	return out
}

func TestFastExpBatch64_MatchesScalarBitExactly(t *testing.T) {
	t.Parallel()

	src := batchRamp(4001, -20, 20)
	dst := make([]float64, len(src))

	approx.FastExpBatch64(dst, src)

	for i, x := range src {
		if want := approx.FastExp64(x); dst[i] != want {
			t.Fatalf("FastExpBatch64 at x=%v: got %v (%#x), want %v (%#x)",
				x, dst[i], math.Float64bits(dst[i]), want, math.Float64bits(want))
		}
	}
}

func TestFastTanhLogCoshBatch64_MatchesScalarBitExactly(t *testing.T) {
	t.Parallel()

	// The ramp straddles both branch points that matter: tanhBranch at 0.625
	// and the exact-saturation point at 19.0625.
	src := batchRamp(8001, -25, 25)
	src = append(src, 0, math.Copysign(0, -1), 0.625, -0.625, 19.0625, -19.0625,
		math.Inf(1), math.Inf(-1), math.NaN(), 1e300, -1e300)

	dstTanh := make([]float64, len(src))
	dstLogCosh := make([]float64, len(src))

	approx.FastTanhLogCoshBatch64(dstTanh, dstLogCosh, src)

	for i, x := range src {
		wantTanh, wantLogCosh := approx.FastTanh64(x), approx.FastLogCosh64(x)

		if !sameFloat64(dstTanh[i], wantTanh) {
			t.Fatalf("batch tanh at x=%v: got %v (%#x), want %v (%#x)",
				x, dstTanh[i], math.Float64bits(dstTanh[i]), wantTanh, math.Float64bits(wantTanh))
		}

		if !sameFloat64(dstLogCosh[i], wantLogCosh) {
			t.Fatalf("batch logCosh at x=%v: got %v (%#x), want %v (%#x)",
				x, dstLogCosh[i], math.Float64bits(dstLogCosh[i]), wantLogCosh, math.Float64bits(wantLogCosh))
		}
	}
}

// sameFloat64 compares bit patterns, with NaN equal to NaN and -0 distinct from
// +0. Plain == would make the -0 cases vacuous, which is where the sign
// reattachment could silently break.
func sameFloat64(got, want float64) bool {
	if math.IsNaN(got) && math.IsNaN(want) {
		return true
	}

	return math.Float64bits(got) == math.Float64bits(want)
}

// The float32 batch path is a different algorithm from the float32 scalar path,
// so this pins an agreement bound rather than bit-equality.
//
// Derivation, from ACCURACY.md: over this sweep FastExp32 measures 38 ulp
// against a float64 reference rounded to float32, while the batch kernel
// measures 1. Two implementations that are 38 and 1 ulp from the truth can be
// 39 ulp apart while both are correct, so 39 is the bound and it is a
// derivation rather than slack. The batch path is the accurate one.
func TestFastExpBatch32_AgreesWithScalarWithinUlpBound(t *testing.T) {
	t.Parallel()

	const bound = 39

	src := batchRamp32(4001, -10, 10)
	dst := make([]float32, len(src))

	approx.FastExpBatch32(dst, src)

	worst, worstAt := 0.0, float32(0)

	for i, x := range src {
		if d := ulpDiff32(dst[i], approx.FastExp32(x)); d > worst {
			worst, worstAt = d, x
		}
	}

	if worst > bound {
		t.Fatalf("FastExpBatch32 vs FastExp32: %v ulp at x=%v, bound %d", worst, worstAt, bound)
	}

	t.Logf("FastExpBatch32 vs FastExp32: max %v ulp at x=%v (bound %d)", worst, worstAt, bound)
}

// Bounds derived as above: FastTanh32 measures 1 ulp and the batch kernel 1, so
// 2; FastLogCosh32 measures 0 and the batch kernel 2 (4 just above the branch
// point at |x| = 0.625), so 4.
func TestFastTanhLogCoshBatch32_AgreesWithScalarWithinUlpBound(t *testing.T) {
	t.Parallel()

	const (
		boundTanh    = 2
		boundLogCosh = 4
	)

	src := batchRamp32(4001, -12, 12)
	dstTanh := make([]float32, len(src))
	dstLogCosh := make([]float32, len(src))

	approx.FastTanhLogCoshBatch32(dstTanh, dstLogCosh, src)

	worstTanh, worstLogCosh := 0.0, 0.0
	atTanh, atLogCosh := float32(0), float32(0)

	for i, x := range src {
		if d := ulpDiff32(dstTanh[i], approx.FastTanh32(x)); d > worstTanh {
			worstTanh, atTanh = d, x
		}

		if d := ulpDiff32(dstLogCosh[i], approx.FastLogCosh32(x)); d > worstLogCosh {
			worstLogCosh, atLogCosh = d, x
		}
	}

	if worstTanh > boundTanh {
		t.Fatalf("batch tanh vs FastTanh32: %v ulp at x=%v, bound %d", worstTanh, atTanh, boundTanh)
	}

	if worstLogCosh > boundLogCosh {
		t.Fatalf("batch logCosh vs FastLogCosh32: %v ulp at x=%v, bound %d",
			worstLogCosh, atLogCosh, boundLogCosh)
	}

	t.Logf("batch tanh max %v ulp at x=%v; batch logCosh max %v ulp at x=%v",
		worstTanh, atTanh, worstLogCosh, atLogCosh)
}

// Odd symmetry survives the batch path bit-for-bit, in both widths. This is one
// of the guarantees the scalar API advertises; a batch kernel that computed the
// magnitude from x rather than from |x| would break it while still passing any
// tolerance-based accuracy test.
func TestBatchTanh_OddSymmetryIsBitExact(t *testing.T) {
	t.Parallel()

	t.Run("float64", func(t *testing.T) {
		t.Parallel()

		pos := batchRamp(2001, 0, 25)
		neg := make([]float64, len(pos))

		for i, x := range pos {
			neg[i] = -x
		}

		tanhPos := make([]float64, len(pos))
		tanhNeg := make([]float64, len(pos))
		scratch := make([]float64, len(pos))

		approx.FastTanhLogCoshBatch64(tanhPos, scratch, pos)
		approx.FastTanhLogCoshBatch64(tanhNeg, scratch, neg)

		for i, x := range pos {
			if math.Float64bits(tanhNeg[i]) != math.Float64bits(-tanhPos[i]) {
				t.Fatalf("odd symmetry broken at x=%v: %v vs %v", x, tanhPos[i], tanhNeg[i])
			}
		}
	})

	t.Run("float32", func(t *testing.T) {
		t.Parallel()

		pos := batchRamp32(2001, 0, 25)
		neg := make([]float32, len(pos))

		for i, x := range pos {
			neg[i] = -x
		}

		tanhPos := make([]float32, len(pos))
		tanhNeg := make([]float32, len(pos))
		scratch := make([]float32, len(pos))

		approx.FastTanhLogCoshBatch32(tanhPos, scratch, pos)
		approx.FastTanhLogCoshBatch32(tanhNeg, scratch, neg)

		for i, x := range pos {
			if math.Float32bits(tanhNeg[i]) != math.Float32bits(-tanhPos[i]) {
				t.Fatalf("odd symmetry broken at x=%v: %v vs %v", x, tanhPos[i], tanhNeg[i])
			}
		}
	})
}

// The derivative identity, checked through the public batch entry points rather
// than through the kernels. tanh is exactly d/dx log(cosh x), and the batch pair
// shares one u = exp(-2|x|) and one branch point precisely so that it survives.
//
// The tolerance is dominated by the central difference itself: at h = 1e-3 the
// O(h^2) truncation of the difference quotient contributes ~1.3e-7 on its own,
// which is most of the 1.7e-6 allowed here.
func TestBatchTanhLogCosh_DerivativeIdentity(t *testing.T) {
	t.Parallel()

	const (
		step = 1e-3
		tol  = 1.7e-6
	)

	samples := batchRamp(4001, -12, 12)

	plus := make([]float64, len(samples))
	minus := make([]float64, len(samples))

	for i, x := range samples {
		plus[i] = x + step
		minus[i] = x - step
	}

	tanhAt := make([]float64, len(samples))
	scratch := make([]float64, len(samples))
	lcPlus := make([]float64, len(samples))
	lcMinus := make([]float64, len(samples))

	approx.FastTanhLogCoshBatch64(tanhAt, scratch, samples)
	approx.FastTanhLogCoshBatch64(scratch, lcPlus, plus)
	approx.FastTanhLogCoshBatch64(scratch, lcMinus, minus)

	for i, x := range samples {
		deriv := (lcPlus[i] - lcMinus[i]) / (2 * step)
		if diff := math.Abs(deriv - tanhAt[i]); diff > tol {
			t.Fatalf("d/dx logCosh != tanh at x=%v: %v vs %v (diff %v)", x, deriv, tanhAt[i], diff)
		}
	}
}

// Every destination is checked separately. A single "some destination is short"
// test would pass while one of the three length checks was missing.
func TestBatchPanicsOnShortDestination(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		run  func()
	}{
		{"FastExpBatch32/dst", func() { approx.FastExpBatch32(make([]float32, 3), make([]float32, 4)) }},
		{"FastExpBatch64/dst", func() { approx.FastExpBatch64(make([]float64, 3), make([]float64, 4)) }},
		{"FastTanhLogCoshBatch32/dstTanh", func() {
			approx.FastTanhLogCoshBatch32(make([]float32, 3), make([]float32, 4), make([]float32, 4))
		}},
		{"FastTanhLogCoshBatch32/dstLogCosh", func() {
			approx.FastTanhLogCoshBatch32(make([]float32, 4), make([]float32, 3), make([]float32, 4))
		}},
		{"FastTanhLogCoshBatch64/dstTanh", func() {
			approx.FastTanhLogCoshBatch64(make([]float64, 3), make([]float64, 4), make([]float64, 4))
		}},
		{"FastTanhLogCoshBatch64/dstLogCosh", func() {
			approx.FastTanhLogCoshBatch64(make([]float64, 4), make([]float64, 3), make([]float64, 4))
		}},
		{"FastExpBatch32/nil dst", func() { approx.FastExpBatch32(nil, make([]float32, 1)) }},
		{"FastExpBatch64/nil dst", func() { approx.FastExpBatch64(nil, make([]float64, 1)) }},
	}

	for _, tcase := range cases {
		t.Run(tcase.name, func(t *testing.T) {
			t.Parallel()

			defer func() {
				if recover() == nil {
					t.Fatalf("%s: expected panic on short destination", tcase.name)
				}
			}()

			tcase.run()
		})
	}
}

// requireSame fails on the first element where two slices differ. Split out so
// that the in-place and long-destination tests stay under the cyclomatic
// complexity limit rather than growing a nest of loops and ifs.
func requireSame[T float32 | float64](tb testing.TB, what string, got, want []T) {
	tb.Helper()

	for i := range want {
		if got[i] != want[i] {
			tb.Fatalf("%s differs at %d: %v vs %v", what, i, got[i], want[i])
		}
	}
}

// requireTail fails if anything past n was written.
func requireTail[T float32 | float64](tb testing.TB, what string, dst []T, n int, sentinel T) {
	tb.Helper()

	for i := n; i < len(dst); i++ {
		if dst[i] != sentinel {
			tb.Fatalf("%s wrote past len(src) at %d: %v", what, i, dst[i])
		}
	}
}

// In-place is a supported mode (dst == src), and it is the one aliasing case
// the SIMD kernels are allowed to see. A kernel that read a whole vector after
// writing part of it would fail here and nowhere else.
//
// The fused functions are exercised with dstTanh aliasing src, which is the
// awkward case: the kernel must have consumed src before overwriting it, and it
// still owes a correct second output afterwards.
func TestBatchInPlace(t *testing.T) {
	t.Parallel()

	t.Run("FastExpBatch64", func(t *testing.T) {
		t.Parallel()

		src := batchRamp(1000, -10, 10)
		want := make([]float64, len(src))

		approx.FastExpBatch64(want, src)
		approx.FastExpBatch64(src, src)
		requireSame(t, "FastExpBatch64 in-place", src, want)
	})

	t.Run("FastExpBatch32", func(t *testing.T) {
		t.Parallel()

		src := batchRamp32(1000, -10, 10)
		want := make([]float32, len(src))

		approx.FastExpBatch32(want, src)
		approx.FastExpBatch32(src, src)
		requireSame(t, "FastExpBatch32 in-place", src, want)
	})

	t.Run("FastTanhLogCoshBatch64", func(t *testing.T) {
		t.Parallel()

		src := batchRamp(1000, -12, 12)
		wantTanh := make([]float64, len(src))
		wantLogCosh := make([]float64, len(src))
		gotLogCosh := make([]float64, len(src))

		approx.FastTanhLogCoshBatch64(wantTanh, wantLogCosh, src)
		approx.FastTanhLogCoshBatch64(src, gotLogCosh, src)
		requireSame(t, "batch64 in-place tanh", src, wantTanh)
		requireSame(t, "batch64 in-place logCosh", gotLogCosh, wantLogCosh)
	})

	t.Run("FastTanhLogCoshBatch32", func(t *testing.T) {
		t.Parallel()

		src := batchRamp32(1000, -12, 12)
		wantTanh := make([]float32, len(src))
		wantLogCosh := make([]float32, len(src))
		gotLogCosh := make([]float32, len(src))

		approx.FastTanhLogCoshBatch32(wantTanh, wantLogCosh, src)
		approx.FastTanhLogCoshBatch32(src, gotLogCosh, src)
		requireSame(t, "batch32 in-place tanh", src, wantTanh)
		requireSame(t, "batch32 in-place logCosh", gotLogCosh, wantLogCosh)
	})
}

// len(dst) > len(src): exactly len(src) elements are written and the tail is
// left alone. The sentinel is what pins "left alone" - without it the test
// would pass on a kernel that zeroed the whole destination.
func TestBatchLongDestinationLeavesTailUntouched(t *testing.T) {
	t.Parallel()

	const (
		n        = 100
		extra    = 37
		sentinel = -12345.0
	)

	t.Run("float64", func(t *testing.T) {
		t.Parallel()

		src := batchRamp(n, -5, 5)
		dst := make([]float64, n+extra)
		tanh := make([]float64, n+extra)
		logcosh := make([]float64, n+extra)

		for i := range dst {
			dst[i] = sentinel
			tanh[i] = sentinel
			logcosh[i] = sentinel
		}

		approx.FastExpBatch64(dst, src)
		requireTail(t, "FastExpBatch64", dst, n, sentinel)

		want := make([]float64, n)
		for i, x := range src {
			want[i] = approx.FastExp64(x)
		}

		requireSame(t, "FastExpBatch64 head", dst[:n], want)

		// Both destinations get their own buffer and both tails are checked:
		// the contract is per-destination, so asserting only dstTanh would let
		// a kernel that overran dstLogCosh through.
		approx.FastTanhLogCoshBatch64(tanh, logcosh, src)
		requireTail(t, "FastTanhLogCoshBatch64/dstTanh", tanh, n, sentinel)
		requireTail(t, "FastTanhLogCoshBatch64/dstLogCosh", logcosh, n, sentinel)
	})

	t.Run("float32", func(t *testing.T) {
		t.Parallel()

		src := batchRamp32(n, -5, 5)
		dst := make([]float32, n+extra)
		tanh := make([]float32, n+extra)
		logcosh := make([]float32, n+extra)

		for i := range dst {
			dst[i] = sentinel
			tanh[i] = sentinel
			logcosh[i] = sentinel
		}

		approx.FastExpBatch32(dst, src)
		requireTail(t, "FastExpBatch32", dst, n, sentinel)

		// See the float64 case: both tails are checked, because the masked
		// tail store in the AVX2 kernel is per-destination and a bug in one
		// would not show up in the other.
		approx.FastTanhLogCoshBatch32(tanh, logcosh, src)
		requireTail(t, "FastTanhLogCoshBatch32/dstTanh", tanh, n, sentinel)
		requireTail(t, "FastTanhLogCoshBatch32/dstLogCosh", logcosh, n, sentinel)
	})
}

// Empty and nil sources are a no-op, not a panic: len(src) == 0 elements are
// processed and a nil destination is trivially "at least as long".
func TestBatchEmptyAndNil(t *testing.T) {
	t.Parallel()

	approx.FastExpBatch32(nil, nil)
	approx.FastExpBatch64(nil, nil)
	approx.FastTanhLogCoshBatch32(nil, nil, nil)
	approx.FastTanhLogCoshBatch64(nil, nil, nil)

	approx.FastExpBatch32([]float32{}, []float32{})
	approx.FastExpBatch64([]float64{}, []float64{})
	approx.FastTanhLogCoshBatch32([]float32{}, []float32{}, []float32{})
	approx.FastTanhLogCoshBatch64([]float64{}, []float64{}, []float64{})

	// A non-empty destination with an empty source must not be touched.
	dst := []float64{1, 2, 3}
	approx.FastExpBatch64(dst, nil)

	if dst[0] != 1 || dst[1] != 2 || dst[2] != 3 {
		t.Fatalf("empty src modified dst: %v", dst)
	}
}

// Non-multiple-of-eight lengths exercise the assembly kernels' masked tail. The
// vector body handles eight elements at a time; every remainder must still come
// out equal to the scalar answer.
func TestBatchTailLengths(t *testing.T) {
	t.Parallel()

	for n := range 33 {
		src := batchRamp32(max(n, 1), -6, 6)[:n]
		dst := make([]float32, n)
		tanh := make([]float32, n)

		approx.FastExpBatch32(dst, src)
		approx.FastTanhLogCoshBatch32(tanh, make([]float32, n), src)

		for i, x := range src {
			if ulpDiff32(dst[i], approx.FastExp32(x)) > 39 {
				t.Fatalf("n=%d: exp tail wrong at %d (x=%v)", n, i, x)
			}

			if ulpDiff32(tanh[i], approx.FastTanh32(x)) > 2 {
				t.Fatalf("n=%d: tanh tail wrong at %d (x=%v)", n, i, x)
			}
		}
	}
}
