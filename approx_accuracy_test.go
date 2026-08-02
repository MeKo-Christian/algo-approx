package approx_test

import (
	"math"
	"testing"

	approx "github.com/cwbudde/algo-approx"
	"github.com/cwbudde/algo-approx/internal/reference"
)

func TestAccuracy_Balanced_MinimumDigits(t *testing.T) {
	t.Parallel()
	// This is a coarse end-to-end check to ensure approximations are in the
	// right ballpark and remain stable across refactors.
	const minDigits = 2.0

	logSamples := logSpaced(2001, -12, 6)

	mLog := reference.MeasureAccuracy[float64](
		logSamples,
		reference.Log[float64],
		func(x float64) float64 { return float64(approx.FastLogPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("log balanced: %+v", mLog)

	if mLog.DecimalDigits < minDigits {
		t.Fatalf("log balanced too inaccurate: digits=%g metrics=%+v", mLog.DecimalDigits, mLog)
	}

	expSamples := linSpaced(2001, -10, 10)

	mExp := reference.MeasureAccuracy[float64](
		expSamples,
		reference.Exp[float64],
		func(x float64) float64 { return float64(approx.FastExpPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("exp balanced: %+v", mExp)

	if mExp.DecimalDigits < minDigits {
		t.Fatalf("exp balanced too inaccurate: digits=%g metrics=%+v", mExp.DecimalDigits, mExp)
	}
}

// TestAccuracy_Balanced_Hyperbolic covers the functions added after the
// Phase 1 MVP. Unlike the block above these carry real targets rather than a
// coarse floor, because they were specified with one.
func TestAccuracy_Balanced_Hyperbolic(t *testing.T) {
	t.Parallel()

	tanhSamples := linSpaced(4001, -20, 20)

	mTanh := reference.MeasureAccuracy[float64](
		tanhSamples,
		reference.Tanh[float64],
		func(x float64) float64 { return approx.FastTanhPrec(x, approx.PrecisionBalanced) },
	)
	t.Logf("tanh balanced: %+v", mTanh)

	if mTanh.MaxAbsError > 1e-7 {
		t.Fatalf("tanh balanced max abs error %g exceeds 1e-7: %+v", mTanh.MaxAbsError, mTanh)
	}

	logCoshSamples := linSpaced(4001, -12, 12)

	mLogCosh := reference.MeasureAccuracy[float64](
		logCoshSamples,
		reference.LogCosh[float64],
		func(x float64) float64 { return approx.FastLogCoshPrec(x, approx.PrecisionBalanced) },
	)
	t.Logf("logcosh balanced: %+v", mLogCosh)

	if mLogCosh.MaxAbsError > 1e-7 {
		t.Fatalf("logcosh balanced max abs error %g exceeds 1e-7: %+v", mLogCosh.MaxAbsError, mLogCosh)
	}
}

// TestAccuracy32_Balanced_MaxUlp is the float32 counterpart of the two blocks
// above. Thresholds are stated in ulps of float32 rather than as absolute
// errors: a float32 kernel can never do better than half an ulp, and an
// absolute gate silently changes meaning as the output magnitude moves.
//
// The reference is the float64 result rounded to float32, not the float64
// result itself. Comparing against the latter would fold the ~6e-8
// representation gap of float32 into every measurement and report the format's
// error rather than the kernel's.
func TestAccuracy32_Balanced_MaxUlp(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		samples  []float32
		refFn    func(float32) float32
		approxFn func(float32) float32
		maxUlp   float64
	}{
		{
			// measured 38 ulp: the balanced exp kernel is only good to
			// ~3.2e-6 relative in its own right, which is ~27 float32 ulp,
			// so this gate is on the polynomial and not on the format.
			name:     "exp",
			samples:  linSpaced32(4001, -10, 10),
			refFn:    reference.Exp[float32],
			approxFn: func(x float32) float32 { return approx.FastExpPrec(x, approx.PrecisionBalanced) },
			maxUlp:   64,
		},
		{
			// measured 103 ulp. See logSamples32 for why the samples skip
			// the neighbourhood of x = 1.
			name:     "log",
			samples:  logSamples32(),
			refFn:    reference.Log[float32],
			approxFn: func(x float32) float32 { return approx.FastLogPrec(x, approx.PrecisionBalanced) },
			maxUlp:   256,
		},
		{
			// measured 1 ulp: tanh is a thin shim over a float64 kernel that
			// is already good to 2.5e-9, so float32 output is correctly
			// rounded up to a final rounding step.
			name:     "tanh",
			samples:  linSpaced32(4001, -20, 20),
			refFn:    reference.Tanh[float32],
			approxFn: func(x float32) float32 { return approx.FastTanhPrec(x, approx.PrecisionBalanced) },
			maxUlp:   4,
		},
		{
			// measured 0 ulp: bit-identical to the rounded float64 reference
			// over the whole consumer domain, for the same reason as tanh.
			name:     "logcosh",
			samples:  linSpaced32(4001, -12, 12),
			refFn:    reference.LogCosh[float32],
			approxFn: func(x float32) float32 { return approx.FastLogCoshPrec(x, approx.PrecisionBalanced) },
			maxUlp:   2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			metrics := reference.MeasureAccuracy[float32](tt.samples, tt.refFn, tt.approxFn)
			worst := maxUlpDiff32(tt.samples, tt.refFn, tt.approxFn)
			t.Logf("%s float32 balanced: maxUlp=%g %+v", tt.name, worst, metrics)

			if worst > tt.maxUlp {
				t.Fatalf("%s float32 max ulp error %g exceeds %g: %+v", tt.name, worst, tt.maxUlp, metrics)
			}
		})
	}
}

// TestAccuracy32_Balanced_LogNearOne covers the band that the ulp gate above
// deliberately drops. Around x = 1 the output passes through zero, so a
// perfectly good kernel is millions of ulps out and only the absolute error
// says anything; this is the float32 restatement of the note in ACCURACY.md.
func TestAccuracy32_Balanced_LogNearOne(t *testing.T) {
	t.Parallel()

	// measured: 1.242e-05.
	const maxAbs = 4e-5

	samples := logSpaced32(4001, -1, 1)

	metrics := reference.MeasureAccuracy[float32](
		samples,
		reference.Log[float32],
		func(x float32) float32 { return approx.FastLogPrec(x, approx.PrecisionBalanced) },
	)
	t.Logf("log float32 balanced near 1: %+v", metrics)

	if metrics.MaxAbsError > maxAbs {
		t.Fatalf("log float32 max abs error %g exceeds %g: %+v", metrics.MaxAbsError, maxAbs, metrics)
	}
}

// TestUlpDiff32 exercises the float32 ulp helper itself, so that a regression
// in the measuring stick cannot quietly pass the gates above.
func TestUlpDiff32(t *testing.T) {
	t.Parallel()

	const one float32 = 1

	next := math.Float32frombits(math.Float32bits(one) + 1)
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())

	tests := []struct {
		name       string
		got, want  float32
		expectULPs float64
	}{
		{name: "equal", got: one, want: one, expectULPs: 0},
		{name: "adjacent above", got: next, want: one, expectULPs: 1},
		{name: "adjacent below", got: one, want: next, expectULPs: 1},
		{name: "signed zeros", got: 0, want: float32(math.Copysign(0, -1)), expectULPs: 0},
		{name: "zero to smallest", got: math.SmallestNonzeroFloat32, want: 0, expectULPs: 1},
		{name: "across zero", got: math.SmallestNonzeroFloat32, want: -math.SmallestNonzeroFloat32, expectULPs: 2},
		{name: "same infinity", got: inf, want: inf, expectULPs: 0},
		{name: "opposite infinity", got: inf, want: -inf, expectULPs: math.Inf(1)},
		{name: "infinity vs finite", got: inf, want: math.MaxFloat32, expectULPs: math.Inf(1)},
		{name: "both NaN", got: nan, want: nan, expectULPs: 0},
		{name: "NaN vs finite", got: nan, want: one, expectULPs: math.Inf(1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := ulpDiff32(tt.got, tt.want); got != tt.expectULPs {
				t.Fatalf("ulpDiff32(%v, %v) = %g, want %g", tt.got, tt.want, got, tt.expectULPs)
			}
		})
	}
}

// TestUlpDiff64 is TestUlpDiff32 for the float64 helper.
func TestUlpDiff64(t *testing.T) {
	t.Parallel()

	const one float64 = 1

	next := math.Nextafter(one, 2)

	tests := []struct {
		name       string
		got, want  float64
		expectULPs float64
	}{
		{name: "equal", got: one, want: one, expectULPs: 0},
		{name: "adjacent above", got: next, want: one, expectULPs: 1},
		{name: "adjacent below", got: one, want: next, expectULPs: 1},
		{name: "signed zeros", got: 0, want: math.Copysign(0, -1), expectULPs: 0},
		{name: "zero to smallest", got: math.SmallestNonzeroFloat64, want: 0, expectULPs: 1},
		{name: "across zero", got: math.SmallestNonzeroFloat64, want: -math.SmallestNonzeroFloat64, expectULPs: 2},
		{name: "same infinity", got: math.Inf(1), want: math.Inf(1), expectULPs: 0},
		{name: "opposite infinity", got: math.Inf(1), want: math.Inf(-1), expectULPs: math.Inf(1)},
		{name: "infinity vs finite", got: math.Inf(1), want: math.MaxFloat64, expectULPs: math.Inf(1)},
		{name: "both NaN", got: math.NaN(), want: math.NaN(), expectULPs: 0},
		{name: "NaN vs finite", got: math.NaN(), want: one, expectULPs: math.Inf(1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := ulpDiff64(tt.got, tt.want); got != tt.expectULPs {
				t.Fatalf("ulpDiff64(%v, %v) = %g, want %g", tt.got, tt.want, got, tt.expectULPs)
			}
		})
	}
}

// ulpOrder32 maps a float32 onto a monotonically increasing integer line, so
// that the difference of two orders counts the representable float32 values
// between them. Both zeros map to 0.
func ulpOrder32(x float32) int64 {
	const signMask uint32 = 1 << 31

	bits := math.Float32bits(x)
	if bits&signMask != 0 {
		return -int64(bits &^ signMask)
	}

	return int64(bits)
}

// ulpOrder64 is ulpOrder32 for float64.
func ulpOrder64(x float64) int64 {
	const signMask uint64 = 1 << 63

	bits := math.Float64bits(x)

	// Masking the sign bit leaves at most 2^63-1, so both conversions fit.
	if bits&signMask != 0 {
		return -int64(bits &^ signMask) //nolint:gosec // sign bit masked off
	}

	return int64(bits) //nolint:gosec // sign bit masked off
}

// ulpDiff32 reports the distance between got and want in units in the last
// place of float32.
//
// Equal values are zero apart, including +0 against -0 and two infinities of
// the same sign. Any other case involving an infinity or a NaN is reported as
// +Inf, so that such a mismatch can never be averaged away or pass a gate.
func ulpDiff32(got, want float32) float64 {
	gotF, wantF := float64(got), float64(want)

	if math.IsNaN(gotF) || math.IsNaN(wantF) {
		if math.IsNaN(gotF) && math.IsNaN(wantF) {
			return 0
		}

		return math.Inf(1)
	}

	if math.IsInf(gotF, 0) || math.IsInf(wantF, 0) {
		if got == want {
			return 0
		}

		return math.Inf(1)
	}

	return float64(absInt64(ulpOrder32(got) - ulpOrder32(want)))
}

// ulpDiff64 is ulpDiff32 for float64.
func ulpDiff64(got, want float64) float64 {
	if math.IsNaN(got) || math.IsNaN(want) {
		if math.IsNaN(got) && math.IsNaN(want) {
			return 0
		}

		return math.Inf(1)
	}

	if math.IsInf(got, 0) || math.IsInf(want, 0) {
		if got == want {
			return 0
		}

		return math.Inf(1)
	}

	return float64(absInt64(ulpOrder64(got) - ulpOrder64(want)))
}

func absInt64(v int64) int64 {
	if v < 0 {
		return -v
	}

	return v
}

// maxUlpDiff32 returns the largest ulp distance between approxFn and refFn over
// samples.
func maxUlpDiff32(samples []float32, refFn, approxFn func(float32) float32) float64 {
	var worst float64

	for _, x := range samples {
		if d := ulpDiff32(approxFn(x), refFn(x)); d > worst {
			worst = d
		}
	}

	return worst
}

// linSpaced32 is linSpaced with float32 samples. The grid is built in float64
// and rounded once, so the sample points stay independent of float32 rounding.
func linSpaced32(n int, lo, hi float64) []float32 {
	src := linSpaced(n, lo, hi)
	out := make([]float32, len(src))

	for i, v := range src {
		out[i] = float32(v)
	}

	return out
}

// logSamples32 returns the float32 log sweep with the zero crossing removed:
// the samples span [10^-12, 10^6] but keep only those with |ln x| >= 1.
//
// Near x = 1 the output passes through zero, where an ulp count is unbounded
// for any nonzero absolute error and measures the crossing rather than the
// kernel. That band is gated on absolute error instead, by
// TestAccuracy32_Balanced_LogNearOne.
func logSamples32() []float32 {
	src := logSpaced32(4001, -12, 6)
	out := make([]float32, 0, len(src))

	for _, x := range src {
		if math.Abs(math.Log(float64(x))) >= 1 {
			out = append(out, x)
		}
	}

	return out
}

// logSpaced32 is logSpaced with float32 samples.
func logSpaced32(n int, loExp, hiExp float64) []float32 {
	src := logSpaced(n, loExp, hiExp)
	out := make([]float32, len(src))

	for i, v := range src {
		out[i] = float32(v)
	}

	return out
}

// linSpaced returns n samples spaced evenly over [lo, hi].
func linSpaced(n int, lo, hi float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = lo + (hi-lo)*float64(i)/float64(n-1)
	}

	return out
}

// logSpaced returns n samples spaced evenly over [10^loExp, 10^hiExp].
func logSpaced(n int, loExp, hiExp float64) []float64 {
	out := linSpaced(n, loExp, hiExp)
	for i, e := range out {
		out[i] = math.Pow(10, e)
	}

	return out
}
