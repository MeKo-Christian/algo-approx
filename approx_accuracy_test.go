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

	sqrtSamples := logSpaced(2001, -12, 12)

	mSqrt := reference.MeasureAccuracy[float64](
		sqrtSamples,
		reference.Sqrt[float64],
		func(x float64) float64 { return float64(approx.FastSqrtPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("sqrt balanced: %+v", mSqrt)

	if mSqrt.DecimalDigits < minDigits {
		t.Fatalf("sqrt balanced too inaccurate: digits=%g metrics=%+v", mSqrt.DecimalDigits, mSqrt)
	}

	mInvSqrt := reference.MeasureAccuracy[float64](
		sqrtSamples,
		reference.InvSqrt[float64],
		func(x float64) float64 { return float64(approx.FastInvSqrtPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("invsqrt balanced: %+v", mInvSqrt)

	if mInvSqrt.DecimalDigits < minDigits {
		t.Fatalf("invsqrt balanced too inaccurate: digits=%g metrics=%+v", mInvSqrt.DecimalDigits, mInvSqrt)
	}

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

// TestAccuracy_Balanced_HyperbolicAndRecip covers the functions added after the
// Phase 1 MVP. Unlike the block above these carry real targets rather than a
// coarse floor, because they were specified with one.
func TestAccuracy_Balanced_HyperbolicAndRecip(t *testing.T) {
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

	recipSamples := logSpaced(4001, -150, 150)

	mRecip := reference.MeasureAccuracy[float64](
		recipSamples,
		reference.Recip[float64],
		func(x float64) float64 { return approx.FastRecipPrec(x, approx.PrecisionBalanced) },
	)
	t.Logf("recip balanced: %+v", mRecip)

	if mRecip.MaxRelError > 1e-15 {
		t.Fatalf("recip balanced max rel error %g exceeds 1e-15: %+v", mRecip.MaxRelError, mRecip)
	}
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
