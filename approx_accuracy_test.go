package approx_test

import (
	"math"
	"testing"

	approx "github.com/meko-christian/algo-approx"
	"github.com/meko-christian/algo-approx/internal/reference"
)

func TestAccuracy_Balanced_MinimumDigits(t *testing.T) {
	t.Parallel()
	// This is a coarse end-to-end check to ensure approximations are in the
	// right ballpark and remain stable across refactors.
	const minDigits = 2.0

	sqrtSamples := make([]float64, 0, 2000)
	for i := range 2001 {
		// Log-spaced across [1e-12, 1e12]
		exp := -12.0 + 24.0*float64(i)/2000.0
		sqrtSamples = append(sqrtSamples, math.Pow(10, exp))
	}

	mSqrt := reference.MeasureAccuracy[float64](sqrtSamples,
		reference.Sqrt[float64],
		func(x float64) float64 { return float64(approx.FastSqrtPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("sqrt balanced: %+v", mSqrt)

	if mSqrt.DecimalDigits < minDigits {
		t.Fatalf("sqrt balanced too inaccurate: digits=%g metrics=%+v", mSqrt.DecimalDigits, mSqrt)
	}

	mInvSqrt := reference.MeasureAccuracy[float64](sqrtSamples,
		reference.InvSqrt[float64],
		func(x float64) float64 { return float64(approx.FastInvSqrtPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("invsqrt balanced: %+v", mInvSqrt)

	if mInvSqrt.DecimalDigits < minDigits {
		t.Fatalf("invsqrt balanced too inaccurate: digits=%g metrics=%+v", mInvSqrt.DecimalDigits, mInvSqrt)
	}

	logSamples := make([]float64, 0, 2000)
	for i := range 2001 {
		exp := -12.0 + 18.0*float64(i)/2000.0 // [1e-12, 1e6]
		logSamples = append(logSamples, math.Pow(10, exp))
	}

	mLog := reference.MeasureAccuracy[float64](logSamples,
		reference.Log[float64],
		func(x float64) float64 { return float64(approx.FastLogPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("log balanced: %+v", mLog)

	if mLog.DecimalDigits < minDigits {
		t.Fatalf("log balanced too inaccurate: digits=%g metrics=%+v", mLog.DecimalDigits, mLog)
	}

	expSamples := make([]float64, 0, 2001)
	for i := range 2001 {
		x := -10.0 + 20.0*float64(i)/2000.0
		expSamples = append(expSamples, x)
	}

	mExp := reference.MeasureAccuracy[float64](expSamples,
		reference.Exp[float64],
		func(x float64) float64 { return float64(approx.FastExpPrec(x, approx.PrecisionBalanced)) },
	)
	t.Logf("exp balanced: %+v", mExp)

	if mExp.DecimalDigits < minDigits {
		t.Fatalf("exp balanced too inaccurate: digits=%g metrics=%+v", mExp.DecimalDigits, mExp)
	}
}
