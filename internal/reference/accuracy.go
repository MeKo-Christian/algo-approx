package reference

import (
	"math"

	approx "github.com/meko-christian/algo-approx"
)

// AccuracyMetrics summarizes error statistics for an approximation.
type AccuracyMetrics struct {
	MaxAbsError   float64
	MaxRelError   float64
	MeanAbsError  float64
	RMSError      float64
	DecimalDigits float64 // -log10(maxRelError)
}

// MeasureAccuracy computes error metrics between approxFn and refFn over samples.
//
// Relative error is computed as |err|/|ref| when ref != 0, otherwise it falls
// back to absolute error.
func MeasureAccuracy[T approx.Float](samples []T, refFn, approxFn func(T) T) AccuracyMetrics {
	if len(samples) == 0 {
		return AccuracyMetrics{DecimalDigits: math.Inf(1)}
	}

	var (
		maxAbs float64
		maxRel float64
		sumAbs float64
		sumSq  float64
	)

	for _, x := range samples {
		ref := float64(refFn(x))
		got := float64(approxFn(x))
		err := got - ref
		absErr := math.Abs(err)

		sumAbs += absErr
		sumSq += err * err
		if absErr > maxAbs {
			maxAbs = absErr
		}

		den := math.Abs(ref)
		rel := absErr
		if den != 0 {
			rel = absErr / den
		}
		if rel > maxRel {
			maxRel = rel
		}
	}

	meanAbs := sumAbs / float64(len(samples))
	rms := math.Sqrt(sumSq / float64(len(samples)))

	digits := math.Inf(1)
	if maxRel > 0 {
		digits = -math.Log10(maxRel)
	}

	return AccuracyMetrics{
		MaxAbsError:   maxAbs,
		MaxRelError:   maxRel,
		MeanAbsError:  meanAbs,
		RMSError:      rms,
		DecimalDigits: digits,
	}
}
