package reference

import (
	"math"

	approx "github.com/cwbudde/algo-approx"
)

// Sin computes the reference sine using math.Sin.
// This provides the baseline for accuracy measurements.
func Sin[T approx.Float](x T) T {
	return T(math.Sin(float64(x)))
}

// Cos computes the reference cosine using math.Cos.
// This provides the baseline for accuracy measurements.
func Cos[T approx.Float](x T) T {
	return T(math.Cos(float64(x)))
}
