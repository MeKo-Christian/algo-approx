package reference

import (
	"math"

	approx "github.com/meko-christian/algo-approx"
)

func Sqrt[T approx.Float](x T) T {
	return T(math.Sqrt(float64(x)))
}

func InvSqrt[T approx.Float](x T) T {
	return T(1.0 / math.Sqrt(float64(x)))
}
