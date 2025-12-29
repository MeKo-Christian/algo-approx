package reference

import (
	"math"

	approx "github.com/meko-christian/algo-approx"
)

func Exp[T approx.Float](x T) T {
	return T(math.Exp(float64(x)))
}
