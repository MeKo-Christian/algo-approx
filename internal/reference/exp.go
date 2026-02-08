package reference

import (
	"math"

	approx "github.com/cwbudde/algo-approx"
)

func Exp[T approx.Float](x T) T {
	return T(math.Exp(float64(x)))
}
