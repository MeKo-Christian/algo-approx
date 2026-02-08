package reference

import (
	"math"

	approx "github.com/cwbudde/algo-approx"
)

func Log[T approx.Float](x T) T {
	return T(math.Log(float64(x)))
}
