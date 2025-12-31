package approx

import (
	"math"
)

func Sqrt[T Float](x T, prec Precision) T {
	impl := selectImpl(sqrtFast[T], sqrtBalanced[T], sqrtHigh[T], prec)
	return impl(x)
}

func sqrtFast[T Float](x T) T     { return sqrtBabylonian(x, 1) }
func sqrtBalanced[T Float](x T) T { return sqrtBabylonian(x, 2) }
func sqrtHigh[T Float](x T) T     { return sqrtBabylonian(x, 3) }

//nolint:varnamelen
func sqrtBabylonian[T Float](x T, iterations int) T {
	// Edge cases.
	if x == 0 {
		return 0
	}

	if x < 0 {
		return T(math.NaN())
	}

	if x != x { //nolint:gocritic
		return x
	}

	// +Inf stays +Inf.
	if math.IsInf(float64(x), 0) {
		return x
	}

	y := sqrtInitialGuess(x)
	if y == 0 {
		// Fallback, should be rare.
		y = x
	}

	// Babylonian iteration: y_{n+1} = 0.5*(y + x/y)
	half := T(0.5)
	for range iterations {
		y = half * (y + x/y)
	}

	return y
}

func sqrtInitialGuess[T Float](x T) T {
	var zero T
	switch any(zero).(type) {
	case float32:
		ux := math.Float32bits(float32(x))
		// Approximate sqrt by halving exponent; constant chosen empirically.
		ux = (ux >> 1) + 0x1fc00000

		return T(math.Float32frombits(ux))
	default:
		ux := math.Float64bits(float64(x))
		ux = (ux >> 1) + 0x1ff8000000000000

		return T(math.Float64frombits(ux))
	}
}
