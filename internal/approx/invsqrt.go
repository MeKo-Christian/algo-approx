package approx

import (
	"math"
)

func InvSqrt[T Float](x T, prec Precision) T {
	impl := selectImpl(invSqrtFast[T], invSqrtBalanced[T], invSqrtHigh[T], prec)
	return impl(x)
}

func invSqrtFast[T Float](x T) T     { return invSqrtQuakeNR(x, 1) }
func invSqrtBalanced[T Float](x T) T { return invSqrtQuakeNR(x, 2) }
func invSqrtHigh[T Float](x T) T     { return invSqrtQuakeNR(x, 3) }

//nolint:varnamelen
func invSqrtQuakeNR[T Float](x T, iters int) T {
	// Edge cases.
	if x == 0 {
		return T(math.Inf(1))
	}

	if x < 0 {
		return T(math.NaN())
	}

	if x != x { //nolint:gocritic
		return x
	}

	if math.IsInf(float64(x), 0) {
		// 1/sqrt(+Inf) = 0
		return 0
	}

	y := invSqrtQuake(x)
	half := T(0.5)
	threeHalf := T(1.5)

	// Newton-Raphson refinement for 1/sqrt(x): y = y*(1.5 - 0.5*x*y*y)
	for range iters {
		y *= (threeHalf - half*x*y*y)
	}

	return y
}

func invSqrtQuake[T Float](x T) T {
	var zero T
	switch any(zero).(type) {
	case float32:
		xf := float32(x)
		i := math.Float32bits(xf)
		i = 0x5f3759df - (i >> 1)
		y := math.Float32frombits(i)

		return T(y)
	default:
		xf := float64(x)
		i := math.Float64bits(xf)
		// Commonly used 64-bit magic constant for the Quake-style seed.
		i = 0x5fe6eb50c7b537a9 - (i >> 1)
		y := math.Float64frombits(i)

		return T(y)
	}
}
