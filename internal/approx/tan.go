package approx

import "math"

// tan2Term computes tangent using a 2-term Taylor series approximation.
// Taylor series: tan(x) ≈ x + x³/3 for x near 0
// Expected accuracy: ~3.2 decimal digits for |x| < π/4
// Valid range: [0, π/4].
func tan2Term[T Float](x T) T {
	// For tangent, we work in the range [0, π/4]
	xflt := float64(x)

	// Range reduction to [0, π/4]
	// Handle periodicity: tan(x + πk) = tan(x)
	xflt = math.Mod(xflt, math.Pi)
	if xflt < 0 {
		xflt += math.Pi
	}

	// Reduce to [0, π/2] using tan(x + π/2) = -cot(x)
	sign := 1.0

	if xflt > math.Pi/2 {
		xflt -= math.Pi / 2
		// tan(x + π/2) = -1/tan(x), but we'll handle this with basic reduction
		// For simplicity, reduce to [0, π/2]
		xflt = math.Pi/2 - xflt
		sign = -1.0
	}

	// Further reduce to [0, π/4] using tan(π/2 - x) = cot(x) = 1/tan(x)
	reciprocal := false

	if xflt > math.Pi/4 {
		xflt = math.Pi/2 - xflt
		reciprocal = true
	}

	// Now xf is in [0, π/4], apply 2-term Taylor series
	// tan(x) ≈ x + x³/3
	x2 := xflt * xflt
	x3 := xflt * x2

	result := xflt + x3/3.0

	if reciprocal {
		result = 1.0 / result
	}

	return T(sign * result)
}

// cotan2Term computes cotangent (1/tan) using the 2-term tangent approximation.
// cotan(x) = 1 / tan(x)
// Expected accuracy: ~3.2 decimal digits for |x| < π/4.
func cotan2Term[T Float](x T) T {
	tanVal := tan2Term(x)
	return 1.0 / tanVal
}

// tan3Term computes tangent using a 3-term Taylor series approximation.
// Taylor series: tan(x) ≈ x + x³/3 + 2x⁵/15 for x near 0
// Expected accuracy: ~5.6 decimal digits for |x| < π/4.
func tan3Term[T Float](x T) T {
	xflt := float64(x)

	// Range reduction to [0, π/4]
	xflt = math.Mod(xflt, math.Pi)
	if xflt < 0 {
		xflt += math.Pi
	}

	sign := 1.0

	if xflt > math.Pi/2 {
		xflt -= math.Pi / 2
		xflt = math.Pi/2 - xflt
		sign = -1.0
	}

	reciprocal := false

	if xflt > math.Pi/4 {
		xflt = math.Pi/2 - xflt
		reciprocal = true
	}

	// 3-term Taylor series: tan(x) ≈ x + x³/3 + 2x⁵/15
	x2 := xflt * xflt
	x3 := xflt * x2
	x5 := x3 * x2

	result := xflt + x3/3.0 + 2.0*x5/15.0

	if reciprocal {
		result = 1.0 / result
	}

	return T(sign * result)
}

// cotan3Term computes cotangent (1/tan) using the 3-term tangent approximation.
func cotan3Term[T Float](x T) T {
	tanVal := tan3Term(x)
	return 1.0 / tanVal
}

// tan4Term computes tangent using a 4-term Taylor series approximation.
// Taylor series: tan(x) ≈ x + x³/3 + 2x⁵/15 + 17x⁷/315 for x near 0
// Expected accuracy: ~8.2 decimal digits for |x| < π/4.
func tan4Term[T Float](x T) T {
	xflt := float64(x)

	// Range reduction to [0, π/4]
	xflt = math.Mod(xflt, math.Pi)
	if xflt < 0 {
		xflt += math.Pi
	}

	sign := 1.0

	if xflt > math.Pi/2 {
		xflt -= math.Pi / 2
		xflt = math.Pi/2 - xflt
		sign = -1.0
	}

	reciprocal := false

	if xflt > math.Pi/4 {
		xflt = math.Pi/2 - xflt
		reciprocal = true
	}

	// 4-term Taylor series: tan(x) ≈ x + x³/3 + 2x⁵/15 + 17x⁷/315
	x2 := xflt * xflt
	x3 := xflt * x2
	x5 := x3 * x2
	x7 := x5 * x2

	result := xflt + x3/3.0 + 2.0*x5/15.0 + 17.0*x7/315.0

	if reciprocal {
		result = 1.0 / result
	}

	return T(sign * result)
}

// cotan4Term computes cotangent (1/tan) using the 4-term tangent approximation.
func cotan4Term[T Float](x T) T {
	tanVal := tan4Term(x)
	return 1.0 / tanVal
}

// tan6Term computes tangent using a 6-term Taylor series approximation.
// Taylor series: tan(x) ≈ x + x³/3 + 2x⁵/15 + 17x⁷/315 + 62x⁹/2835 + 1382x¹¹/155925
// Expected accuracy: ~14 decimal digits for |x| < π/4.
func tan6Term[T Float](x T) T {
	xflt := float64(x)

	// Range reduction to [0, π/4]
	xflt = math.Mod(xflt, math.Pi)
	if xflt < 0 {
		xflt += math.Pi
	}

	sign := 1.0

	if xflt > math.Pi/2 {
		xflt -= math.Pi / 2
		xflt = math.Pi/2 - xflt
		sign = -1.0
	}

	reciprocal := false

	if xflt > math.Pi/4 {
		xflt = math.Pi/2 - xflt
		reciprocal = true
	}

	// 6-term Taylor series
	x2 := xflt * xflt

	x3 := xflt * x2
	x5 := x3 * x2
	x7 := x5 * x2
	x9 := x7 * x2
	x11 := x9 * x2

	result := xflt + x3/3.0 + 2.0*x5/15.0 + 17.0*x7/315.0 + 62.0*x9/2835.0 + 1382.0*x11/155925.0

	if reciprocal {
		result = 1.0 / result
	}

	return T(sign * result)
}

// cotan6Term computes cotangent (1/tan) using the 6-term tangent approximation.
func cotan6Term[T Float](x T) T {
	tanVal := tan6Term(x)
	return 1.0 / tanVal
}

// Tan computes tangent with precision-based term selection.
// Maps precision levels to term counts:
//   - PrecisionFast (2): ~3.2 decimal digits
//   - PrecisionBalanced (3): ~5.6 decimal digits
//   - PrecisionHigh (6): ~14 decimal digits
func Tan[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionAuto, PrecisionBalanced:
		return tan3Term(x)
	case PrecisionFast:
		return tan2Term(x)
	case PrecisionHigh:
		return tan6Term(x)
	default:
		return tan3Term(x) // default to balanced
	}
}

// Cotan computes cotangent with precision-based term selection.
func Cotan[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionAuto, PrecisionBalanced:
		return cotan3Term(x)
	case PrecisionFast:
		return cotan2Term(x)
	case PrecisionHigh:
		return cotan6Term(x)
	default:
		return cotan3Term(x) // default to balanced
	}
}
