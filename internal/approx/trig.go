package approx

import "math"

// sin3Term computes sine using a 3-term Taylor series approximation.
// Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! for x near 0
// Expected accuracy: ~3.2 decimal digits for |x| < π/2
func sin3Term[T Float](x T) T {
	// Range reduction: reduce x to [-π/2, π/2]
	xf := float64(x)

	// Handle periodicity: sin(x + 2πk) = sin(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	// Reduce to [-π, π]
	if xf > math.Pi {
		xf -= twoPi
	} else if xf < -math.Pi {
		xf += twoPi
	}

	// Reduce to [-π/2, π/2] using sin(π - x) = sin(x) and sin(-π - x) = -sin(x)
	sign := T(1.0)
	if xf > math.Pi/2 {
		xf = math.Pi - xf
	} else if xf < -math.Pi/2 {
		xf = -math.Pi - xf
	}

	// Now xf is in [-π/2, π/2], apply 3-term Taylor series
	// sin(x) ≈ x - x³/6 + x⁵/120
	x2 := xf * xf
	x3 := xf * x2
	x5 := x3 * x2

	result := xf - x3/6.0 + x5/120.0

	return sign * T(result)
}

// cos3Term computes cosine using a 3-term Taylor series approximation.
// Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! for x near 0
// Expected accuracy: ~3.2 decimal digits for |x| < π/2
func cos3Term[T Float](x T) T {
	// Range reduction: reduce x to [0, π]
	xf := float64(x)

	// Handle periodicity: cos(x + 2πk) = cos(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	// Reduce to [0, 2π]
	if xf < 0 {
		xf += twoPi
	}

	// Reduce to [0, π] using cos(2π - x) = cos(x)
	if xf > math.Pi {
		xf = twoPi - xf
	}

	// Now xf is in [0, π], apply 3-term Taylor series
	// cos(x) ≈ 1 - x²/2 + x⁴/24
	x2 := xf * xf
	x4 := x2 * x2

	result := 1.0 - x2/2.0 + x4/24.0

	return T(result)
}

// sec3Term computes secant (1/cos) using the 3-term cosine approximation.
// sec(x) = 1 / cos(x)
// Expected accuracy: ~3.2 decimal digits for |x| < π/2
func sec3Term[T Float](x T) T {
	cosVal := cos3Term(x)
	return 1.0 / cosVal
}

// csc3Term computes cosecant (1/sin) using the 3-term sine approximation.
// csc(x) = 1 / sin(x)
// Expected accuracy: ~3.2 decimal digits for |x| < π/2
func csc3Term[T Float](x T) T {
	sinVal := sin3Term(x)
	return 1.0 / sinVal
}

// sin4Term computes sine using a 4-term Taylor series approximation.
// Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! for x near 0
// Expected accuracy: ~5.2 decimal digits for |x| < π/2
func sin4Term[T Float](x T) T {
	// Range reduction: reduce x to [-π/2, π/2]
	xf := float64(x)

	// Handle periodicity: sin(x + 2πk) = sin(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	// Reduce to [-π, π]
	if xf > math.Pi {
		xf -= twoPi
	} else if xf < -math.Pi {
		xf += twoPi
	}

	// Reduce to [-π/2, π/2]
	sign := T(1.0)
	if xf > math.Pi/2 {
		xf = math.Pi - xf
	} else if xf < -math.Pi/2 {
		xf = -math.Pi - xf
	}

	// Apply 4-term Taylor series
	// sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
	x2 := xf * xf
	x3 := xf * x2
	x5 := x3 * x2
	x7 := x5 * x2

	result := xf - x3/6.0 + x5/120.0 - x7/5040.0

	return sign * T(result)
}

// cos4Term computes cosine using a 4-term Taylor series approximation.
// Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! for x near 0
// Expected accuracy: ~5.2 decimal digits for |x| < π/2
func cos4Term[T Float](x T) T {
	// Range reduction: reduce x to [0, π]
	xf := float64(x)

	// Handle periodicity: cos(x + 2πk) = cos(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	// Reduce to [0, 2π]
	if xf < 0 {
		xf += twoPi
	}

	// Reduce to [0, π]
	if xf > math.Pi {
		xf = twoPi - xf
	}

	// Apply 4-term Taylor series
	// cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
	x2 := xf * xf
	x4 := x2 * x2
	x6 := x4 * x2

	result := 1.0 - x2/2.0 + x4/24.0 - x6/720.0

	return T(result)
}

// sec4Term computes secant using the 4-term cosine approximation.
func sec4Term[T Float](x T) T {
	cosVal := cos4Term(x)
	return 1.0 / cosVal
}

// csc4Term computes cosecant using the 4-term sine approximation.
func csc4Term[T Float](x T) T {
	sinVal := sin4Term(x)
	return 1.0 / sinVal
}

// sin5Term computes sine using a 5-term Taylor series approximation.
// Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! for x near 0
// Expected accuracy: ~7.3 decimal digits for |x| < π/2
func sin5Term[T Float](x T) T {
	// Range reduction: reduce x to [-π/2, π/2]
	xf := float64(x)

	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	if xf > math.Pi {
		xf -= twoPi
	} else if xf < -math.Pi {
		xf += twoPi
	}

	sign := T(1.0)
	if xf > math.Pi/2 {
		xf = math.Pi - xf
	} else if xf < -math.Pi/2 {
		xf = -math.Pi - xf
	}

	// Apply 5-term Taylor series
	// sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040 + x⁹/362880
	x2 := xf * xf
	x3 := xf * x2
	x5 := x3 * x2
	x7 := x5 * x2
	x9 := x7 * x2

	result := xf - x3/6.0 + x5/120.0 - x7/5040.0 + x9/362880.0

	return sign * T(result)
}

// cos5Term computes cosine using a 5-term Taylor series approximation.
// Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8! for x near 0
// Expected accuracy: ~7.3 decimal digits for |x| < π/2
func cos5Term[T Float](x T) T {
	// Range reduction: reduce x to [0, π]
	xf := float64(x)

	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	if xf < 0 {
		xf += twoPi
	}

	if xf > math.Pi {
		xf = twoPi - xf
	}

	// Apply 5-term Taylor series
	// cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720 + x⁸/40320
	x2 := xf * xf
	x4 := x2 * x2
	x6 := x4 * x2
	x8 := x6 * x2

	result := 1.0 - x2/2.0 + x4/24.0 - x6/720.0 + x8/40320.0

	return T(result)
}

// sec5Term computes secant using the 5-term cosine approximation.
func sec5Term[T Float](x T) T {
	cosVal := cos5Term(x)
	return 1.0 / cosVal
}

// csc5Term computes cosecant using the 5-term sine approximation.
func csc5Term[T Float](x T) T {
	sinVal := sin5Term(x)
	return 1.0 / sinVal
}

// sin6Term computes sine using a 6-term Taylor series approximation.
// Expected accuracy: ~9 decimal digits for |x| < π/2
func sin6Term[T Float](x T) T {
	xf := float64(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	if xf > math.Pi {
		xf -= twoPi
	} else if xf < -math.Pi {
		xf += twoPi
	}

	sign := T(1.0)
	if xf > math.Pi/2 {
		xf = math.Pi - xf
	} else if xf < -math.Pi/2 {
		xf = -math.Pi - xf
	}

	// 6-term: add x¹¹/11!
	x2 := xf * xf
	x3 := xf * x2
	x5 := x3 * x2
	x7 := x5 * x2
	x9 := x7 * x2
	x11 := x9 * x2

	result := xf - x3/6.0 + x5/120.0 - x7/5040.0 + x9/362880.0 - x11/39916800.0

	return sign * T(result)
}

// cos6Term computes cosine using a 6-term Taylor series approximation.
// Expected accuracy: ~9 decimal digits for |x| < π/2
func cos6Term[T Float](x T) T {
	xf := float64(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	if xf < 0 {
		xf += twoPi
	}

	if xf > math.Pi {
		xf = twoPi - xf
	}

	// 6-term: add x¹⁰/10!
	x2 := xf * xf
	x4 := x2 * x2
	x6 := x4 * x2
	x8 := x6 * x2
	x10 := x8 * x2

	result := 1.0 - x2/2.0 + x4/24.0 - x6/720.0 + x8/40320.0 - x10/3628800.0

	return T(result)
}

// sec6Term computes secant using the 6-term cosine approximation.
func sec6Term[T Float](x T) T {
	cosVal := cos6Term(x)
	return 1.0 / cosVal
}

// csc6Term computes cosecant using the 6-term sine approximation.
func csc6Term[T Float](x T) T {
	sinVal := sin6Term(x)
	return 1.0 / sinVal
}

// sin7Term computes sine using a 7-term Taylor series approximation.
// Expected accuracy: ~12.1 decimal digits for |x| < π/2
func sin7Term[T Float](x T) T {
	xf := float64(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	if xf > math.Pi {
		xf -= twoPi
	} else if xf < -math.Pi {
		xf += twoPi
	}

	sign := T(1.0)
	if xf > math.Pi/2 {
		xf = math.Pi - xf
	} else if xf < -math.Pi/2 {
		xf = -math.Pi - xf
	}

	// 7-term: add x¹³/13!
	x2 := xf * xf
	x3 := xf * x2
	x5 := x3 * x2
	x7 := x5 * x2
	x9 := x7 * x2
	x11 := x9 * x2
	x13 := x11 * x2

	result := xf - x3/6.0 + x5/120.0 - x7/5040.0 + x9/362880.0 - x11/39916800.0 + x13/6227020800.0

	return sign * T(result)
}

// cos7Term computes cosine using a 7-term Taylor series approximation.
// Expected accuracy: ~12.1 decimal digits for |x| < π/2
func cos7Term[T Float](x T) T {
	xf := float64(x)
	const twoPi = 2 * math.Pi
	xf = math.Mod(xf, twoPi)

	if xf < 0 {
		xf += twoPi
	}

	if xf > math.Pi {
		xf = twoPi - xf
	}

	// 7-term: add x¹²/12!
	x2 := xf * xf
	x4 := x2 * x2
	x6 := x4 * x2
	x8 := x6 * x2
	x10 := x8 * x2
	x12 := x10 * x2

	result := 1.0 - x2/2.0 + x4/24.0 - x6/720.0 + x8/40320.0 - x10/3628800.0 + x12/479001600.0

	return T(result)
}

// sec7Term computes secant using the 7-term cosine approximation.
func sec7Term[T Float](x T) T {
	cosVal := cos7Term(x)
	return 1.0 / cosVal
}

// csc7Term computes cosecant using the 7-term sine approximation.
func csc7Term[T Float](x T) T {
	sinVal := sin7Term(x)
	return 1.0 / sinVal
}

// Sin computes sine with the requested precision level.
// Maps precision to term count: Fast=3, Balanced=5, High=7
func Sin[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionFast:
		return sin3Term(x)
	case PrecisionBalanced:
		return sin5Term(x)
	case PrecisionHigh:
		return sin7Term(x)
	default:
		return sin5Term(x) // Default to balanced
	}
}

// Cos computes cosine with the requested precision level.
// Maps precision to term count: Fast=3, Balanced=5, High=7
func Cos[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionFast:
		return cos3Term(x)
	case PrecisionBalanced:
		return cos5Term(x)
	case PrecisionHigh:
		return cos7Term(x)
	default:
		return cos5Term(x) // Default to balanced
	}
}

// Sec computes secant with the requested precision level.
func Sec[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionFast:
		return sec3Term(x)
	case PrecisionBalanced:
		return sec5Term(x)
	case PrecisionHigh:
		return sec7Term(x)
	default:
		return sec5Term(x)
	}
}

// Csc computes cosecant with the requested precision level.
func Csc[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionFast:
		return csc3Term(x)
	case PrecisionBalanced:
		return csc5Term(x)
	case PrecisionHigh:
		return csc7Term(x)
	default:
		return csc5Term(x)
	}
}
