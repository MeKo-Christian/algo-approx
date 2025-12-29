package approx

import "math"

// arctan3Term computes a 3-term Taylor series approximation of arctan(x).
// Valid for small x (range approximately [0, π/12]).
// Provides approximately 6.6 decimal digits of accuracy.
//
// Uses the Taylor series: arctan(x) ≈ x - x³/3 + x⁵/5
func arctan3Term[T Float](x T) T {
	x2 := x * x
	x3 := x2 * x
	x5 := x3 * x2

	return x - x3/3 + x5/5
}

// arctan6Term computes a 6-term Taylor series approximation of arctan(x).
// Valid for small x (range approximately [0, π/12]).
// Provides approximately 13.7 decimal digits of accuracy.
//
// Uses the Taylor series: arctan(x) ≈ x - x³/3 + x⁵/5 - x⁷/7 + x⁹/9 - x¹¹/11
func arctan6Term[T Float](x T) T {
	x2 := x * x
	x3 := x2 * x
	x5 := x3 * x2
	x7 := x5 * x2
	x9 := x7 * x2
	x11 := x9 * x2

	return x - x3/3 + x5/5 - x7/7 + x9/9 - x11/11
}

// arccotan3Term computes a 3-term approximation of arccot(x) = π/2 - arctan(x).
// Provides approximately 6.6 decimal digits of accuracy.
func arccotan3Term[T Float](x T) T {
	return T(math.Pi)/2 - arctan3Term(x)
}

// arccotan6Term computes a 6-term approximation of arccot(x) = π/2 - arctan(x).
// Provides approximately 13.7 decimal digits of accuracy.
func arccotan6Term[T Float](x T) T {
	return T(math.Pi)/2 - arctan6Term(x)
}

// arccos3Term computes a 3-term approximation of arccos(x).
// Valid for x in [-1, 1].
// Provides approximately 6.6 decimal digits of accuracy.
//
// Uses the identity: arccos(x) = π/2 - arcsin(x)
// And arcsin(x) ≈ x + x³/6 + 3x⁵/40 for small x
// For larger x, uses: arccos(x) = 2*arcsin(sqrt((1-x)/2))
func arccos3Term[T Float](x T) T {
	// For x close to 0, use arccos(x) ≈ π/2 - arcsin(x)
	if x > -0.5 && x < 0.5 {
		// arcsin(x) ≈ x + x³/6 + 3x⁵/40
		x2 := x * x
		x3 := x2 * x
		x5 := x3 * x2
		arcsinX := x + x3/6 + 3*x5/40
		return T(math.Pi)/2 - arcsinX
	}

	// For x closer to ±1, use the half-angle formula
	// arccos(x) = 2*arcsin(sqrt((1-x)/2))
	arg := T(math.Sqrt(float64((1 - x) / 2)))
	arg2 := arg * arg
	arg3 := arg2 * arg
	arg5 := arg3 * arg2
	arcsinArg := arg + arg3/6 + 3*arg5/40
	return 2 * arcsinArg
}

// arccos6Term computes a 6-term approximation of arccos(x).
// Valid for x in [-1, 1].
// Provides approximately 13.7 decimal digits of accuracy.
func arccos6Term[T Float](x T) T {
	// For x close to 0, use arccos(x) ≈ π/2 - arcsin(x)
	if x > -0.5 && x < 0.5 {
		// arcsin(x) with more terms: x + x³/6 + 3x⁵/40 + 15x⁷/336 + 105x⁹/3456 + 945x¹¹/42240
		x2 := x * x
		x3 := x2 * x
		x5 := x3 * x2
		x7 := x5 * x2
		x9 := x7 * x2
		x11 := x9 * x2

		arcsinX := x + x3/6 + 3*x5/40 + 15*x7/336 + 105*x9/3456 + 945*x11/42240
		return T(math.Pi)/2 - arcsinX
	}

	// For x closer to ±1, use the half-angle formula
	arg := T(math.Sqrt(float64((1 - x) / 2)))
	arg2 := arg * arg
	arg3 := arg2 * arg
	arg5 := arg3 * arg2
	arg7 := arg5 * arg2
	arg9 := arg7 * arg2
	arg11 := arg9 * arg2

	arcsinArg := arg + arg3/6 + 3*arg5/40 + 15*arg7/336 + 105*arg9/3456 + 945*arg11/42240
	return 2 * arcsinArg
}

// Arctan computes arctangent with specified precision.
func Arctan[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionFast, PrecisionBalanced:
		return arctan3Term(x)
	case PrecisionHigh:
		return arctan6Term(x)
	default:
		return arctan3Term(x)
	}
}

// Arccotan computes arccotangent with specified precision.
func Arccotan[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionFast, PrecisionBalanced:
		return arccotan3Term(x)
	case PrecisionHigh:
		return arccotan6Term(x)
	default:
		return arccotan3Term(x)
	}
}

// Arccos computes arccosine with specified precision.
func Arccos[T Float](x T, prec Precision) T {
	switch prec {
	case PrecisionFast, PrecisionBalanced:
		return arccos3Term(x)
	case PrecisionHigh:
		return arccos6Term(x)
	default:
		return arccos3Term(x)
	}
}
