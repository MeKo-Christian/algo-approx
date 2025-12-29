package approx

import (
	"math"
)

// Power computes base^exponent using the exp/log composition.
// This function uses the identity: base^exponent = exp(exponent * ln(base))
func Power[T Float](base, exponent T) T {
	// Handle special cases
	if base <= 0 {
		// For negative bases with non-integer exponents, result is undefined
		if base < 0 {
			return T(math.NaN())
		}
		// 0^x
		if exponent == 0 {
			return 1
		}
		if exponent > 0 {
			return 0
		}
		return T(math.Inf(1)) // 0^negative = infinity
	}

	if exponent == 0 {
		return 1
	}

	if exponent == 1 {
		return base
	}

	// Use exp/log composition: base^exponent = exp(exponent * ln(base))
	return Exp(exponent * Ln(base))
}

// Root computes the nth root of value using Power.
// This function uses the identity: root(value, n) = value^(1/n)
func Root[T Float](value T, n int) T {
	if n == 0 {
		return T(math.NaN())
	}

	if n == 1 {
		return value
	}

	if value < 0 {
		// Negative values only have real nth roots for odd n
		// For now, return NaN for simplicity
		return T(math.NaN())
	}

	if value == 0 {
		return 0
	}

	// Special case for square root (most common case)
	if n == 2 {
		return Sqrt2(value)
	}

	// For nth root: value^(1/n)
	return Power(value, T(1)/T(n))
}

// IntPower computes base^exponent for integer exponents using binary exponentiation.
// This is more efficient than the general Power function for integer exponents.
func IntPower[T Float](base T, exponent int) T {
	// Handle special cases
	if exponent == 0 {
		return 1
	}

	if exponent == 1 {
		return base
	}

	if base == 0 {
		if exponent > 0 {
			return 0
		}
		return T(math.Inf(1))
	}

	// Handle negative exponents
	if exponent < 0 {
		return 1 / IntPower(base, -exponent)
	}

	// Binary exponentiation for positive exponents
	result := T(1)
	currentBase := base
	n := exponent

	for n > 0 {
		if n%2 == 1 {
			result *= currentBase
		}
		currentBase *= currentBase
		n /= 2
	}

	return result
}
