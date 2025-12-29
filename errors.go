package approx

import "errors"

var (
	// ErrDomainError indicates the input is outside the valid domain.
	ErrDomainError = errors.New("input outside valid domain")
	// ErrNaN indicates the result is not a number.
	ErrNaN = errors.New("result is not a number")
	// ErrInfinity indicates the result is infinite.
	ErrInfinity = errors.New("result is infinite")
)

func isNaN[T Float](x T) bool { return x != x }

func isInf[T Float](x T) bool {
	// IEEE-754: +Inf compares greater than MaxFloat, -Inf compares less than -MaxFloat.
	if x > 0 {
		return x > T(maxFinite[T]())
	}
	return x < -T(maxFinite[T]())
}

func maxFinite[T Float]() float64 {
	// Avoid importing math here to keep this file tiny; constants are exact.
	var zero T
	switch any(zero).(type) {
	case float32:
		return 3.4028234663852886e+38
	default:
		return 1.7976931348623157e+308
	}
}
