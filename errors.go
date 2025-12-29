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
