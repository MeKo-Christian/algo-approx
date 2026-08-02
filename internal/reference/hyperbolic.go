package reference

import (
	"math"

	approx "github.com/cwbudde/algo-approx"
)

func Tanh[T approx.Float](x T) T {
	return T(math.Tanh(float64(x)))
}

// LogCosh is the reference log(cosh(x)).
//
// It is deliberately not math.Log(math.Cosh(x)): that overflows to +Inf above
// |x| ~ 710, which would make the reference wrong exactly where the
// approximation is most interesting. Above the branch point it uses the same
// asymptote the approximation does, but with math.Log1p and math.Exp, so it
// stays an independent check of everything except the identity itself.
func LogCosh[T approx.Float](x T) T {
	a := math.Abs(float64(x))
	if a < 0.5 {
		return T(math.Log(math.Cosh(a)))
	}

	return T(a - math.Ln2 + math.Log1p(math.Exp(-2*a)))
}

// Recip is the reference reciprocal: a true hardware divide.
func Recip[T approx.Float](x T) T {
	return T(1 / float64(x))
}
