package approx

import "math"

// Taylor coefficients of log(cosh(x)) written in z = x*x, i.e.
//
//	log(cosh(x)) = z * (c0 + c1*z + c2*z^2 + ...)
//
// These are not fitted. They are the term-by-term antiderivative of the tanh
// Taylor series
//
//	tanh(x) = x - x^3/3 + 2x^5/15 - 17x^7/315 + 62x^9/2835 - ...
//
// which is why the small-|x| branch of LogCosh differentiates back to tanh
// exactly, coefficient for coefficient, rather than approximately.
const (
	logCoshC0 = 1.0 / 2.0
	logCoshC1 = -1.0 / 12.0
	logCoshC2 = 1.0 / 45.0
	logCoshC3 = -17.0 / 2520.0
	logCoshC4 = 31.0 / 14175.0
	logCoshC5 = -691.0 / 935550.0
	logCoshC6 = 10922.0 / 42567525.0
	logCoshC7 = -929569.0 / 10216206000.0
	logCoshC8 = 3202291.0 / 97692469875.0
	logCoshC9 = -221930581.0 / 18561569276250.0
)

// LogCosh returns an approximate log(cosh(x)).
//
// # Consistency with Tanh
//
// tanh is exactly d/dx log(cosh(x)), and consumers such as discrete-gradient
// energy schemes depend on that identity holding for the approximations, not
// just for the exact functions. The two are therefore designed as one object:
//
//   - Below |x| = 0.625 LogCosh evaluates the term-by-term antiderivative of
//     the tanh Taylor series, so differentiating the polynomial reproduces
//     tanh's own small-argument expansion exactly.
//   - At or above |x| = 0.625 both read the same u = exp(-2|x|) from the same
//     expNeg2 helper. LogCosh returns |x| - ln2 + log1p(u) and Tanh returns
//     (1-u)/(1+u); differentiating the former gives 1 - 2u/(1+u), which is the
//     latter identically, for any u whatsoever.
//   - The branch point is the same constant (tanhBranch) in both, so the
//     identity never straddles a seam.
//
// The residual mismatch is therefore only the series truncations, i.e. the
// relationship holds to the approximation's own accuracy (measured below 3e-8
// at PrecisionBalanced over |x| < 12) and not merely to within some looser
// tolerance. It is not exact and is not claimed to be.
//
// # Overflow
//
// The naive math.Log(math.Cosh(x)) overflows for |x| above ~710 because cosh
// overflows before the log can pull it back. This implementation never forms
// cosh: it uses the asymptote |x| - ln2 + log1p(exp(-2|x|)) directly, which is
// finite for every finite input.
//
// Accuracy at PrecisionBalanced is below 1e-7 absolute over |x| < 12 (the
// target consumer domain) and degrades gracefully outside it: beyond |x| = 12
// the exp term is smaller than 4e-11 and the result is dominated by the exact
// |x| - ln2.
//
// A thin generic shim over the concrete float64 kernel; see Log for why.
func LogCosh[T Float](x T, prec Precision) T {
	return T(logCosh64(float64(x), prec))
}

func logCosh64(xflt float64, prec Precision) float64 {
	if math.IsNaN(xflt) {
		return xflt
	}

	// log(cosh) is even, so the magnitude is the whole story. +/-0 -> +0.
	a := math.Abs(xflt)

	if math.IsInf(a, 1) {
		return math.Inf(1)
	}

	if a < tanhBranch {
		z := a * a

		return z * logCoshPoly(z, prec)
	}

	u := expNeg2(a, prec)

	return a - ln2 + log1pSmall(u, prec)
}

// logCoshPoly evaluates the bracket of log(cosh(x)) = z * poly(z) for
// z = x*x < 0.390625.
//
//nolint:varnamelen
func logCoshPoly(z float64, prec Precision) float64 {
	switch normalizePrecision(prec) {
	case PrecisionFast:
		// Through z^3; ~4e-5 absolute at the branch point.
		return logCoshC0 + z*(logCoshC1+z*(logCoshC2+z*logCoshC3))
	case PrecisionAuto, PrecisionBalanced, PrecisionHigh:
		fallthrough
	default:
		// Through z^9; ~1.4e-10 absolute at the branch point, which is the
		// worst point of this branch. Balanced and High share it: the series
		// only gains a factor ~7 per term at z = 0.39, so chasing it further
		// costs more than the exponential branch can match anyway. This is
		// what bounds LogCosh at PrecisionHigh.
		return logCoshC0 + z*(logCoshC1+z*(logCoshC2+z*(logCoshC3+
			z*(logCoshC4+z*(logCoshC5+z*(logCoshC6+z*(logCoshC7+
				z*(logCoshC8+z*logCoshC9))))))))
	}
}

// log1pSmall returns log(1+u) for u in (0, 0.2866], the range produced by
// expNeg2 at and above the branch point.
//
// It uses the atanh form log(1+u) = 2*atanh(w), w = u/(2+u) <= 0.1253, which
// converges in a handful of odd terms where the plain Mercator series in u
// would need eleven. math.Log1p would also work and is more accurate, but it
// costs more than the whole rest of the function.
//
//nolint:varnamelen
func log1pSmall(u float64, prec Precision) float64 {
	w := u / (2 + u)
	w2 := w * w

	switch normalizePrecision(prec) {
	case PrecisionFast:
		// w + w^3/3; ~1e-5 absolute worst case.
		return 2 * w * (1 + w2*(1.0/3.0))
	case PrecisionHigh:
		// through w^13/13; ~4e-15 absolute worst case.
		return 2 * w * (1 + w2*(1.0/3.0+w2*(1.0/5.0+w2*(1.0/7.0+w2*(1.0/9.0+
			w2*(1.0/11.0+w2*(1.0/13.0)))))))
	case PrecisionAuto, PrecisionBalanced:
		fallthrough
	default:
		// through w^9/9; ~2e-11 absolute worst case, chosen so the branch
		// seam is set by the polynomial branch rather than by this one.
		return 2 * w * (1 + w2*(1.0/3.0+w2*(1.0/5.0+w2*(1.0/7.0+w2*(1.0/9.0)))))
	}
}
