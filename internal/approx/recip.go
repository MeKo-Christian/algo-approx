package approx

import "math"

const (
	// recipMagic is the float64 analogue of the classic 0x7EF311C3 float32
	// reciprocal seed: subtracting the bit pattern negates the exponent and
	// gives a piecewise-linear-in-bits estimate of the mantissa reciprocal.
	// Applied to a mantissa in [1,2) it is accurate to 5.051e-2 relative,
	// worst case at m = 1.
	recipMagic = 0x7FDE623822FC16E6

	// recipScale (2^54) lifts a subnormal into the normal range; the result is
	// scaled back by the same amount, which is where 1/subnormal correctly
	// overflows to +Inf.
	recipScale    = 1 << 54
	recipScaleExp = 54
)

// Recip returns an approximate 1/x.
//
// Go has no reciprocal intrinsic: the expression 1/x lowers to a true DIVSD on
// amd64 (and FDIV on arm64). This routine replaces that with a bit-trick seed
// plus Newton-Raphson, which trades one long-latency instruction for a chain
// of short ones. Whether that is a win is entirely a property of the caller's
// dependency structure; see the benchmark discussion in README.md before
// reaching for it.
//
// Precision selects the number of quadratic Newton steps applied on top of the
// seed:
//
//	PrecisionFast      1 step   ~1.7e-8  relative
//	PrecisionBalanced  2 steps  ~3e-16   relative (full float64 in practice)
//	PrecisionHigh      3 steps  <=1 ulp
//
// The seed itself is the magic-constant estimate (5.05e-2) polished by one
// cubic step, y <- y*(1 + r + r^2) with r = 1 - m*y, giving ~1.3e-4. Without
// that polish the requested step counts cannot reach the accuracies above: a
// bare magic-constant seed converges 5.05e-2 -> 2.6e-3 -> 6.6e-6 -> 4.4e-11,
// so two Newton steps would land five orders short of full precision.
//
// Edge cases match 1/x exactly: NaN -> NaN, +/-0 -> +/-Inf, +/-Inf -> +/-0,
// and subnormal inputs are normalized before the seed so they do not fall off
// the bottom of the exponent field.
//
// A thin generic shim over the concrete float64 kernel; see Log for why.
func Recip[T Float](x T, prec Precision) T {
	return T(recip64(float64(x), prec))
}

func recip64(xflt float64, prec Precision) float64 {
	if math.IsNaN(xflt) {
		return xflt
	}

	if xflt == 0 {
		if math.Signbit(xflt) {
			return math.Inf(-1)
		}

		return math.Inf(1)
	}

	if math.IsInf(xflt, 0) {
		if math.Signbit(xflt) {
			return math.Copysign(0, -1)
		}

		return 0
	}

	// Work on the magnitude and reattach the sign, so Recip(-x) is the exact
	// negation of Recip(x) and -0 falls out of the zero case above.
	a := math.Abs(xflt)

	shift := 0

	if a < smallestNormalFloat64 {
		a *= recipScale
		shift = recipScaleExp
	}

	bits := math.Float64bits(a)
	exp := int((bits>>52)&0x7ff) - 1023 - shift

	// Mantissa in [1,2): clear the exponent field and set it to bias.
	mant := math.Float64frombits((bits &^ (uint64(0x7ff) << 52)) | (uint64(1023) << 52))

	y := recipMantissa(mant, prec)

	res := scalePow2(y, -exp)

	if math.Signbit(xflt) {
		return -res
	}

	return res
}

// recipMantissa returns an approximation of 1/m for m in [1,2), so the result
// lies in (0.5, 1] and neither the seed nor any iterate can over- or underflow.
func recipMantissa(m float64, prec Precision) float64 {
	est := math.Float64frombits(recipMagic - math.Float64bits(m))

	// Cubic seed polish: 1/m = est/(1-resid) = est*(1 + resid + resid^2 + ...)
	// with resid = 1 - m*est, |resid| <= 5.06e-2. Truncating after the square
	// term leaves ~resid^3 ~ 1.3e-4.
	resid := 1 - m*est
	est += est * (resid + resid*resid)

	steps := 2

	switch normalizePrecision(prec) {
	case PrecisionFast:
		steps = 1
	case PrecisionHigh:
		steps = 3
	case PrecisionAuto, PrecisionBalanced:
		steps = 2
	}

	// Newton-Raphson for f(e) = 1/e - m: e <- e + e*(1 - m*e). Written as an
	// increment rather than e*(2 - m*e) because the residual is formed in full
	// precision and the correction is small, which costs the same and rounds
	// better.
	for range steps {
		resid = 1 - m*est
		est += est * resid
	}

	return est
}

// scalePow2 returns y * 2^n. y is in (0.5, 1] here, so the fast path cannot
// overflow; only the subnormal-input case can leave the exactly-representable
// exponent window, and that falls through to math.Ldexp.
func scalePow2(y float64, n int) float64 {
	if n >= -1021 && n <= 1023 {
		return y * math.Float64frombits(uint64(n+1023)<<52)
	}

	return math.Ldexp(y, n)
}
