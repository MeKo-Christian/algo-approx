package approx

import "math"

// Branch structure shared by Tanh and LogCosh.
//
// Both functions are built from exactly one shared quantity for |x| >= the
// branch point: u = exp(-2|x|), evaluated by expNeg2. With that single
// quantity,
//
//	tanh(|x|)  = (1 - u) / (1 + u)
//	logCosh(x) = |x| - ln2 + log1p(u)
//
// and d/dx of the second is algebraically identical to the first. Sharing u is
// what makes the pair consistent rather than merely individually accurate; see
// the LogCosh doc comment.
const (
	// tanhBranch is the switch point between the rational core and the
	// exponential form. Below it, (1-u)/(1+u) would cancel badly (u -> 1);
	// above it the rational core would need far more terms.
	tanhBranch = 0.625

	// tanhSaturation is a value at or above which float64 tanh is
	// indistinguishable from 1. The true crossover is 19.06154746539849...;
	// 19.0625 = 305/16 is the next exactly-representable value above it.
	tanhSaturation = 19.0625

	// expNeg2Zero is the smallest a for which exp(-2a) underflows to zero
	// (2a below minLogFloat64 = -745.133...). Guarding here also keeps the
	// range reduction's int conversion in range for huge a.
	expNeg2Zero = 373.0
)

// Rational core coefficients for tanh on |x| < 0.625.
//
// The form is the odd [7/6] rational
//
//	tanh(x) ~= x + x*z*P(z)/Q(z),  z = x*x
//	P(z) = ((p0*z + p1)*z + p2)
//	Q(z) = ((z + q0)*z + q1)*z + q2
//
// i.e. a Pade-style rational in z whose numerator degree in x is 7 and whose
// denominator degree in x is 6, so the odd symmetry and the x -> 0 behaviour
// (tanh(x) = x - x^3/3 + ...) are structural rather than fitted. The
// coefficients are the classic Cephes minimax set for this interval; their
// error profile is a near-equioscillating relative error bounded by ~2e-17
// over |x| < 0.625, i.e. below one float64 ulp, so on this branch the rational
// is not the accuracy-limiting step. Truncating to fewer terms was rejected:
// the branch is only ~4 multiply-adds wide and the shared branch point with
// LogCosh is worth more than the saving.
const (
	tanhP0 = -9.64399179425052238628e-1
	tanhP1 = -9.92877231001918586564e+1
	tanhP2 = -1.61468768441708447952e+3

	tanhQ0 = +1.12811678491632931402e+2
	tanhQ1 = +2.23548839060100448583e+3
	tanhQ2 = +4.84406305325125486048e+3
)

// Tanh returns an approximate hyperbolic tangent.
//
// Guarantees that hold at every precision:
//
//   - Exact odd symmetry: Tanh(-x) is the bit-for-bit negation of Tanh(x).
//     The magnitude is computed from |x| and the sign is reattached, so the
//     two can never disagree.
//   - Exact saturation: |x| >= 19.0625 returns exactly +/-1, which is what
//     math.Tanh returns there too.
//   - Edge cases match math.Tanh: NaN -> NaN, +/-Inf -> +/-1, +/-0 -> +/-0.
//
// At PrecisionBalanced the maximum absolute error over x in [-20, 20] is
// below 1e-7 (measured ~2e-9); see ACCURACY.md.
//
// A thin generic shim over the concrete float64 kernel; see Log for why.
func Tanh[T Float](x T, prec Precision) T {
	return T(tanh64(float64(x), prec))
}

func tanh64(xflt float64, prec Precision) float64 {
	if math.IsNaN(xflt) {
		return xflt
	}

	// Everything below works on the magnitude; the sign is reattached last.
	// This is what makes odd symmetry exact rather than approximate, and it
	// carries -0 through to -0 for free.
	a := math.Abs(xflt)

	var mag float64

	switch {
	case a >= tanhSaturation:
		// Also the +Inf case.
		mag = 1
	case a < tanhBranch:
		z := a * a
		num := ((tanhP0*z+tanhP1)*z + tanhP2)
		den := (((z+tanhQ0)*z+tanhQ1)*z + tanhQ2)
		mag = a + a*z*num/den
	default:
		u := expNeg2(a, prec)
		mag = 1 - 2*u/(1+u)
	}

	if math.Signbit(xflt) {
		return -mag
	}

	return mag
}

// expNeg2 returns exp(-2a) for a >= tanhBranch.
//
// It is the single shared building block of Tanh and LogCosh: both read the
// same u, so the derivative relationship between them is limited only by the
// series truncation, never by the two disagreeing about exp.
//
// Range reduction is -2a = k*ln2 + r with |r| <= ln2/2, then a Taylor
// polynomial in r whose degree is chosen by prec.
func expNeg2(a float64, prec Precision) float64 {
	// LogCosh accepts any finite input, so a can be astronomically large.
	// Bail out before the range reduction, whose int conversion would
	// overflow: exp(-2a) is a true zero well before that.
	if a >= expNeg2Zero {
		return 0
	}

	y := -2 * a

	// a >= tanhBranch keeps k <= -1; the k < -1022 subnormal-result path is
	// still reachable for a of a few hundred.
	k := int(math.Floor(y*invLn2 + 0.5))
	r := y - float64(k)*ln2

	p := expNeg2Poly(r, prec)

	if k < -1022 {
		return math.Ldexp(p, k)
	}

	return p * math.Float64frombits(uint64(k+1023)<<52)
}

// expNeg2Poly evaluates a truncated Taylor series for exp(r), |r| <= ln2/2.
//
// Term counts are chosen from what Tanh and LogCosh need, not from the generic
// Exp ladder: the balanced tier must land below 1e-7 absolute in tanh, and
// tanh's sensitivity to a relative error eps in u is at most 0.35*eps over the
// exponential branch, so ~5e-9 relative here is comfortably inside budget.
//
//nolint:varnamelen
func expNeg2Poly(r float64, prec Precision) float64 {
	switch normalizePrecision(prec) {
	case PrecisionFast:
		// Through r^4/4!; truncation ~4e-5 relative.
		return 1 + r*(1+r*(0.5+r*(1.0/6.0+r*(1.0/24.0))))
	case PrecisionHigh:
		// Through r^9/9!; truncation ~7e-12 relative.
		return 1 + r*(1+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0+
			r*(1.0/720.0+r*(1.0/5040.0+r*(1.0/40320.0+r*(1.0/362880.0)))))))))
	case PrecisionAuto, PrecisionBalanced:
		fallthrough
	default:
		// Through r^7/7!; truncation ~5e-9 relative.
		return 1 + r*(1+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0+
			r*(1.0/720.0+r*(1.0/5040.0)))))))
	}
}
