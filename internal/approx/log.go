package approx

import "math"

// Log returns an approximate natural logarithm ln(x).
//
// This is a thin generic shim over the concrete float64 kernel, and
// deliberately so: a generic body is compiled once per gcshape and called
// through a dictionary from other packages, which costs a real call frame that
// no caller can inline away. Keeping the arithmetic in a non-generic function
// means a cross-package caller inlines this wrapper and lands on a direct call.
// See the cross-module benchmarks in consumerbench/.
func Log[T Float](x T, prec Precision) T {
	return T(log64(float64(x), prec))
}

//nolint:funlen,varnamelen
func log64(x float64, prec Precision) float64 {
	// Edge cases.
	if math.IsNaN(x) {
		return x
	}

	if x == 0 {
		return math.Inf(-1)
	}

	if x < 0 {
		return math.NaN()
	}

	if math.IsInf(x, 1) {
		return math.Inf(1)
	}

	// Fast range reduction without calling math.Frexp:
	// x = m * 2^e, with m in [0.5, 1).
	xf := x

	// Subnormals (biased exponent 0) carry no implicit leading 1, so the
	// mantissa reconstruction below would be wrong by up to ~36 nats.
	// Scale into the normal range first and pay it back in the exponent.
	subnormalShift := 0

	if xf < smallestNormalFloat64 {
		xf *= subnormalScale
		subnormalShift = subnormalScaleExp
	}

	bits := math.Float64bits(xf)
	// The 0x7ff mask bounds the value to 11 bits before the conversion, so it
	// always fits an int.
	expBits := int((bits>>52)&0x7ff) - 1023 - subnormalShift
	mant := bits & ((uint64(1) << 52) - 1)

	// m in [1,2) initially.
	m := 1.0 + float64(mant)*(1.0/(1<<52))
	e := expBits
	// Convert to [0.5,1) like Frexp.
	m *= 0.5
	e++

	// Transform to improve convergence:
	// ln(m) = 2 * ( y + y^3/3 + y^5/5 + ... ), y = (m-1)/(m+1)
	y := (m - 1) / (m + 1)
	y2 := y * y

	// Unrolled odd-power series; fewer terms for faster precision.
	sum := y
	p := y * y2

	switch normalizePrecision(prec) {
	case PrecisionFast:
		// y + y^3/3
		sum += p * (1.0 / 3.0)
	case PrecisionAuto, PrecisionBalanced:
		// y + y^3/3 + y^5/5 + y^7/7
		sum += p * (1.0 / 3.0)
		p *= y2
		sum += p * (1.0 / 5.0)
		p *= y2
		sum += p * (1.0 / 7.0)
	case PrecisionHigh:
		// y + y^3/3 + y^5/5 + y^7/7 + y^9/9 + y^11/11
		sum += p * (1.0 / 3.0)
		p *= y2
		sum += p * (1.0 / 5.0)
		p *= y2
		sum += p * (1.0 / 7.0)
		p *= y2
		sum += p * (1.0 / 9.0)
		p *= y2
		sum += p * (1.0 / 11.0)
	default:
		// Balanced: y + y^3/3 + y^5/5 + y^7/7
		sum += p * (1.0 / 3.0)
		p *= y2
		sum += p * (1.0 / 5.0)
		p *= y2
		sum += p * (1.0 / 7.0)
	}

	lnm := 2 * sum

	return lnm + float64(e)*ln2
}

const (
	ln2 = 0.693147180559945309417232121458176568

	// smallestNormalFloat64 is 2^-1022, the smallest positive normal float64.
	// Anything strictly below it is subnormal.
	smallestNormalFloat64 = 2.2250738585072014e-308

	// subnormalScale (2^54) lifts any positive subnormal into the normal range
	// without losing a bit; subnormalScaleExp is the exponent it adds.
	subnormalScale    = 1 << 54
	subnormalScaleExp = 54
)
