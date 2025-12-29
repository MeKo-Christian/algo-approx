package approx

import "math"

// Log returns an approximate natural logarithm ln(x).
func Log[T Float](x T, prec Precision) T {
	// Edge cases.
	if x != x {
		return x
	}
	if x == 0 {
		return T(math.Inf(-1))
	}
	if x < 0 {
		return T(math.NaN())
	}
	if math.IsInf(float64(x), 1) {
		return T(math.Inf(1))
	}

	// Fast range reduction without calling math.Frexp:
	// x = m * 2^e, with m in [0.5, 1).
	xf := float64(x)
	bits := math.Float64bits(xf)
	expBits := int((bits>>52)&0x7ff) - 1023
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
	return T(lnm + float64(e)*ln2)
}

const ln2 = 0.693147180559945309417232121458176568
