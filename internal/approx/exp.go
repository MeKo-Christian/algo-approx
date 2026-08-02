package approx

import "math"

// Exp returns an approximate exponential e^x.
//
// A thin generic shim over the concrete float64 kernel; see Log for why.
func Exp[T Float](x T, prec Precision) T {
	return T(exp64(float64(x), prec))
}

func exp64(xflt float64, prec Precision) float64 {
	// Edge cases.
	if math.IsNaN(xflt) {
		return xflt
	}

	if math.IsInf(xflt, 1) {
		return math.Inf(1)
	}

	if math.IsInf(xflt, -1) {
		return 0
	}

	// Clamp to float64 overflow bounds.
	if xflt > maxLogFloat64 {
		return math.Inf(1)
	}

	if xflt < minLogFloat64 {
		return 0
	}

	// Range reduction: x = k*ln2 + r, r in roughly [-ln2/2, ln2/2].
	k := int(math.Floor(xflt*invLn2 + 0.5))
	r := xflt - float64(k)*ln2

	expr := expPoly(r, normalizePrecision(prec))

	// Faster scaling than math.Ldexp for the common normal range.
	// 2^k is exactly representable as a float64 when k is within the normal exponent range.
	var res float64

	if k > -1023 && k < 1024 {
		pow2k := math.Float64frombits(uint64(k+1023) << 52) //nolint:gosec
		res = expr * pow2k
	} else {
		res = math.Ldexp(expr, k)
	}

	return res
}

//nolint:varnamelen
func expPoly(r float64, prec Precision) float64 {
	// Evaluate truncated Taylor polynomial via Horner.
	switch prec {
	case PrecisionFast:
		// 1 + r + r^2/2 + r^3/6
		return 1 + r*(1+r*(0.5+r*(1.0/6.0)))
	case PrecisionAuto, PrecisionBalanced:
		// up to r^5/5!
		return 1 + r*(1+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0)))))
	case PrecisionHigh:
		// up to r^7/7!
		return 1 + r*(1+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0+r*(1.0/720.0+r*(1.0/5040.0)))))))
	default:
		// up to r^5/5!
		return 1 + r*(1+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0)))))
	}
}

const (
	// Natural-log bounds for float64 exp overflow/underflow.
	maxLogFloat64 = 709.782712893384
	minLogFloat64 = -745.133219101941
	invLn2        = 1.442695040888963407359924681001892137
)
