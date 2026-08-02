package approx

import "math"

// Batch (slice-at-a-time) kernels for float64.
//
// There is no float64 SIMD kernel and there is not going to be one: the
// float32 assembly wins eight lanes at a time, a float64 kernel would win four,
// and the scalar float64 kernels here already measure at or below math's cost
// (PLAN.md 6.0). What these loops buy is therefore not vectorisation but
// amortisation — one call frame and one precision resolution for the whole
// slice instead of one per element. For ExpBatch64 that is nearly all of it;
// for TanhLogCoshBatch64 the fused pass additionally shares the expensive
// u = exp(-2|x|) between the two outputs, which is the same argument that
// justifies the fused float32 kernel and it applies unchanged at float64.
//
// # Precision
//
// batchPrecision is a compile-time constant rather than a parameter, and that
// is deliberate: see the "Precision must not be a runtime argument in a loop"
// section of AGENTS.md. Passing prec as a runtime value defeats constant
// folding of the switch in expPoly/logCoshPoly/log1pSmall, and the dispatch
// then costs more than the polynomial it selects (measured: 12.2/14.7/16.8 ns
// against 5.6 ns). The batch entry points therefore ship no ...Prec variant,
// and the tier is resolved here, once, at compile time.
const batchPrecision = PrecisionBalanced

// ExpBatch64 writes exp(src[i]) to dst[i] for every element of src.
//
// dst must be at least as long as src; the number of elements processed is
// always len(src). See the aliasing rule in the root package doc.
func ExpBatch64(dst, src []float64) {
	if len(dst) < len(src) {
		panic("approx: FastExpBatch64: dst shorter than src")
	}

	dst = dst[:len(src)]

	for i, x := range src {
		dst[i] = exp64(x, batchPrecision)
	}
}

// TanhLogCoshBatch64 writes tanh(src[i]) to dstTanh[i] and log(cosh(src[i])) to
// dstLogCosh[i] for every element of src.
//
// Both destinations must be at least as long as src; the number of elements
// processed is always len(src). See the aliasing rule in the root package doc,
// which applies to each destination independently.
//
// The two outputs come from one fused pass that shares u = exp(-2|x|); see
// tanhLogCosh64.
func TanhLogCoshBatch64(dstTanh, dstLogCosh, src []float64) {
	if len(dstTanh) < len(src) {
		panic("approx: FastTanhLogCoshBatch64: dstTanh shorter than src")
	}

	if len(dstLogCosh) < len(src) {
		panic("approx: FastTanhLogCoshBatch64: dstLogCosh shorter than src")
	}

	n := len(src)
	dstTanh = dstTanh[:n]
	dstLogCosh = dstLogCosh[:n]

	for i, x := range src {
		t, l := tanhLogCosh64(x, batchPrecision)
		dstTanh[i] = t
		dstLogCosh[i] = l
	}
}

// tanhLogCosh64 returns tanh(x) and log(cosh(x)) from a single pass.
//
// This is tanh64 and logCosh64 interleaved, not a reimplementation: it reads
// the same branch point (tanhBranch), the same rational core, the same
// logCoshPoly and log1pSmall, and above the branch point the same single
// u = expNeg2(a, prec). Sharing that u is what keeps tanh exactly
// d/dx log(cosh x) — the invariant a downstream discrete-gradient energy scheme
// depends on — in the batch path as well as the scalar one. Every result here
// is bit-for-bit identical to the corresponding scalar kernel, and a test pins
// that.
//
// Two edge cases are carried by the arithmetic rather than by a branch:
//
//   - a = +Inf. expNeg2 returns 0 (it bails out above expNeg2Zero), so
//     a - ln2 + log1pSmall(0) is +Inf - ln2 + 0 = +Inf, which is exactly what
//     logCosh64 returns from its explicit infinity check.
//   - -0. The magnitude is computed from |x| and the sign reattached, so
//     tanh(-0) is -0 and log(cosh(-0)) is +0, as in the scalar kernels.
//
// Returns tanh(x) first, log(cosh(x)) second.
//
//nolint:varnamelen // u is the shared exp(-2|x|); tanh.go and logcosh.go name it the same.
func tanhLogCosh64(xflt float64, prec Precision) (float64, float64) {
	if math.IsNaN(xflt) {
		return xflt, xflt
	}

	// Both functions work on the magnitude; tanh reattaches the sign last,
	// which is what makes its odd symmetry bit-exact, and log(cosh) is even.
	a := math.Abs(xflt)

	var mag, lcosh float64

	if a < tanhBranch {
		z := a * a
		num := ((tanhP0*z+tanhP1)*z + tanhP2)
		den := (((z+tanhQ0)*z+tanhQ1)*z + tanhQ2)
		mag = a + a*z*num/den
		lcosh = z * logCoshPoly(z, prec)
	} else {
		// The shared quantity. One expNeg2 feeds both outputs.
		u := expNeg2(a, prec)

		if a >= tanhSaturation {
			// Also the +Inf case; matches tanh64's exact saturation.
			mag = 1
		} else {
			mag = 1 - 2*u/(1+u)
		}

		lcosh = a - ln2 + log1pSmall(u, prec)
	}

	if math.Signbit(xflt) {
		return -mag, lcosh
	}

	return mag, lcosh
}
