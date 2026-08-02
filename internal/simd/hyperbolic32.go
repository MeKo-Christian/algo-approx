package simd

import "math"

// Shared shape of the fused tanh / log(cosh) kernel.
//
// tanh is exactly d/dx log(cosh(x)), and the consumers of this pair depend on
// that identity surviving approximation. It survives because both outputs are
// built from ONE quantity, u = exp(-2|x|):
//
//	tanh(|x|)  = 1 - 2u/(1+u)
//	logCosh(x) = |x| - ln2 + log1p(u)
//
// and the second differentiates into the first for any u whatsoever. Two
// separate approximations that each happened to be accurate would not give
// that; sharing u does. The fused kernel evaluates the exponential once and
// feeds both results from it, which is also why fusing is cheaper than two
// passes rather than merely more convenient.
//
// Both branches are always evaluated and combined with a bitmask select. That
// is the cost this baseline is meant to expose: a branchy version would look
// fast on smooth data, fall apart on mixed data, and is not what a vector
// kernel would be doing.
const (
	// tanhBranch32 is the switch point between the rational core and the
	// exponential form, the same 0.625 the scalar kernels use. Below it,
	// (1-u)/(1+u) cancels badly as u -> 1; above it the rational core would
	// need far more terms. Keeping the constant identical to the scalar path
	// means the derivative identity never straddles a different seam.
	tanhBranch32 = 0.625

	// ln2f is ln2 rounded to float32 (0x3f317218).
	ln2f = 0.6931471805599453

	// signMask32 is the binary32 sign bit.
	signMask32 = 0x80000000
)

// Rational core for tanh on |x| < 0.625, the classic Cephes [7/6] form
//
//	tanh(a) = a + a*z*P(z)/Q(z),  z = a*a
//
// evaluated in float32. The odd symmetry and the a -> 0 behaviour are
// structural in this form rather than fitted, and the underlying rational is
// accurate to ~2e-17, far below a float32 ulp, so on this branch the float32
// rounding of the coefficients and of the evaluation is the entire error.
const (
	tanhP0 = -9.64399179425052238628e-1
	tanhP1 = -9.92877231001918586564e+1
	tanhP2 = -1.61468768441708447952e+3

	tanhQ0 = +1.12811678491632931402e+2
	tanhQ1 = +2.23548839060100448583e+3
	tanhQ2 = +4.84406305325125486048e+3
)

// Taylor coefficients of log(cosh(x)) in z = x*x:
//
//	log(cosh(x)) = z * (c0 + c1*z + c2*z^2 + ...)
//
// These are the term-by-term antiderivative of the tanh series
// a - a^3/3 + 2a^5/15 - ..., not an independent fit, so the small branch of
// logCosh differentiates back into the small branch of tanh coefficient for
// coefficient. The series is carried to z^8, where the first dropped term is
// ~5e-9 relative at the branch point, comfortably under a float32 ulp; going
// further would cost throughput in a branch that is always evaluated and buy
// nothing observable.
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
)

// abs32 returns |x| by masking the sign bit: no comparison, no branch, and it
// leaves NaN payloads alone.
func abs32(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ signMask32)
}

// signBit32 extracts the sign bit of x, ready to be XORed back onto a
// magnitude.
func signBit32(x float32) uint32 {
	return math.Float32bits(x) & signMask32
}

// withSign32 reattaches a sign bit to a magnitude.
//
// Reattaching by XOR rather than negating a computed value is what makes the
// odd symmetry of tanh BIT-EXACT: f(-x) and f(x) share every single arithmetic
// operation, because both are computed from the same |x|, so the results can
// differ in nothing but this bit.
func withSign32(v float32, sign uint32) float32 {
	return math.Float32frombits(math.Float32bits(v) ^ sign)
}

// tanhBranchMask32 returns all-ones when a < 0.625 and zero otherwise, for a
// known non-negative a.
//
// For non-negative floats the IEEE bit pattern is monotonic when read as an
// integer, so the integer comparison is the same predicate as the float one
// and it produces the mask directly, with no compare instruction and no
// branch. NaN's pattern sorts above 0.625's, so NaN takes the exponential
// branch, where it propagates to NaN in both outputs.
//
//nolint:gosec // deliberate reinterpretation to extract a sign as a mask.
func tanhBranchMask32(a float32) uint32 {
	return uint32(int32(math.Float32bits(a)-math.Float32bits(tanhBranch32)) >> 31)
}

// select32 picks small where mask is all-ones and large where it is zero.
func select32(small, large float32, mask uint32) float32 {
	return math.Float32frombits((math.Float32bits(small) & mask) | (math.Float32bits(large) &^ mask))
}

// tanhSmall32 is the rational core, valid for a in [0, 0.625).
func tanhSmall32(a float32) float32 {
	z := a * a
	num := (tanhP0*z+tanhP1)*z + tanhP2
	den := ((z+tanhQ0)*z+tanhQ1)*z + tanhQ2

	return a + a*z*(num/den)
}

// logCoshSmall32 is the antiderivative series, valid for a in [0, 0.625).
func logCoshSmall32(a float32) float32 {
	z := a * a

	return z * (logCoshC0 + z*(logCoshC1+z*(logCoshC2+z*(logCoshC3+
		z*(logCoshC4+z*(logCoshC5+z*(logCoshC6+z*(logCoshC7+
			z*logCoshC8))))))))
}

// tanhLarge32 turns the shared u = exp(-2a) into tanh(a).
//
// Written as 1 - 2u/(1+u) rather than (1-u)/(1+u) so that the subtraction
// happens against a small quantity: for a above the branch point u <= 0.2866,
// so 2u/(1+u) is at most 0.446 and the cancellation that would wreck the
// direct form near u = 1 never arises.
//
// # Saturation
//
// The scalar float64 path carries an explicit saturation constant of 19.0625,
// the point at which float64 tanh becomes indistinguishable from 1. In float32
// the crossover is at 9.010914 (0x41102cb4): that is the smallest float32
// whose tanh rounds to exactly 1.0f, its predecessor 9.010913 still giving
// 0.99999994. Carrying 19.0625 into a float32 kernel would pay for an
// exponential evaluation over the whole range 9.01 .. 19.06 that cannot
// produce anything except 1.0.
//
// This kernel spends nothing on either constant. Since both branches are
// always evaluated, an early-out saturation test would remove no work, only
// add a comparison; and this expression saturates on its own at exactly the
// right place, because 2u/(1+u) falls below half an ulp of 1 precisely when
// u drops under ~1.5e-8, i.e. at the measured crossover. The clamp inside the
// exp stages carries the rest: a >= 52 gives u = 0, so +Inf yields tanh = 1
// and logCosh = +Inf with no special case anywhere.
func tanhLarge32(u float32) float32 {
	return 1 - 2*u/(1+u)
}

// logCoshLarge32 turns a and the shared u into a - ln2 + log1p(u).
//
// log1p(u) uses the atanh form 2*atanh(w), w = u/(2+u) <= 0.1253, whose four
// odd terms leave a truncation below 2e-9 absolute, an order under a float32
// ulp of the result. The Mercator series in u would need eleven terms for the
// same, and math.Log1p would cost more than the rest of the kernel and drag
// the arithmetic back into float64.
//
// Note that cosh is never formed, so this stays finite for every finite input
// instead of overflowing near |x| = 89 the way log(cosh(x)) would.
//
// This branch, not the exponential, sets the accuracy floor of logCosh: just
// above the branch point a - ln2 is small and cancels against log1p(u), which
// costs up to 4 ulp at |x| ~ 0.626. Everywhere else logCosh is within 2 ulp,
// and tanh, which has no such subtraction, is within 1.
func logCoshLarge32(a, u float32) float32 {
	w := u / (2 + u)
	w2 := w * w

	return a - ln2f + 2*w*(1+w2*(1.0/3.0+w2*(1.0/5.0+w2*(1.0/7.0+w2*(1.0/9.0)))))
}

// tanhLogCoshKernel32 returns (tanh(x), log(cosh(x))) for one element. It is
// the scalar reference for the batch loop's hand-inlined copy and handles the
// tail.
func tanhLogCoshKernel32(x float32) (float32, float32) {
	a := abs32(x)
	sign := signBit32(x)

	u := expKernel32(-2 * a)
	mask := tanhBranchMask32(a)

	tanh := select32(tanhSmall32(a), tanhLarge32(u), mask)
	logCosh := select32(logCoshSmall32(a), logCoshLarge32(a, u), mask)

	return withSign32(tanh, sign), logCosh
}

// tanhLogCoshBatch32Go computes tanh and log(cosh) for the same inputs in one
// fused pass, float32 throughout, branchless.
//
// The block body is tanhLogCoshKernel32 written out four lanes wide, one stage
// at a time, twice, with the exponential's five stages spliced in rather than
// called: the composed kernel is far over the inline budget, so calling it per
// element would measure Go's calling convention instead of Go's arithmetic,
// and the lanes would not be schedulable against each other. See the unroll
// constant for why the block is 2x4 rather than 1x8.
//
// All three slices are resliced to a common length first so the compiler can
// prove the block indexing safe and drop the bounds checks.
//
//nolint:varnamelen,funlen // four-lane pipelining.
func tanhLogCoshBatch32Go(dstTanh, dstLogCosh, src []float32) {
	n := min(len(src), len(dstTanh), len(dstLogCosh))

	src = src[:n]
	dstTanh = dstTanh[:n]
	dstLogCosh = dstLogCosh[:n]

	i := 0

	for ; i+unroll <= n; i += unroll {
		s := src[i : i+unroll : i+unroll]
		dt := dstTanh[i : i+unroll : i+unroll]
		dl := dstLogCosh[i : i+unroll : i+unroll]

		// --- lanes 0..3 ---
		a0, a1, a2, a3 := abs32(s[0]), abs32(s[1]), abs32(s[2]), abs32(s[3])
		g0, g1, g2, g3 := signBit32(s[0]), signBit32(s[1]), signBit32(s[2]), signBit32(s[3])
		m0, m1, m2, m3 := tanhBranchMask32(a0), tanhBranchMask32(a1), tanhBranchMask32(a2), tanhBranchMask32(a3)

		// The shared exponential: clamp, rint, reduce, polynomial, scale.
		e0, e1, e2, e3 := expClamp32(-2*a0), expClamp32(-2*a1), expClamp32(-2*a2), expClamp32(-2*a3)
		k0, k1, k2, k3 := expRint32(e0), expRint32(e1), expRint32(e2), expRint32(e3)
		r0, r1, r2, r3 := expCodyWaite32(e0, k0), expCodyWaite32(e1, k1), expCodyWaite32(e2, k2), expCodyWaite32(e3, k3)
		u0, u1 := expScaleApply32(expPoly32(r0), k0), expScaleApply32(expPoly32(r1), k1)
		u2, u3 := expScaleApply32(expPoly32(r2), k2), expScaleApply32(expPoly32(r3), k3)

		dt[0] = withSign32(select32(tanhSmall32(a0), tanhLarge32(u0), m0), g0)
		dt[1] = withSign32(select32(tanhSmall32(a1), tanhLarge32(u1), m1), g1)
		dt[2] = withSign32(select32(tanhSmall32(a2), tanhLarge32(u2), m2), g2)
		dt[3] = withSign32(select32(tanhSmall32(a3), tanhLarge32(u3), m3), g3)

		dl[0] = select32(logCoshSmall32(a0), logCoshLarge32(a0, u0), m0)
		dl[1] = select32(logCoshSmall32(a1), logCoshLarge32(a1, u1), m1)
		dl[2] = select32(logCoshSmall32(a2), logCoshLarge32(a2, u2), m2)
		dl[3] = select32(logCoshSmall32(a3), logCoshLarge32(a3, u3), m3)

		// --- lanes 4..7 ---
		a4, a5, a6, a7 := abs32(s[4]), abs32(s[5]), abs32(s[6]), abs32(s[7])
		g4, g5, g6, g7 := signBit32(s[4]), signBit32(s[5]), signBit32(s[6]), signBit32(s[7])
		m4, m5, m6, m7 := tanhBranchMask32(a4), tanhBranchMask32(a5), tanhBranchMask32(a6), tanhBranchMask32(a7)

		e4, e5, e6, e7 := expClamp32(-2*a4), expClamp32(-2*a5), expClamp32(-2*a6), expClamp32(-2*a7)
		k4, k5, k6, k7 := expRint32(e4), expRint32(e5), expRint32(e6), expRint32(e7)
		r4, r5, r6, r7 := expCodyWaite32(e4, k4), expCodyWaite32(e5, k5), expCodyWaite32(e6, k6), expCodyWaite32(e7, k7)
		u4, u5 := expScaleApply32(expPoly32(r4), k4), expScaleApply32(expPoly32(r5), k5)
		u6, u7 := expScaleApply32(expPoly32(r6), k6), expScaleApply32(expPoly32(r7), k7)

		dt[4] = withSign32(select32(tanhSmall32(a4), tanhLarge32(u4), m4), g4)
		dt[5] = withSign32(select32(tanhSmall32(a5), tanhLarge32(u5), m5), g5)
		dt[6] = withSign32(select32(tanhSmall32(a6), tanhLarge32(u6), m6), g6)
		dt[7] = withSign32(select32(tanhSmall32(a7), tanhLarge32(u7), m7), g7)

		dl[4] = select32(logCoshSmall32(a4), logCoshLarge32(a4, u4), m4)
		dl[5] = select32(logCoshSmall32(a5), logCoshLarge32(a5, u5), m5)
		dl[6] = select32(logCoshSmall32(a6), logCoshLarge32(a6, u6), m6)
		dl[7] = select32(logCoshSmall32(a7), logCoshLarge32(a7, u7), m7)
	}

	for ; i < n; i++ {
		dstTanh[i], dstLogCosh[i] = tanhLogCoshKernel32(src[i])
	}
}
