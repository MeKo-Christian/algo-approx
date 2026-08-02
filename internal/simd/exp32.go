package simd

import "math"

// float32 constants for the exp kernel.
//
// Every one of these is quoted as the decimal that rounds to the stated float32
// bit pattern; TestExpConstantBits pins each against its hex so a future edit
// cannot silently perturb one by an ulp.
const (
	// expLog2e = 1/ln2, the scale that turns x into the exponent k.
	expLog2e = 1.44269504088896341 // 0x3fb8aa3b

	// expC1 and expC2 are the Cody-Waite split of ln2: ln2 = C1 + C2 with C1
	// carrying only 9 significant mantissa bits, so fx*C1 is *exact* in
	// float32 for every fx this kernel can produce (|fx| <= 150 needs 8 bits,
	// and 8+9 <= 24). The subtraction x - fx*C1 is then exact as well, so the
	// only rounding in the whole range reduction is the second step.
	//
	// The single-step reduction r = x - fx*ln2f is not an acceptable
	// substitute: it rounds fx*ln2f to float32 first, and at x ~ 88 that
	// rounding is large relative to r, which shows up as roughly 22 ulp of
	// error in the result.
	expC1 = 0.693359375    // 0x3f318000
	expC2 = -2.12194440e-4 // 0xb95e8083

	// Minimax coefficients for exp(r) on |r| <= ln2/2, in the form
	// exp(r) = 1 + r + r*r*P(r).
	expP4 = 0.0013814628 // 0x3ab51233
	expP3 = 0.008368711  // 0x3c091ceb
	expP2 = 0.041668389  // 0x3d2aac79
	expP1 = 0.16666521   // 0x3e2aaa49
	expP0 = 0.49999994   // 0x3efffffe

	// expHI is ln(MaxFloat32) rounded to float32. Clamping to it maps every
	// larger input (including +Inf) onto k = 128, whose two-step scaling
	// overflows to +Inf, which is what exp must return there.
	expHI = 88.7228391 // 0x42b17218

	// expLO must be exactly -104. It is the largest round number at which
	// exp(x) still rounds to a true float32 zero: exp(-104) = 6.8136e-46,
	// just under half the smallest subnormal (7.0065e-46), so it flushes,
	// while exp(-103.28) = 1.0e-45 does not. Clamping any higher (the
	// tempting -87.34, i.e. ln(SmallestNormalFloat32)) would wrongly flush the
	// entire subnormal band to zero; clamping lower is pointless because
	// nothing below -104 can produce a nonzero result.
	expLO = -104.0 // 0xc2d00000

	// expRoundMagic = 1.5 * 2^23. Adding it to a float32 of magnitude below
	// 2^22 lands the sum in the binade whose ulp is exactly 1, so the
	// hardware's round-to-nearest-even performs the rint; subtracting it back
	// is exact. Branchless, and unlike a truncating int conversion it rounds
	// the right way for negative arguments.
	expRoundMagic = 12582912.0 // 0x4b400000

	// expBias is the IEEE-754 binary32 exponent bias.
	expBias = 127

	// expMantBits is the width of the binary32 significand field.
	expMantBits = 23
)

// unroll is the block size of every batch loop here. Each block is worked as
// two groups of four independent lanes.
//
// Four independent chains already cover the ~4-cycle latency of a float32
// multiply-add on any contemporary core, so the loop is bound by issue
// throughput rather than by the serial dependency inside one Horner
// evaluation. That distinction is the entire point of this file: a
// latency-bound baseline would understate pure Go by several times and make
// any vector kernel look better than it is.
//
// The block is two groups of four rather than one group of eight because x86
// has sixteen XMM registers: eight lanes keep sixteen values live across the
// reduction with nothing left for the constants, and the register allocator
// starts spilling to the stack. Measured on the generated code, 2x4 costs
// ~63 instructions per element against ~67 for 1x8, and the out-of-order
// window overlaps the two groups anyway.
const unroll = 8

// The kernel is written as five small stages rather than one function on
// purpose. A single function costs 139 inline-budget units against a budget of
// 80, so it would be a real call per element: eight calls per block, no
// scheduling across lanes, and a baseline that measures Go's calling
// convention instead of Go's arithmetic. Each stage below is under budget and
// therefore inlines, which lets the batch loop below interleave eight
// independent copies of the whole chain.

// expClamp32 pins x into [expLO, expHI].
//
// This must happen before anything else. If +Inf reached the range reduction,
// fx would be +Inf and r = Inf - Inf = NaN, so a perfectly ordinary +Inf input
// would come back NaN. Clamping first turns +Inf into the ordinary overflow
// path.
//
// The builtin min and max are used rather than a pair of ifs on purpose. The
// ifs are two data-dependent conditional branches per element; min and max
// compile to a straight-line MINSS/MAXSS sequence with no branch at all, which
// is both what the requirement asks for and what the vector kernel this is a
// baseline for would do. They also give exactly the NaN behaviour wanted: Go
// specifies that min and max propagate NaN, so a NaN input survives the clamp
// and comes out of the polynomial as NaN.
func expClamp32(x float32) float32 {
	return min(expHI, max(expLO, x))
}

// expRint32 returns rint(x * log2e), i.e. the reduction exponent as a float,
// rounded to nearest with ties to even.
func expRint32(x float32) float32 {
	return (x*expLog2e + expRoundMagic) - expRoundMagic
}

// expCodyWaite32 returns the reduced argument r = x - fx*ln2, split over two
// steps; see expC1 and expC2 for why one step is not enough.
func expCodyWaite32(x, fx float32) float32 {
	r := x - fx*expC1

	return r - fx*expC2
}

// expPoly32 evaluates exp(r) for |r| <= ln2/2.
func expPoly32(r float32) float32 {
	z := r * r
	p := (((expP4*r+expP3)*r+expP2)*r+expP1)*r + expP0

	return p*z + r + 1.0
}

// expScaleApply32 returns val * 2^k where k = int32(fx), built from TWO
// half-exponent factors.
//
// A single (k+bias)<<23 breaks at both ends of the range this kernel must
// cover:
//
//	k = 128  -> biased 255, which is the Inf/NaN encoding, so every input in
//	            roughly (88.03, 88.72] would return +Inf instead of the
//	            perfectly representable finite result near MaxFloat32.
//	k = -150 -> biased -23, whose two's-complement pattern sets the sign bit
//	            and a large exponent: garbage of the wrong sign rather than
//	            the intended flush to zero.
//
// Splitting k in half keeps both biased exponents inside [1, 254] for every k
// in [-150, 128], so each factor is an ordinary finite power of two. The
// result is reached in two multiplications, of which only the last can round,
// so a subnormal result such as exp(-90) is rounded exactly once and is
// correct to half an ulp of the subnormal grid.
//
//nolint:gosec // the conversions are deliberate bit-pattern construction.
func expScaleApply32(val, fx float32) float32 {
	k := int32(fx)
	k1 := k >> 1
	k2 := k - k1

	f1 := math.Float32frombits(uint32(k1+expBias) << expMantBits)
	f2 := math.Float32frombits(uint32(k2+expBias) << expMantBits)

	return val * f1 * f2
}

// expKernel32 is one float32 exp, branchless end to end. It is the scalar
// reference for the batch loop's hand-inlined copy and handles the tail.
func expKernel32(x float32) float32 {
	xc := expClamp32(x)
	fx := expRint32(xc)

	return expScaleApply32(expPoly32(expCodyWaite32(xc, fx)), fx)
}

// expBatch32Go computes exp elementwise, float32 in and float32 out, with no
// intermediate float64 and no data-dependent branch.
//
// The block body is the five stages of expKernel32 written out four lanes
// wide, one stage at a time, twice. Written that way the four chains are
// visibly independent and the scheduler can keep the multiply pipeline full;
// written as eight sequential calls to expKernel32 it would be a call per
// element and latency-bound on a chain roughly a dozen operations deep, which
// is the failure mode this baseline exists to avoid.
//
// dst and src are resliced to a common length first so the compiler can prove
// the block indexing safe and drop the bounds checks.
//
//nolint:varnamelen // four-lane pipelining.
func expBatch32Go(dst, src []float32) {
	n := min(len(src), len(dst))

	src = src[:n]
	dst = dst[:n]

	i := 0

	for ; i+unroll <= n; i += unroll {
		s := src[i : i+unroll : i+unroll]
		d := dst[i : i+unroll : i+unroll]

		x0, x1, x2, x3 := expClamp32(s[0]), expClamp32(s[1]), expClamp32(s[2]), expClamp32(s[3])
		k0, k1, k2, k3 := expRint32(x0), expRint32(x1), expRint32(x2), expRint32(x3)
		r0, r1, r2, r3 := expCodyWaite32(x0, k0), expCodyWaite32(x1, k1), expCodyWaite32(x2, k2), expCodyWaite32(x3, k3)
		y0, y1, y2, y3 := expPoly32(r0), expPoly32(r1), expPoly32(r2), expPoly32(r3)

		d[0], d[1], d[2], d[3] = expScaleApply32(y0, k0), expScaleApply32(y1, k1),
			expScaleApply32(y2, k2), expScaleApply32(y3, k3)

		x4, x5, x6, x7 := expClamp32(s[4]), expClamp32(s[5]), expClamp32(s[6]), expClamp32(s[7])
		k4, k5, k6, k7 := expRint32(x4), expRint32(x5), expRint32(x6), expRint32(x7)
		r4, r5, r6, r7 := expCodyWaite32(x4, k4), expCodyWaite32(x5, k5), expCodyWaite32(x6, k6), expCodyWaite32(x7, k7)
		y4, y5, y6, y7 := expPoly32(r4), expPoly32(r5), expPoly32(r6), expPoly32(r7)

		d[4], d[5], d[6], d[7] = expScaleApply32(y4, k4), expScaleApply32(y5, k5),
			expScaleApply32(y6, k6), expScaleApply32(y7, k7)
	}

	for ; i < n; i++ {
		dst[i] = expKernel32(src[i])
	}
}
