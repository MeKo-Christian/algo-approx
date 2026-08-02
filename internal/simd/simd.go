// Package simd holds the batch (slice-at-a-time) kernels.
//
// # Why this package exists
//
// The scalar API in the parent package is generic and evaluates every function
// in float64, converting on the way in and out. That is the right trade for a
// general-purpose call, but it is the wrong denominator for a vectorisation
// decision: a float64 kernel wrapped in float32 conversions is roughly half the
// throughput a scalar float32 kernel reaches, so measuring an AVX2 kernel
// against it would flatter the AVX2 kernel by ~2x before a single vector
// instruction ran.
//
// Everything here is therefore float32-native: no widening to float64, no
// math.* calls in the inner loop, no branches on data, and enough independent
// work in flight that the loop is limited by issue throughput rather than by
// the latency of the serial Horner chain. This is the honest pure-Go baseline
// that an assembly implementation has to beat.
//
// # Dispatch
//
// ExpFloat32 runs a hand-written AVX2+FMA kernel on amd64 hosts that have both
// features, and the pure-Go kernel everywhere else. The decision is made once
// in an init and cached; build with -tags purego to compile the assembly out
// entirely. The two kernels implement the identical algorithm and agree to
// within 1 ulp — they are not bit-identical only because the assembly kernel
// contracts its multiply-adds into FMAs and the Go compiler, whose amd64
// baseline has no FMA, does not.
//
// # Aliasing
//
// For every function in this package the destination and source slices must be
// either identical (in-place, dst == src, which is supported and tested) or
// non-overlapping. Partial overlap is undefined: the pure-Go kernels process
// elements in blocks and the SIMD kernels will read a whole vector before
// writing any of it, so a shifted alias would observe a mixture of old and new
// values. Passing a partially overlapping pair is a programming error, not a
// supported mode.
//
// # Accuracy
//
// Measured against float64 references rounded once to float32, over the whole
// representable domain of each function:
//
//	exp      within 1 ulp, including the subnormal tail and the overflow edge
//	tanh     within 1 ulp
//	logCosh  within 2 ulp, except 4 ulp just above |x| = 0.625
//
// See the tests, which pin these bounds, and logCoshLarge32 for where the
// 4 ulp comes from.
//
// # Lengths
//
// The exported wrappers panic if a destination is shorter than the source; the
// number of elements processed is always len(src).
package simd

// ExpFloat32 writes exp(src[i]) to dst[i] for every element of src.
//
// dst must be at least as long as src. See the package doc for the aliasing
// rule.
//
// The panic below names approx.FastExpBatch32 rather than this function, and
// that is deliberate: this package is internal, so the only way a caller can
// reach this panic is through that public entry point, and a message naming a
// package the caller cannot import would send them looking in the wrong place.
func ExpFloat32(dst, src []float32) {
	if len(dst) < len(src) {
		panic("approx: FastExpBatch32: dst shorter than src")
	}

	n := len(src)

	expBatch32(dst[:n], src[:n])
}

// TanhLogCoshFloat32 writes tanh(src[i]) to dstTanh[i] and log(cosh(src[i])) to
// dstLogCosh[i] for every element of src.
//
// Both destinations must be at least as long as src. The two outputs are
// produced by a single fused pass that shares the exponential; see
// tanhLogCoshBatch32Go for why that sharing is load-bearing rather than merely
// an optimisation. See the package doc for the aliasing rule, which applies to
// both destinations independently.
func TanhLogCoshFloat32(dstTanh, dstLogCosh, src []float32) {
	if len(dstTanh) < len(src) {
		panic("approx: FastTanhLogCoshBatch32: dstTanh shorter than src")
	}

	if len(dstLogCosh) < len(src) {
		panic("approx: FastTanhLogCoshBatch32: dstLogCosh shorter than src")
	}

	n := len(src)

	tanhLogCoshBatch32(dstTanh[:n], dstLogCosh[:n], src[:n])
}
