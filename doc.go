// Package approx provides fast, allocation-free mathematical approximations.
//
// The API is generic over float32 and float64 using the Float constraint.
//
// # Batch functions
//
// The FastXBatch32 / FastXBatch64 entry points process a whole slice per call.
// They exist because that is where the library's measured performance is: the
// scalar functions are roughly break-even with math, while the float32 batch
// path runs a hand-written AVX2+FMA kernel and measures ~11x faster per element
// than a scalar loop. The following rules apply to all of them, and each
// function's own doc refers back here rather than repeating them.
//
// Lengths: a batch function panics if any destination is shorter than src. The
// number of elements processed is always len(src); destinations may be longer,
// and the elements past len(src) are left untouched.
//
// Aliasing: each destination must be either identical to src (in-place,
// dst == src, which is supported and tested) or non-overlapping with it.
// Partial overlap is undefined. This is not a hedge — the SIMD kernels read a
// whole eight-element vector before writing any of it, so a shifted alias
// observes a mixture of old and new values. Passing a partially overlapping
// pair is a programming error, not a supported mode, and a test using two
// distinct slices will never catch it.
//
// Batch functions take no Precision argument. Resolving a precision tier per
// element costs more than the polynomial it selects (see the measurements in
// AGENTS.md), so the tier is fixed at PrecisionBalanced and constant-folded;
// there are deliberately no ...BatchPrec variants.
//
// The float64 batch functions are scalar loops over the same kernels the scalar
// API uses and are bit-identical to them. The float32 batch functions are not:
// they run a float32-native minimax kernel where the scalar API widens to
// float64, which makes them both faster and more accurate. See ACCURACY.md.
package approx
