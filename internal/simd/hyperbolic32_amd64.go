//go:build amd64 && !purego

package simd

// tanhLogCoshBatch32AVX2 computes tanh and log(cosh) elementwise over
// min(len(dstTanh), len(dstLogCosh), len(src)) elements using AVX2 and FMA3,
// and reports whether it did the work.
//
// The bool is the same decline contract the FFT kernels use: a false return
// means the caller must fall back. This implementation never declines, but the
// signature keeps the call site honest if a future variant grows a shape
// restriction.
//
// It must not be called unless HasAVX2 && HasFMA; the FMA opcodes fault with
// SIGILL otherwise. Dispatch through tanhLogCoshBatch32 rather than calling
// this directly.
//
//go:noescape
func tanhLogCoshBatch32AVX2(dstTanh, dstLogCosh, src []float32) bool

// tanhLogCoshBatch32 is the dispatching entry point used by TanhLogCoshFloat32.
//
// It reuses expUseAVX2: both kernels are gated on exactly the same pair of CPU
// features, because the fused kernel's exponential IS the exp kernel's, shared
// through EXPBODY in exp32_amd64.h. A second identical flag would be one more
// thing to keep in step for no benefit.
func tanhLogCoshBatch32(dstTanh, dstLogCosh, src []float32) {
	if expUseAVX2 && tanhLogCoshBatch32AVX2(dstTanh, dstLogCosh, src) {
		return
	}

	tanhLogCoshBatch32Go(dstTanh, dstLogCosh, src)
}
