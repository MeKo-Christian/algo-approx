//go:build arm64 && !purego

package simd

// tanhLogCoshBatch32NEON computes tanh and log(cosh) elementwise over
// min(len(dstTanh), len(dstLogCosh), len(src)) elements using NEON, and reports
// whether it did the work.
//
// The bool is the same decline contract the amd64 kernel uses: a false return
// means the caller must fall back. This implementation never declines, but the
// signature keeps the call site honest if a future variant grows a shape
// restriction.
//
// Dispatch through tanhLogCoshBatch32 rather than calling this directly.
//
//go:noescape
func tanhLogCoshBatch32NEON(dstTanh, dstLogCosh, src []float32) bool

// tanhLogCoshBatch32 is the dispatching entry point used by TanhLogCoshFloat32.
//
// It reuses expUseNEON: both kernels are gated on the same feature, because the
// fused kernel's exponential IS the exp kernel's, shared through EXPBODY in
// exp32_arm64.h. A second identical flag would be one more thing to keep in
// step for no benefit.
func tanhLogCoshBatch32(dstTanh, dstLogCosh, src []float32) {
	if expUseNEON && tanhLogCoshBatch32NEON(dstTanh, dstLogCosh, src) {
		return
	}

	tanhLogCoshBatch32Go(dstTanh, dstLogCosh, src)
}
