//go:build amd64 && !purego

package simd

import "github.com/cwbudde/algo-approx/internal/cpu"

// expBatch32AVX2 computes exp elementwise over min(len(dst), len(src)) elements
// using AVX2 and FMA3, and reports whether it did the work.
//
// The bool is the same decline contract the FFT kernels use: a false return
// means the caller must fall back. This implementation never declines, but the
// signature keeps the call site honest if a future variant grows a shape
// restriction.
//
// It must not be called unless HasAVX2 && HasFMA; the FMA opcodes fault with
// SIGILL otherwise. Dispatch through expBatch32 rather than calling this
// directly.
//
//go:noescape
func expBatch32AVX2(dst, src []float32) bool

// expUseAVX2 caches the dispatch decision.
//
// The feature query is resolved once here rather than per call because
// cpu.DetectFeatures takes two mutexes; paying that on every ExpFloat32 would
// dwarf the arithmetic for short slices.
//
//nolint:gochecknoglobals // dispatch decision, resolved once.
var expUseAVX2 bool

//nolint:gochecknoinits // one-shot CPU dispatch; there is nowhere else to put it.
func init() {
	features := cpu.DetectFeatures()
	expUseAVX2 = features.HasAVX2 && features.HasFMA && !features.ForceGeneric
}

// expBatch32 is the dispatching entry point used by ExpFloat32.
func expBatch32(dst, src []float32) {
	if expUseAVX2 && expBatch32AVX2(dst, src) {
		return
	}

	expBatch32Go(dst, src)
}
