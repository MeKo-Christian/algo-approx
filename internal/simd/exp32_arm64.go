//go:build arm64 && !purego

package simd

import "github.com/cwbudde/algo-approx/internal/cpu"

// expBatch32NEON computes exp elementwise over min(len(dst), len(src)) elements
// using NEON, and reports whether it did the work.
//
// The bool is the same decline contract the amd64 kernel uses: a false return
// means the caller must fall back. This implementation never declines, but the
// signature keeps the call site honest if a future variant grows a shape
// restriction.
//
// Unlike the AVX2 kernel there is no instruction here that can fault for want
// of a CPU feature: Advanced SIMD is mandatory in ARMv8-A, and every
// instruction used is base ASIMD with no optional extension behind it. The
// dispatch flag below therefore exists to honour ForceGeneric, not to avoid a
// SIGILL.
//
//go:noescape
func expBatch32NEON(dst, src []float32) bool

// expUseNEON caches the dispatch decision.
//
// The feature query is resolved once here rather than per call because
// cpu.DetectFeatures takes two mutexes; paying that on every ExpFloat32 would
// dwarf the arithmetic for short slices.
//
//nolint:gochecknoglobals // dispatch decision, resolved once.
var expUseNEON bool

//nolint:gochecknoinits // one-shot CPU dispatch; there is nowhere else to put it.
func init() {
	features := cpu.DetectFeatures()
	expUseNEON = features.HasNEON && !features.ForceGeneric
}

// expBatch32 is the dispatching entry point used by ExpFloat32.
func expBatch32(dst, src []float32) {
	if expUseNEON && expBatch32NEON(dst, src) {
		return
	}

	expBatch32Go(dst, src)
}
