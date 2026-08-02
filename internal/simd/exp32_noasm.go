//go:build !amd64 || purego

package simd

// expBatch32 is the dispatching entry point used by ExpFloat32. On
// architectures without a hand-written kernel, and under the purego build tag,
// there is nothing to dispatch to.
func expBatch32(dst, src []float32) {
	expBatch32Go(dst, src)
}
