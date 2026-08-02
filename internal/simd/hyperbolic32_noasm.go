//go:build !amd64 || purego

package simd

// tanhLogCoshBatch32 is the dispatching entry point used by TanhLogCoshFloat32.
// On platforms without an assembly kernel, and under -tags purego, it is the
// pure-Go kernel.
func tanhLogCoshBatch32(dstTanh, dstLogCosh, src []float32) {
	tanhLogCoshBatch32Go(dstTanh, dstLogCosh, src)
}
