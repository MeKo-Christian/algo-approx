// Package consumerbench benchmarks algo-approx the way a real caller sees it.
//
// It is a separate Go module (see go.mod) that imports algo-approx by module
// path, so nothing here gets the same-package inlining that flatters the
// in-package benchmarks. That is the only configuration in which the library's
// cross-module inlining property is observable at all.
//
// This file holds nothing but call sites. It exists so that inline_test.go can
// compile a plain (non-test) package with -gcflags=-m and assert on the
// compiler's inlining decisions without depending on the shape or the line
// numbering of the benchmark file next to it.
package consumerbench

import approx "github.com/cwbudde/algo-approx"

// The generic entry points. Calling these from outside the module is what
// forces the go.shape instantiation of the internal shim to be compiled into
// *this* package, which is what inline_test.go inspects.

func CallFastLog(x float64) float64     { return approx.FastLog(x) }
func CallFastExp(x float64) float64     { return approx.FastExp(x) }
func CallFastTanh(x float64) float64    { return approx.FastTanh(x) }
func CallFastLogCosh(x float64) float64 { return approx.FastLogCosh(x) }

// The concrete float64 entry points.

func CallFastLog64(x float64) float64     { return approx.FastLog64(x) }
func CallFastExp64(x float64) float64     { return approx.FastExp64(x) }
func CallFastTanh64(x float64) float64    { return approx.FastTanh64(x) }
func CallFastLogCosh64(x float64) float64 { return approx.FastLogCosh64(x) }

// The batch entry points.
//
// These are here for the same reason as everything above -- so the public
// surface is compiled from a real consumer module rather than only from inside
// the library -- but they are deliberately *not* in inline_test.go's assertion
// list. A batch body is a loop over a whole slice; it is far past the inline
// budget and will never be inlined, and it does not need to be. One call
// amortises over thousands of elements, which is the entire premise of the
// batch API. TestCrossModuleInlining must stay green on exactly its original
// four operations: adding these would turn the repo's core regression gate into
// a permanently failing one.

func CallFastExpBatch32(dst, src []float32) { approx.FastExpBatch32(dst, src) }
func CallFastExpBatch64(dst, src []float64) { approx.FastExpBatch64(dst, src) }

func CallFastTanhLogCoshBatch32(dstTanh, dstLogCosh, src []float32) {
	approx.FastTanhLogCoshBatch32(dstTanh, dstLogCosh, src)
}

func CallFastTanhLogCoshBatch64(dstTanh, dstLogCosh, src []float64) {
	approx.FastTanhLogCoshBatch64(dstTanh, dstLogCosh, src)
}
