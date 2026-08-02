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
