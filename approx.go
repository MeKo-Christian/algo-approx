package approx

import (
	iapprox "github.com/cwbudde/algo-approx/internal/approx"
	"github.com/cwbudde/algo-approx/internal/simd"
)

// FastLog returns an approximate natural logarithm ln(x) using the default precision.
func FastLog[T Float](x T) T { return FastLogPrec(x, PrecisionAuto) }

// FastLogPrec returns an approximate natural logarithm ln(x) using the requested precision.
func FastLogPrec[T Float](x T, prec Precision) T {
	return iapprox.Log(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastLog32(x float32) float32 { return FastLog[float32](x) }
func FastLog64(x float64) float64 { return FastLog[float64](x) }

// FastExp returns an approximate exponential e^x using the default precision.
func FastExp[T Float](x T) T { return FastExpPrec(x, PrecisionAuto) }

// FastExpPrec returns an approximate exponential e^x using the requested precision.
func FastExpPrec[T Float](x T, prec Precision) T {
	return iapprox.Exp(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastExp32(x float32) float32 { return FastExp[float32](x) }
func FastExp64(x float64) float64 { return FastExp[float64](x) }

// FastExpBatch32 writes exp(src[i]) to dst[i] for every element of src.
//
// This is the float32 SIMD path, not a loop over FastExp32: it runs an AVX2+FMA
// kernel where the CPU has both features and a float32-native pure-Go kernel
// otherwise, measures ~11-12x faster per element than a scalar loop, and is
// more accurate than FastExp32 (1 ulp against 38). It is therefore not
// bit-identical to FastExp32; see ACCURACY.md. See "Batch functions" in the
// package doc for the length and aliasing rules.
func FastExpBatch32(dst, src []float32) { simd.ExpFloat32(dst, src) }

// FastExpBatch64 writes exp(src[i]) to dst[i] for every element of src.
//
// There is no float64 SIMD kernel; this is a scalar loop over the same code
// FastExp64 runs, so it is bit-identical to it and the only saving is the
// amortised call frame. See "Batch functions" in the package doc for the length
// and aliasing rules.
func FastExpBatch64(dst, src []float64) { iapprox.ExpBatch64(dst, src) }

// FastTanh returns an approximate hyperbolic tangent using the default precision.
func FastTanh[T Float](x T) T { return FastTanhPrec(x, PrecisionAuto) }

// FastTanhPrec returns an approximate hyperbolic tangent using the requested precision.
//
// Odd symmetry is exact (FastTanh(-x) is the bit-for-bit negation of
// FastTanh(x)), saturation to +/-1 is exact for |x| >= 19.0625, and NaN,
// +/-Inf and +/-0 behave as in math.Tanh. At PrecisionBalanced the maximum
// absolute error over [-20, 20] is below 1e-7.
//
// FastTanh and FastLogCosh are a consistent pair: see FastLogCoshPrec.
func FastTanhPrec[T Float](x T, prec Precision) T {
	return iapprox.Tanh(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastTanh32(x float32) float32 { return FastTanh[float32](x) }
func FastTanh64(x float64) float64 { return FastTanh[float64](x) }

// FastLogCosh returns an approximate log(cosh(x)) using the default precision.
func FastLogCosh[T Float](x T) T { return FastLogCoshPrec(x, PrecisionAuto) }

// FastLogCoshPrec returns an approximate log(cosh(x)) using the requested precision.
//
// FastLogCosh is designed as a consistent pair with FastTanh rather than fitted
// independently: tanh is exactly d/dx log(cosh(x)), and the two share one
// branch point and one exp(-2|x|) evaluation so that the derivative
// relationship holds to the approximation's own accuracy. Consumers that rely
// on the identity - a discrete-gradient energy scheme, for instance, where it
// is what makes the scheme passive - can use the pair directly.
//
// Unlike math.Log(math.Cosh(x)), which overflows for |x| above ~710, this
// never forms cosh: it evaluates the asymptote |x| - ln2 + log1p(exp(-2|x|)).
//
// At PrecisionBalanced the maximum absolute error over |x| < 12 is below 1e-7,
// degrading gracefully outside that range.
func FastLogCoshPrec[T Float](x T, prec Precision) T {
	return iapprox.LogCosh(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastLogCosh32(x float32) float32 { return FastLogCosh[float32](x) }
func FastLogCosh64(x float64) float64 { return FastLogCosh[float64](x) }

// FastTanhLogCoshBatch32 writes tanh(src[i]) to dstTanh[i] and log(cosh(src[i]))
// to dstLogCosh[i] for every element of src.
//
// Both outputs come from one fused pass that shares u = exp(-2|x|) and one
// branch point, so the pair stays consistent exactly as the scalar pair does.
// This is the float32 SIMD path: an AVX2+FMA kernel where available, ~10-12x
// faster per element than a scalar loop, and not bit-identical to
// FastTanh32/FastLogCosh32 (see ACCURACY.md). See "Batch functions" in the
// package doc for the length and aliasing rules.
func FastTanhLogCoshBatch32(dstTanh, dstLogCosh, src []float32) {
	simd.TanhLogCoshFloat32(dstTanh, dstLogCosh, src)
}

// FastTanhLogCoshBatch64 writes tanh(src[i]) to dstTanh[i] and log(cosh(src[i]))
// to dstLogCosh[i] for every element of src.
//
// A scalar loop, bit-identical to FastTanh64 and FastLogCosh64 element by
// element, but a genuinely fused one: the expensive u = exp(-2|x|) is evaluated
// once per element instead of once per output, so it beats two separate scalar
// loops. See "Batch functions" in the package doc for the length and aliasing
// rules.
func FastTanhLogCoshBatch64(dstTanh, dstLogCosh, src []float64) {
	iapprox.TanhLogCoshBatch64(dstTanh, dstLogCosh, src)
}

// FastSin returns an approximate sine using the default precision.
func FastSin[T Float](x T) T { return FastSinPrec(x, PrecisionAuto) }

// FastSinPrec returns an approximate sine using the requested precision.
// Fast=3-term (~3.2 digits), Balanced=5-term (~7.3 digits), High=7-term (~12.1 digits).
func FastSinPrec[T Float](x T, prec Precision) T {
	return iapprox.Sin(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastSin32(x float32) float32 { return FastSin[float32](x) }
func FastSin64(x float64) float64 { return FastSin[float64](x) }

// FastCos returns an approximate cosine using the default precision.
func FastCos[T Float](x T) T { return FastCosPrec(x, PrecisionAuto) }

// FastCosPrec returns an approximate cosine using the requested precision.
// Fast=3-term (~3.2 digits), Balanced=5-term (~7.3 digits), High=7-term (~12.1 digits).
func FastCosPrec[T Float](x T, prec Precision) T {
	return iapprox.Cos(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastCos32(x float32) float32 { return FastCos[float32](x) }
func FastCos64(x float64) float64 { return FastCos[float64](x) }

// FastSec returns an approximate secant using the default precision.
func FastSec[T Float](x T) T { return FastSecPrec(x, PrecisionAuto) }

// FastSecPrec returns an approximate secant using the requested precision.
func FastSecPrec[T Float](x T, prec Precision) T {
	return iapprox.Sec(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastSec32(x float32) float32 { return FastSec[float32](x) }
func FastSec64(x float64) float64 { return FastSec[float64](x) }

// FastCsc returns an approximate cosecant using the default precision.
func FastCsc[T Float](x T) T { return FastCscPrec(x, PrecisionAuto) }

// FastCscPrec returns an approximate cosecant using the requested precision.
func FastCscPrec[T Float](x T, prec Precision) T {
	return iapprox.Csc(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastCsc32(x float32) float32 { return FastCsc[float32](x) }
func FastCsc64(x float64) float64 { return FastCsc[float64](x) }

// FastTan returns an approximate tangent using the default precision.
func FastTan[T Float](x T) T { return FastTanPrec(x, PrecisionAuto) }

// FastTanPrec returns an approximate tangent using the requested precision.
func FastTanPrec[T Float](x T, prec Precision) T {
	return iapprox.Tan(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastTan32(x float32) float32 { return FastTan[float32](x) }
func FastTan64(x float64) float64 { return FastTan[float64](x) }

// FastCotan returns an approximate cotangent using the default precision.
func FastCotan[T Float](x T) T { return FastCotanPrec(x, PrecisionAuto) }

// FastCotanPrec returns an approximate cotangent using the requested precision.
func FastCotanPrec[T Float](x T, prec Precision) T {
	return iapprox.Cotan(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastCotan32(x float32) float32 { return FastCotan[float32](x) }
func FastCotan64(x float64) float64 { return FastCotan[float64](x) }

// FastArctan returns an approximate arctangent using the default precision.
func FastArctan[T Float](x T) T { return FastArctanPrec(x, PrecisionAuto) }

// FastArctanPrec returns an approximate arctangent using the requested precision.
// Fast/Balanced=3-term (~6.6 digits), High=6-term (~13.7 digits).
func FastArctanPrec[T Float](x T, prec Precision) T {
	return iapprox.Arctan(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastArctan32(x float32) float32 { return FastArctan[float32](x) }
func FastArctan64(x float64) float64 { return FastArctan[float64](x) }

// FastArccotan returns an approximate arccotangent using the default precision.
func FastArccotan[T Float](x T) T { return FastArccotanPrec(x, PrecisionAuto) }

// FastArccotanPrec returns an approximate arccotangent using the requested precision.
// Fast/Balanced=3-term (~6.6 digits), High=6-term (~13.7 digits).
func FastArccotanPrec[T Float](x T, prec Precision) T {
	return iapprox.Arccotan(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastArccotan32(x float32) float32 { return FastArccotan[float32](x) }
func FastArccotan64(x float64) float64 { return FastArccotan[float64](x) }

// FastArccos returns an approximate arccosine using the default precision.
func FastArccos[T Float](x T) T { return FastArccosPrec(x, PrecisionAuto) }

// FastArccosPrec returns an approximate arccosine using the requested precision.
// Fast/Balanced=3-term (~6.6 digits), High=6-term (~13.7 digits).
func FastArccosPrec[T Float](x T, prec Precision) T {
	return iapprox.Arccos(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastArccos32(x float32) float32 { return FastArccos[float32](x) }
func FastArccos64(x float64) float64 { return FastArccos[float64](x) }

// FastPower returns an approximate power base^exponent.
// Uses exp/log composition: base^exponent = exp(exponent * ln(base)).
func FastPower[T Float](base, exponent T) T {
	return iapprox.Power(base, exponent)
}

func FastPower32(base, exponent float32) float32 { return FastPower[float32](base, exponent) }
func FastPower64(base, exponent float64) float64 { return FastPower[float64](base, exponent) }

// FastRoot returns an approximate nth root of value.
// Uses the identity: root(value, n) = value^(1/n).
func FastRoot[T Float](value T, n int) T {
	return iapprox.Root(value, n)
}

func FastRoot32(value float32, n int) float32 { return FastRoot[float32](value, n) }
func FastRoot64(value float64, n int) float64 { return FastRoot[float64](value, n) }

// FastIntPower returns an approximate integer power base^exponent.
// Uses efficient binary exponentiation for integer exponents.
func FastIntPower[T Float](base T, exponent int) T {
	return iapprox.IntPower(base, exponent)
}

func FastIntPower32(base float32, exponent int) float32 {
	return FastIntPower[float32](base, exponent)
}

func FastIntPower64(base float64, exponent int) float64 {
	return FastIntPower[float64](base, exponent)
}
