package approx

import iapprox "github.com/meko-christian/algo-approx/internal/approx"

// FastSqrt returns an approximate square root using the default precision.
func FastSqrt[T Float](x T) T { return FastSqrtPrec(x, PrecisionAuto) }

// FastSqrtPrec returns an approximate square root using the requested precision.
func FastSqrtPrec[T Float](x T, prec Precision) T {
	return iapprox.Sqrt(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastSqrt32(x float32) float32 { return FastSqrt[float32](x) }
func FastSqrt64(x float64) float64 { return FastSqrt[float64](x) }

// FastInvSqrt returns an approximate inverse square root using the default precision.
func FastInvSqrt[T Float](x T) T { return FastInvSqrtPrec(x, PrecisionAuto) }

// FastInvSqrtPrec returns an approximate inverse square root using the requested precision.
func FastInvSqrtPrec[T Float](x T, prec Precision) T {
	return iapprox.InvSqrt(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastInvSqrt32(x float32) float32 { return FastInvSqrt[float32](x) }
func FastInvSqrt64(x float64) float64 { return FastInvSqrt[float64](x) }

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

// FastSin returns an approximate sine using the default precision.
func FastSin[T Float](x T) T { return FastSinPrec(x, PrecisionAuto) }

// FastSinPrec returns an approximate sine using the requested precision.
// Fast=3-term (~3.2 digits), Balanced=5-term (~7.3 digits), High=7-term (~12.1 digits)
func FastSinPrec[T Float](x T, prec Precision) T {
	return iapprox.Sin(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastSin32(x float32) float32 { return FastSin[float32](x) }
func FastSin64(x float64) float64 { return FastSin[float64](x) }

// FastCos returns an approximate cosine using the default precision.
func FastCos[T Float](x T) T { return FastCosPrec(x, PrecisionAuto) }

// FastCosPrec returns an approximate cosine using the requested precision.
// Fast=3-term (~3.2 digits), Balanced=5-term (~7.3 digits), High=7-term (~12.1 digits)
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
// Fast/Balanced=3-term (~6.6 digits), High=6-term (~13.7 digits)
func FastArctanPrec[T Float](x T, prec Precision) T {
	return iapprox.Arctan(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastArctan32(x float32) float32 { return FastArctan[float32](x) }
func FastArctan64(x float64) float64 { return FastArctan[float64](x) }

// FastArccotan returns an approximate arccotangent using the default precision.
func FastArccotan[T Float](x T) T { return FastArccotanPrec(x, PrecisionAuto) }

// FastArccotanPrec returns an approximate arccotangent using the requested precision.
// Fast/Balanced=3-term (~6.6 digits), High=6-term (~13.7 digits)
func FastArccotanPrec[T Float](x T, prec Precision) T {
	return iapprox.Arccotan(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastArccotan32(x float32) float32 { return FastArccotan[float32](x) }
func FastArccotan64(x float64) float64 { return FastArccotan[float64](x) }

// FastArccos returns an approximate arccosine using the default precision.
func FastArccos[T Float](x T) T { return FastArccosPrec(x, PrecisionAuto) }

// FastArccosPrec returns an approximate arccosine using the requested precision.
// Fast/Balanced=3-term (~6.6 digits), High=6-term (~13.7 digits)
func FastArccosPrec[T Float](x T, prec Precision) T {
	return iapprox.Arccos(x, iapprox.Precision(normalizePrecision(prec)))
}

func FastArccos32(x float32) float32 { return FastArccos[float32](x) }
func FastArccos64(x float64) float64 { return FastArccos[float64](x) }
