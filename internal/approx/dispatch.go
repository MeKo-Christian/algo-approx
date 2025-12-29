package approx

func selectImpl[T Float](fast, balanced, high func(T) T, prec Precision) func(T) T {
	switch normalizePrecision(prec) {
	case PrecisionFast:
		return fast
	case PrecisionHigh:
		return high
	default:
		return balanced
	}
}
