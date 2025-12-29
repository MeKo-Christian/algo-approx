package approx

// Float is the type constraint for internal scalar approximation routines.
//
// This is intentionally duplicated from the public package to avoid an import
// cycle (the public API depends on internal implementations).
type Float interface {
	~float32 | ~float64
}

// Precision controls the speed/accuracy tradeoff for internal implementations.
//
// Values are aligned with the public approx.Precision constants.
type Precision int

const (
	PrecisionAuto Precision = iota
	PrecisionFast
	PrecisionBalanced
	PrecisionHigh
)

func (p Precision) IsValid() bool {
	switch p {
	case PrecisionAuto, PrecisionFast, PrecisionBalanced, PrecisionHigh:
		return true
	default:
		return false
	}
}

func normalizePrecision(p Precision) Precision {
	if p == PrecisionAuto {
		return PrecisionBalanced
	}
	if !p.IsValid() {
		return PrecisionBalanced
	}
	return p
}
