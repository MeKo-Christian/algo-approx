package approx

// Precision controls the accuracy/speed tradeoff of approximation routines.
//
// PrecisionBalanced is the recommended default.
type Precision int

const (
	// PrecisionAuto uses the library default for the operation.
	PrecisionAuto Precision = iota

	// PrecisionFast prioritizes speed over accuracy.
	PrecisionFast

	// PrecisionBalanced balances speed and accuracy (default).
	PrecisionBalanced

	// PrecisionHigh prioritizes accuracy over speed.
	PrecisionHigh
)

func (p Precision) String() string {
	switch p {
	case PrecisionAuto:
		return "auto"
	case PrecisionFast:
		return "fast"
	case PrecisionBalanced:
		return "balanced"
	case PrecisionHigh:
		return "high"
	default:
		return "unknown"
	}
}

// IsValid reports whether p is a recognized precision value.
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
