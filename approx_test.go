package approx

import (
	"math"
	"testing"
)

func TestPublicAPI_Sqrt(t *testing.T) {
	got := FastSqrt(16.0)
	if math.Abs(got-4.0) > 1e-2 {
		t.Fatalf("FastSqrt(16) got %g", got)
	}
}

func TestPublicAPI_InvSqrt(t *testing.T) {
	got := FastInvSqrt(4.0)
	if math.Abs(got-0.5) > 1e-2 {
		t.Fatalf("FastInvSqrt(4) got %g", got)
	}
}

func TestPublicAPI_LogExp(t *testing.T) {
	x := 3.0
	if math.Abs(FastExp(FastLog(x))-x) > 5e-2 {
		t.Fatalf("exp(log(x)) composition too far")
	}
}

// TestFastSin tests the public FastSin API
func TestFastSin(t *testing.T) {
	x := math.Pi / 6.0
	got := FastSin(x)
	want := 0.5
	if math.Abs(got-want) > 0.01 { // Balanced precision
		t.Errorf("FastSin(%v) = %v, want ~%v", x, got, want)
	}
}

// TestFastSinPrec tests FastSin with explicit precision
func TestFastSinPrec(t *testing.T) {
	x := math.Pi / 6.0

	// Test each precision level
	precisions := []Precision{PrecisionFast, PrecisionBalanced, PrecisionHigh}
	for _, prec := range precisions {
		got := FastSinPrec(x, prec)
		want := 0.5
		// Higher precision should have smaller error
		maxError := 0.1 // Conservative for all precisions
		if math.Abs(got-want) > maxError {
			t.Errorf("FastSinPrec(%v, %v) = %v, want ~%v", x, prec, got, want)
		}
	}
}

// TestFastCos tests the public FastCos API
func TestFastCos(t *testing.T) {
	x := math.Pi / 3.0
	got := FastCos(x)
	want := 0.5
	if math.Abs(got-want) > 0.01 {
		t.Errorf("FastCos(%v) = %v, want ~%v", x, got, want)
	}
}

// TestFastCosPrec tests FastCos with explicit precision
func TestFastCosPrec(t *testing.T) {
	x := math.Pi / 3.0

	precisions := []Precision{PrecisionFast, PrecisionBalanced, PrecisionHigh}
	for _, prec := range precisions {
		got := FastCosPrec(x, prec)
		want := 0.5
		maxError := 0.1
		if math.Abs(got-want) > maxError {
			t.Errorf("FastCosPrec(%v, %v) = %v, want ~%v", x, prec, got, want)
		}
	}
}

// TestFastTan tests the public FastTan API
func TestFastTan(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"π/6", math.Pi / 6, 0.01},
		{"π/4", math.Pi / 4, 0.02},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FastTan(tt.input)
			want := math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastTan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastTanPrec tests the public FastTanPrec API with different precision levels
func TestFastTanPrec(t *testing.T) {
	x := math.Pi / 6

	t.Run("PrecisionFast", func(t *testing.T) {
		got := FastTanPrec(x, PrecisionFast)
		want := math.Tan(x)
		diff := math.Abs(got - want)
		if diff > 0.01 {
			t.Errorf("FastTanPrec(%v, PrecisionFast) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionBalanced", func(t *testing.T) {
		got := FastTanPrec(x, PrecisionBalanced)
		want := math.Tan(x)
		diff := math.Abs(got - want)
		if diff > 0.001 {
			t.Errorf("FastTanPrec(%v, PrecisionBalanced) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionHigh", func(t *testing.T) {
		got := FastTanPrec(x, PrecisionHigh)
		want := math.Tan(x)
		diff := math.Abs(got - want)
		if diff > 0.000001 {
			t.Errorf("FastTanPrec(%v, PrecisionHigh) diff too large: %v", x, diff)
		}
	})
}

// TestFastCotan tests the public FastCotan API
func TestFastCotan(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"π/12", math.Pi / 12, 0.01},
		{"π/6", math.Pi / 6, 0.01},
		{"π/4", math.Pi / 4, 0.02},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FastCotan(tt.input)
			want := 1.0 / math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastCotan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArctan tests the public FastArctan API
func TestFastArctan(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"small positive", 0.1, 1e-5},
		{"π/12", math.Pi / 12, 2e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FastArctan(tt.input)
			want := math.Atan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastArctan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArctanPrec tests the public FastArctanPrec API with different precision levels
func TestFastArctanPrec(t *testing.T) {
	x := 0.1

	t.Run("PrecisionFast", func(t *testing.T) {
		got := FastArctanPrec(x, PrecisionFast)
		want := math.Atan(x)
		diff := math.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("FastArctanPrec(%v, PrecisionFast) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionBalanced", func(t *testing.T) {
		got := FastArctanPrec(x, PrecisionBalanced)
		want := math.Atan(x)
		diff := math.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("FastArctanPrec(%v, PrecisionBalanced) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionHigh", func(t *testing.T) {
		got := FastArctanPrec(x, PrecisionHigh)
		want := math.Atan(x)
		diff := math.Abs(got - want)
		if diff > 1e-10 {
			t.Errorf("FastArctanPrec(%v, PrecisionHigh) diff too large: %v", x, diff)
		}
	})
}

// TestFastArccotan tests the public FastArccotan API
func TestFastArccotan(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"small positive", 0.1, 1e-5},
		{"π/12", math.Pi / 12, 2e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FastArccotan(tt.input)
			want := math.Pi/2 - math.Atan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastArccotan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArccos tests the public FastArccos API
func TestFastArccos(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-5},
		{"half", 0.5, 1e-3},
		{"sqrt(2)/2", math.Sqrt(2) / 2, 2e-4},
		{"one", 1.0, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FastArccos(tt.input)
			want := math.Acos(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastArccos(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArccosPrec tests the public FastArccosPrec API with different precision levels
func TestFastArccosPrec(t *testing.T) {
	x := 0.5

	t.Run("PrecisionFast", func(t *testing.T) {
		got := FastArccosPrec(x, PrecisionFast)
		want := math.Acos(x)
		diff := math.Abs(got - want)
		if diff > 1e-3 {
			t.Errorf("FastArccosPrec(%v, PrecisionFast) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionBalanced", func(t *testing.T) {
		got := FastArccosPrec(x, PrecisionBalanced)
		want := math.Acos(x)
		diff := math.Abs(got - want)
		if diff > 1e-3 {
			t.Errorf("FastArccosPrec(%v, PrecisionBalanced) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionHigh", func(t *testing.T) {
		got := FastArccosPrec(x, PrecisionHigh)
		want := math.Acos(x)
		diff := math.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("FastArccosPrec(%v, PrecisionHigh) diff too large: %v", x, diff)
		}
	})
}
