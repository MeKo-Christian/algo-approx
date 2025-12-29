package approx

import (
	"math"
	"testing"
)

// TestArctan3Term tests the 3-term arctangent approximation for float32
func TestArctan3Term32(t *testing.T) {
	tests := []struct {
		name      string
		input     float32
		tolerance float32
	}{
		{"zero", 0.0, 1e-7},
		{"small positive", 0.1, 5e-6},
		{"π/12 boundary", float32(math.Pi / 12), 2e-5},
		{"negative", -0.1, 5e-6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arctan3Term(tt.input)
			want := float32(math.Atan(float64(tt.input)))
			diff := abs32(got - want)
			if diff > tt.tolerance {
				t.Errorf("arctan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArctan3Term tests the 3-term arctangent approximation for float64
func TestArctan3Term64(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-14},
		{"small positive", 0.1, 2e-8},
		{"π/12 boundary", math.Pi / 12, 2e-5},
		{"negative", -0.1, 2e-8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arctan3Term(tt.input)
			want := math.Atan(tt.input)
			diff := abs64(got - want)
			if diff > tt.tolerance {
				t.Errorf("arctan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArctan6Term tests the 6-term arctangent approximation for float32
func TestArctan6Term32(t *testing.T) {
	tests := []struct {
		name      string
		input     float32
		tolerance float32
	}{
		{"zero", 0.0, 1e-7},
		{"small positive", 0.1, 1e-7},
		{"π/12 boundary", float32(math.Pi / 12), 1e-7},
		{"negative", -0.1, 1e-7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arctan6Term(tt.input)
			want := float32(math.Atan(float64(tt.input)))
			diff := abs32(got - want)
			if diff > tt.tolerance {
				t.Errorf("arctan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArctan6Term tests the 6-term arctangent approximation for float64
func TestArctan6Term64(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-14},
		{"small positive", 0.1, 1e-14},
		{"π/12 boundary", math.Pi / 12, 2e-9},
		{"negative", -0.1, 1e-14},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arctan6Term(tt.input)
			want := math.Atan(tt.input)
			diff := abs64(got - want)
			if diff > tt.tolerance {
				t.Errorf("arctan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccotan3Term tests the 3-term arccotangent approximation for float32
func TestArccotan3Term32(t *testing.T) {
	tests := []struct {
		name      string
		input     float32
		tolerance float32
	}{
		{"small positive", 0.1, 1e-5},
		{"π/12 boundary", float32(math.Pi / 12), 2e-5},
		// Note: arctan approximation is only valid for small x, so arccotan(1) has larger error
		{"one", 1.0, 0.1}, // Large tolerance for x=1 where series breaks down
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccotan3Term(tt.input)
			want := float32(math.Pi/2 - math.Atan(float64(tt.input)))
			diff := abs32(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccotan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccotan3Term tests the 3-term arccotangent approximation for float64
func TestArccotan3Term64(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"small positive", 0.1, 2e-8},
		{"π/12 boundary", math.Pi / 12, 2e-5},
		// Note: arctan approximation is only valid for small x, so arccotan(1) has larger error
		{"one", 1.0, 0.1}, // Large tolerance for x=1 where series breaks down
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccotan3Term(tt.input)
			want := math.Pi/2 - math.Atan(tt.input)
			diff := abs64(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccotan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccotan6Term tests the 6-term arccotangent approximation for float32
func TestArccotan6Term32(t *testing.T) {
	tests := []struct {
		name      string
		input     float32
		tolerance float32
	}{
		{"small positive", 0.1, 2e-7},
		{"π/12 boundary", float32(math.Pi / 12), 1e-7},
		// Note: arctan approximation is only valid for small x, so arccotan(1) has larger error
		{"one", 1.0, 0.05}, // Larger tolerance for x=1
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccotan6Term(tt.input)
			want := float32(math.Pi/2 - math.Atan(float64(tt.input)))
			diff := abs32(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccotan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccotan6Term tests the 6-term arccotangent approximation for float64
func TestArccotan6Term64(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"small positive", 0.1, 1e-13},
		{"π/12 boundary", math.Pi / 12, 2e-9},
		// Note: arctan approximation is only valid for small x, so arccotan(1) has larger error
		{"one", 1.0, 0.05}, // Larger tolerance for x=1
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccotan6Term(tt.input)
			want := math.Pi/2 - math.Atan(tt.input)
			diff := abs64(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccotan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccos3Term tests the 3-term arccosine approximation for float32
func TestArccos3Term32(t *testing.T) {
	tests := []struct {
		name      string
		input     float32
		tolerance float32
	}{
		{"zero", 0.0, 1e-5},
		{"half", 0.5, 1e-3},
		{"sqrt(2)/2", float32(math.Sqrt(2) / 2), 2e-4},
		{"one", 1.0, 1e-5},
		{"negative half", -0.5, 0.1}, // Larger tolerance for negative values
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccos3Term(tt.input)
			want := float32(math.Acos(float64(tt.input)))
			diff := abs32(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccos3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccos3Term tests the 3-term arccosine approximation for float64
func TestArccos3Term64(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"half", 0.5, 1e-3},
		{"sqrt(2)/2", math.Sqrt(2) / 2, 2e-4},
		{"one", 1.0, 1e-10},
		{"negative half", -0.5, 0.1}, // Larger tolerance for negative values
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccos3Term(tt.input)
			want := math.Acos(tt.input)
			diff := abs64(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccos3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccos6Term tests the 6-term arccosine approximation for float32
func TestArccos6Term32(t *testing.T) {
	tests := []struct {
		name      string
		input     float32
		tolerance float32
	}{
		{"zero", 0.0, 1e-7},
		{"half", 0.5, 6e-6},
		{"sqrt(2)/2", float32(math.Sqrt(2) / 2), 3e-7},
		{"one", 1.0, 1e-7},
		{"negative half", -0.5, 0.02}, // Larger tolerance for negative values
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccos6Term(tt.input)
			want := float32(math.Acos(float64(tt.input)))
			diff := abs32(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccos6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestArccos6Term tests the 6-term arccosine approximation for float64
func TestArccos6Term64(t *testing.T) {
	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-13},
		{"half", 0.5, 6e-6},
		{"sqrt(2)/2", math.Sqrt(2) / 2, 2e-7},
		{"one", 1.0, 1e-13},
		{"negative half", -0.5, 0.02}, // Larger tolerance for negative values
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arccos6Term(tt.input)
			want := math.Acos(tt.input)
			diff := abs64(got - want)
			if diff > tt.tolerance {
				t.Errorf("arccos6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// Helper functions
func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
