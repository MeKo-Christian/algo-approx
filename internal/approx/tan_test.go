package approx

import (
	"math"
	"testing"
)

// TestTan2Term tests the 2-term Taylor series tangent approximation.
// According to PLAN.md: ~3.2 digits accuracy, range [0, π/4].
//
//nolint:dupl
func TestTan2Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"π/12", math.Pi / 12, 0.01}, // ~15 degrees
		{"π/8", math.Pi / 8, 0.01},   // ~22.5 degrees
		{"π/6", math.Pi / 6, 0.01},   // ~30 degrees
		{"π/4", math.Pi / 4, 0.06},   // ~45 degrees (upper bound, relaxed tolerance)
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := tan2Term(tt.input)
			want := math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("tan2Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := tan2Term(float32(tt.input))
			want := float32(math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("tan2Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestCotan2Term tests the 2-term Taylor series cotangent approximation.
func TestCotan2Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"π/12", math.Pi / 12, 0.01},
		{"π/8", math.Pi / 8, 0.01},
		{"π/6", math.Pi / 6, 0.02}, // Relaxed tolerance for cotangent
		{"π/4", math.Pi / 4, 0.06}, // Upper bound, more relaxed
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := cotan2Term(tt.input)
			want := 1.0 / math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("cotan2Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := cotan2Term(float32(tt.input))
			want := float32(1.0 / math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("cotan2Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestTan3Term tests the 3-term Taylor series tangent approximation.
// According to PLAN.md: ~5.6 digits accuracy.
//
//nolint:dupl
func TestTan3Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"π/12", math.Pi / 12, 0.001},
		{"π/8", math.Pi / 8, 0.001},
		{"π/6", math.Pi / 6, 0.001},
		{"π/4", math.Pi / 4, 0.015},
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := tan3Term(tt.input)
			want := math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("tan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := tan3Term(float32(tt.input))
			want := float32(math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("tan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestCotan3Term tests the 3-term Taylor series cotangent approximation.
func TestCotan3Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"π/12", math.Pi / 12, 0.001},
		{"π/8", math.Pi / 8, 0.001},
		{"π/6", math.Pi / 6, 0.002},
		{"π/4", math.Pi / 4, 0.015},
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := cotan3Term(tt.input)
			want := 1.0 / math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("cotan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := cotan3Term(float32(tt.input))
			want := float32(1.0 / math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("cotan3Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestTan4Term tests the 4-term Taylor series tangent approximation.
// According to PLAN.md: ~8.2 digits accuracy.
//
//nolint:dupl
func TestTan4Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"π/12", math.Pi / 12, 0.0001},
		{"π/8", math.Pi / 8, 0.0001},
		{"π/6", math.Pi / 6, 0.0001},
		{"π/4", math.Pi / 4, 0.005},
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := tan4Term(tt.input)
			want := math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("tan4Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := tan4Term(float32(tt.input))
			want := float32(math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("tan4Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestCotan4Term tests the 4-term Taylor series cotangent approximation.
func TestCotan4Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"π/12", math.Pi / 12, 0.0001},
		{"π/8", math.Pi / 8, 0.0001},
		{"π/6", math.Pi / 6, 0.0003},
		{"π/4", math.Pi / 4, 0.004},
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := cotan4Term(tt.input)
			want := 1.0 / math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("cotan4Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := cotan4Term(float32(tt.input))
			want := float32(1.0 / math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("cotan4Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestTan6Term tests the 6-term Taylor series tangent approximation.
// According to PLAN.md: ~14 digits accuracy.
//
//nolint:dupl
func TestTan6Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-15},
		{"π/12", math.Pi / 12, 3e-8},
		{"π/8", math.Pi / 8, 3e-8},
		{"π/6", math.Pi / 6, 1e-6},
		{"π/4", math.Pi / 4, 3e-4},
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := tan6Term(tt.input)
			want := math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("tan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := tan6Term(float32(tt.input))
			want := float32(math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("tan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestCotan6Term tests the 6-term Taylor series cotangent approximation.
func TestCotan6Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"π/12", math.Pi / 12, 3e-8},
		{"π/8", math.Pi / 8, 3e-7},
		{"π/6", math.Pi / 6, 3e-6},
		{"π/4", math.Pi / 4, 3e-4},
	}

	for _, tt := range tests {
		t.Run(tt.name+"_float64", func(t *testing.T) {
			t.Parallel()

			got := cotan6Term(tt.input)
			want := 1.0 / math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("cotan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})

		t.Run(tt.name+"_float32", func(t *testing.T) {
			t.Parallel()

			got := cotan6Term(float32(tt.input))
			want := float32(1.0 / math.Tan(tt.input))
			diff := float32(math.Abs(float64(got - want)))

			if diff > float32(tt.tolerance) {
				t.Errorf("cotan6Term(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}
