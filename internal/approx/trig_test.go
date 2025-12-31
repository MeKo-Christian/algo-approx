package approx

import (
	"math"
	"testing"
)

// TestSin3Term tests the 3-term Taylor series approximation for sine.
func TestSin3Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64 // acceptable error
	}{
		{"zero", 0.0, 0.0, 1e-10},
		{"π/6", math.Pi / 6, 0.5, 0.001}, // ~3 decimal digits accuracy
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.001},
		{"π/3", math.Pi / 3, math.Sqrt(3) / 2, 0.001},
		{"π/2", math.Pi / 2, 1.0, 0.005}, // Less accurate at edge of range
		{"-π/6", -math.Pi / 6, -0.5, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sin3Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sin3Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSin3TermFloat32 tests the 3-term sine approximation with float32.
func TestSin3TermFloat32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float32
		want  float32
		delta float32
	}{
		{"zero", 0.0, 0.0, 1e-6},
		{"π/6", float32(math.Pi / 6), 0.5, 0.001},
		{"π/4", float32(math.Pi / 4), float32(math.Sqrt2 / 2), 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sin3Term(tt.input)
			if float32(math.Abs(float64(got-tt.want))) > tt.delta {
				t.Errorf("sin3Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCos3Term tests the 3-term Taylor series approximation for cosine.
func TestCos3Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, math.Sqrt(3) / 2, 0.001},
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.001},
		{"π/3", math.Pi / 3, 0.5, 0.002}, // 3-term less accurate here
		{"π/2", math.Pi / 2, 0.0, 0.020}, // 3-term less accurate at edge
		{"-π/6", -math.Pi / 6, math.Sqrt(3) / 2, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := cos3Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("cos3Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCos3TermFloat32 tests the 3-term cosine approximation with float32.
func TestCos3TermFloat32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float32
		want  float32
		delta float32
	}{
		{"zero", 0.0, 1.0, 1e-6},
		{"π/6", float32(math.Pi / 6), float32(math.Sqrt(3) / 2), 0.001},
		{"π/4", float32(math.Pi / 4), float32(math.Sqrt2 / 2), 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := cos3Term(tt.input)
			if float32(math.Abs(float64(got-tt.want))) > tt.delta {
				t.Errorf("cos3Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSec3Term tests the 3-term secant approximation (sec = 1/cos).
func TestSec3Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, 2.0 / math.Sqrt(3), 0.002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.002},
		{"π/3", math.Pi / 3, 2.0, 0.008}, // 3-term propagates cosine error
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sec3Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sec3Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCsc3Term tests the 3-term cosecant approximation (csc = 1/sin).
func TestCsc3Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"π/6", math.Pi / 6, 2.0, 0.002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.002},
		{"π/3", math.Pi / 3, 2.0 / math.Sqrt(3), 0.002},
		{"π/2", math.Pi / 2, 1.0, 0.010},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := csc3Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("csc3Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSin4Term tests the 4-term Taylor series approximation for sine.
//
//nolint:dupl
func TestSin4Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 0.0, 1e-10},
		{"π/6", math.Pi / 6, 0.5, 0.0001}, // Better accuracy with 4 terms
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.0001},
		{"π/3", math.Pi / 3, math.Sqrt(3) / 2, 0.0001},
		{"π/2", math.Pi / 2, 1.0, 0.0005},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sin4Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sin4Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCos4Term tests the 4-term Taylor series approximation for cosine.
func TestCos4Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, math.Sqrt(3) / 2, 0.0001},
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.0001},
		{"π/3", math.Pi / 3, 0.5, 0.0001},
		{"π/2", math.Pi / 2, 0.0, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := cos4Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("cos4Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSec4Term tests the 4-term secant approximation.
func TestSec4Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, 2.0 / math.Sqrt(3), 0.0002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.0002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sec4Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sec4Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCsc4Term tests the 4-term cosecant approximation.
func TestCsc4Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"π/6", math.Pi / 6, 2.0, 0.0002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.0002},
		{"π/3", math.Pi / 3, 2.0 / math.Sqrt(3), 0.0002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := csc4Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("csc4Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSin5Term tests the 5-term Taylor series approximation for sine.
//
//nolint:dupl
func TestSin5Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 0.0, 1e-10},
		{"π/6", math.Pi / 6, 0.5, 0.00001}, // Even better accuracy
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.00001},
		{"π/3", math.Pi / 3, math.Sqrt(3) / 2, 0.00001},
		{"π/2", math.Pi / 2, 1.0, 0.0001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sin5Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sin5Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCos5Term tests the 5-term Taylor series approximation for cosine.
func TestCos5Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, math.Sqrt(3) / 2, 0.00001},
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.00001},
		{"π/3", math.Pi / 3, 0.5, 0.00001},
		{"π/2", math.Pi / 2, 0.0, 0.0001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := cos5Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("cos5Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSec5Term tests the 5-term secant approximation.
func TestSec5Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, 2.0 / math.Sqrt(3), 0.00002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.00002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sec5Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sec5Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCsc5Term tests the 5-term cosecant approximation.
func TestCsc5Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"π/6", math.Pi / 6, 2.0, 0.00002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.00002},
		{"π/3", math.Pi / 3, 2.0 / math.Sqrt(3), 0.00002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := csc5Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("csc5Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSin6Term tests the 6-term Taylor series approximation for sine.
//
//nolint:dupl
func TestSin6Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 0.0, 1e-10},
		{"π/6", math.Pi / 6, 0.5, 0.000001},
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.000001},
		{"π/3", math.Pi / 3, math.Sqrt(3) / 2, 0.000001},
		{"π/2", math.Pi / 2, 1.0, 0.00001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sin6Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sin6Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCos6Term tests the 6-term Taylor series approximation for cosine.
func TestCos6Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, math.Sqrt(3) / 2, 0.000001},
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.000001},
		{"π/3", math.Pi / 3, 0.5, 0.000001},
		{"π/2", math.Pi / 2, 0.0, 0.00001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := cos6Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("cos6Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSec6Term tests the 6-term secant approximation.
func TestSec6Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, 2.0 / math.Sqrt(3), 0.000002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.000002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sec6Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sec6Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCsc6Term tests the 6-term cosecant approximation.
func TestCsc6Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"π/6", math.Pi / 6, 2.0, 0.000002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.000002},
		{"π/3", math.Pi / 3, 2.0 / math.Sqrt(3), 0.000002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := csc6Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("csc6Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSin7Term tests the 7-term Taylor series approximation for sine (~12.1 digits).
//
//nolint:dupl
func TestSin7Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 0.0, 1e-10},
		{"π/6", math.Pi / 6, 0.5, 0.0000001},
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.0000001},
		{"π/3", math.Pi / 3, math.Sqrt(3) / 2, 0.0000001},
		{"π/2", math.Pi / 2, 1.0, 0.000001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sin7Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sin7Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCos7Term tests the 7-term Taylor series approximation for cosine.
func TestCos7Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, math.Sqrt(3) / 2, 0.0000001},
		{"π/4", math.Pi / 4, math.Sqrt2 / 2, 0.0000001},
		{"π/3", math.Pi / 3, 0.5, 0.0000001},
		{"π/2", math.Pi / 2, 0.0, 0.000001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := cos7Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("cos7Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestSec7Term tests the 7-term secant approximation.
func TestSec7Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"zero", 0.0, 1.0, 1e-10},
		{"π/6", math.Pi / 6, 2.0 / math.Sqrt(3), 0.0000002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.0000002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := sec7Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("sec7Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}

// TestCsc7Term tests the 7-term cosecant approximation.
func TestCsc7Term(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input float64
		want  float64
		delta float64
	}{
		{"π/6", math.Pi / 6, 2.0, 0.0000002},
		{"π/4", math.Pi / 4, math.Sqrt2, 0.0000002},
		{"π/3", math.Pi / 3, 2.0 / math.Sqrt(3), 0.0000002},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := csc7Term(tt.input)
			if math.Abs(got-tt.want) > tt.delta {
				t.Errorf("csc7Term(%v) = %v, want %v (±%v)", tt.input, got, tt.want, tt.delta)
			}
		})
	}
}
