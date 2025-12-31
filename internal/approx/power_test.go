package approx

import (
	"math"
	"testing"
)

func TestPower32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		base      float32
		exponent  float32
		want      float32
		tolerance float32
	}{
		{"2^3", 2.0, 3.0, 8.0, 1e-3},
		{"3^2", 3.0, 2.0, 9.0, 1e-4},
		{"10^0.5", 10.0, 0.5, 3.162277, 1e-4},
		{"e^1", math.E, 1.0, math.E, 1e-4},
		{"2^-2", 2.0, -2.0, 0.25, 1e-4},
		{"0.5^2", 0.5, 2.0, 0.25, 1e-4},
		{"4^0.5", 4.0, 0.5, 2.0, 5e-5},
		{"27^(1/3)", 27.0, 1.0 / 3.0, 3.0, 1e-4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := Power(tt.base, tt.exponent)
			if math.IsNaN(float64(got)) {
				t.Errorf("Power(%v, %v) = NaN", tt.base, tt.exponent)
				return
			}

			if diff := math.Abs(float64(got - tt.want)); diff > float64(tt.tolerance) {
				t.Errorf("Power(%v, %v) = %v, want %v (diff = %v, tolerance = %v)",
					tt.base, tt.exponent, got, tt.want, diff, tt.tolerance)
			}
		})
	}
}

func TestPower64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		base      float64
		exponent  float64
		want      float64
		tolerance float64
	}{
		{"2^3", 2.0, 3.0, 8.0, 1e-3},
		{"3^2", 3.0, 2.0, 9.0, 1e-6},
		{"10^0.5", 10.0, 0.5, 3.16227766016838, 1e-6},
		{"e^1", math.E, 1.0, math.E, 1e-8},
		{"2^-2", 2.0, -2.0, 0.25, 1e-5},
		{"0.5^2", 0.5, 2.0, 0.25, 1e-5},
		{"4^0.5", 4.0, 0.5, 2.0, 2e-5},
		{"27^(1/3)", 27.0, 1.0 / 3.0, 3.0, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := Power(tt.base, tt.exponent)
			if math.IsNaN(got) {
				t.Errorf("Power(%v, %v) = NaN", tt.base, tt.exponent)
				return
			}

			if diff := math.Abs(got - tt.want); diff > tt.tolerance {
				t.Errorf("Power(%v, %v) = %v, want %v (diff = %v, tolerance = %v)",
					tt.base, tt.exponent, got, tt.want, diff, tt.tolerance)
			}
		})
	}
}

func TestRoot32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		value     float32
		n         int
		want      float32
		tolerance float32
	}{
		{"sqrt(4)", 4.0, 2, 2.0, 1e-5},
		{"cbrt(8)", 8.0, 3, 2.0, 1e-4},
		{"cbrt(27)", 27.0, 3, 3.0, 1e-4},
		{"4th root(16)", 16.0, 4, 2.0, 1e-4},
		{"5th root(32)", 32.0, 5, 2.0, 1e-4},
		{"sqrt(2)", 2.0, 2, 1.4142135, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := Root(tt.value, tt.n)
			if math.IsNaN(float64(got)) {
				t.Errorf("Root(%v, %v) = NaN", tt.value, tt.n)
				return
			}

			if diff := math.Abs(float64(got - tt.want)); diff > float64(tt.tolerance) {
				t.Errorf("Root(%v, %v) = %v, want %v (diff = %v, tolerance = %v)",
					tt.value, tt.n, got, tt.want, diff, tt.tolerance)
			}
		})
	}
}

func TestRoot64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		value     float64
		n         int
		want      float64
		tolerance float64
	}{
		{"sqrt(4)", 4.0, 2, 2.0, 1e-10},
		{"cbrt(8)", 8.0, 3, 2.0, 1e-5},
		{"cbrt(27)", 27.0, 3, 3.0, 1e-5},
		{"4th root(16)", 16.0, 4, 2.0, 1e-5},
		{"5th root(32)", 32.0, 5, 2.0, 1e-5},
		{"sqrt(2)", 2.0, 2, 1.4142135623730951, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := Root(tt.value, tt.n)
			if math.IsNaN(got) {
				t.Errorf("Root(%v, %v) = NaN", tt.value, tt.n)
				return
			}

			if diff := math.Abs(got - tt.want); diff > tt.tolerance {
				t.Errorf("Root(%v, %v) = %v, want %v (diff = %v, tolerance = %v)",
					tt.value, tt.n, got, tt.want, diff, tt.tolerance)
			}
		})
	}
}

func TestIntPower32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		base      float32
		exponent  int
		want      float32
		tolerance float32
	}{
		{"2^0", 2.0, 0, 1.0, 1e-7},
		{"2^1", 2.0, 1, 2.0, 1e-7},
		{"2^2", 2.0, 2, 4.0, 1e-7},
		{"2^3", 2.0, 3, 8.0, 1e-7},
		{"2^10", 2.0, 10, 1024.0, 1e-4},
		{"3^5", 3.0, 5, 243.0, 1e-4},
		{"10^3", 10.0, 3, 1000.0, 1e-3},
		{"2^-1", 2.0, -1, 0.5, 1e-7},
		{"2^-2", 2.0, -2, 0.25, 1e-7},
		{"10^-2", 10.0, -2, 0.01, 1e-7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := IntPower(tt.base, tt.exponent)
			if math.IsNaN(float64(got)) {
				t.Errorf("IntPower(%v, %v) = NaN", tt.base, tt.exponent)
				return
			}

			if diff := math.Abs(float64(got - tt.want)); diff > float64(tt.tolerance) {
				t.Errorf("IntPower(%v, %v) = %v, want %v (diff = %v, tolerance = %v)",
					tt.base, tt.exponent, got, tt.want, diff, tt.tolerance)
			}
		})
	}
}

func TestIntPower64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		base      float64
		exponent  int
		want      float64
		tolerance float64
	}{
		{"2^0", 2.0, 0, 1.0, 1e-15},
		{"2^1", 2.0, 1, 2.0, 1e-15},
		{"2^2", 2.0, 2, 4.0, 1e-15},
		{"2^3", 2.0, 3, 8.0, 1e-15},
		{"2^10", 2.0, 10, 1024.0, 1e-12},
		{"3^5", 3.0, 5, 243.0, 1e-12},
		{"10^3", 10.0, 3, 1000.0, 1e-12},
		{"2^-1", 2.0, -1, 0.5, 1e-15},
		{"2^-2", 2.0, -2, 0.25, 1e-15},
		{"10^-2", 10.0, -2, 0.01, 1e-15},
		{"2^20", 2.0, 20, 1048576.0, 1e-8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := IntPower(tt.base, tt.exponent)
			if math.IsNaN(got) {
				t.Errorf("IntPower(%v, %v) = NaN", tt.base, tt.exponent)
				return
			}

			if diff := math.Abs(got - tt.want); diff > tt.tolerance {
				t.Errorf("IntPower(%v, %v) = %v, want %v (diff = %v, tolerance = %v)",
					tt.base, tt.exponent, got, tt.want, diff, tt.tolerance)
			}
		})
	}
}
