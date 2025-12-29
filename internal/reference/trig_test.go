package reference

import (
	"math"
	"testing"
)

// TestSin tests the reference sine wrapper
func TestSin(t *testing.T) {
	tests := []float64{0, math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2}
	for _, x := range tests {
		got64 := Sin(x)
		want64 := math.Sin(x)
		if got64 != want64 {
			t.Errorf("Sin(%v) = %v, want %v", x, got64, want64)
		}

		// For float32, we need to accept precision loss from conversion
		got32 := Sin(float32(x))
		want32 := float32(math.Sin(float64(float32(x))))
		if got32 != want32 {
			t.Errorf("Sin(%v) = %v, want %v", float32(x), got32, want32)
		}
	}
}

// TestCos tests the reference cosine wrapper
func TestCos(t *testing.T) {
	tests := []float64{0, math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2}
	for _, x := range tests {
		got64 := Cos(x)
		want64 := math.Cos(x)
		if got64 != want64 {
			t.Errorf("Cos(%v) = %v, want %v", x, got64, want64)
		}

		// For float32, we need to accept precision loss from conversion
		got32 := Cos(float32(x))
		want32 := float32(math.Cos(float64(float32(x))))
		if got32 != want32 {
			t.Errorf("Cos(%v) = %v, want %v", float32(x), got32, want32)
		}
	}
}
