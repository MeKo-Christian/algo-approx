package reference

import (
	"testing"

	approx "github.com/cwbudde/algo-approx"
)

func TestMeasureAccuracyBasic(t *testing.T) {
	t.Parallel()

	samples := []float64{1, 2, 3, 4}

	m := MeasureAccuracy[float64](samples,
		func(x float64) float64 { return x },
		func(x float64) float64 { return x },
	)
	if m.MaxAbsError != 0 || m.MaxRelError != 0 {
		t.Fatalf("expected zero error, got %+v", m)
	}

	if !approx.PrecisionBalanced.IsValid() {
		t.Fatalf("precision validity broke")
	}
}
