//go:build amd64 && !purego

package simd

import (
	"math"
	"testing"
)

// TestZZDiagDrift is a throwaway: it reports the true maximum asm-vs-go drift
// over every float32 bit pattern, and where it occurs, without asserting.
func TestZZDiagDrift(t *testing.T) {
	requireAVX2FMA(t)

	const block = 1 << 12

	buf := make([]float32, block)
	gotT, gotL := make([]float32, block), make([]float32, block)
	wantT, wantL := make([]float32, block), make([]float32, block)

	var maxT, maxL int64
	var atT, atL float32

	for base := uint64(0); base < 1<<32; base += block {
		for j := range block {
			buf[j] = math.Float32frombits(uint32(base + uint64(j)))
		}

		tanhLogCoshBatch32AVX2(gotT, gotL, buf)
		tanhLogCoshBatch32Go(wantT, wantL, buf)

		for j := range block {
			if math.IsNaN(float64(buf[j])) {
				continue
			}
			if d := ulpDiff32(gotT[j], wantT[j]); d > maxT {
				maxT, atT = d, buf[j]
			}
			if d := ulpDiff32(gotL[j], wantL[j]); d > maxL {
				maxL, atL = d, buf[j]
			}
		}
	}

	t.Logf("FULL DOMAIN asm-vs-go drift: tanh max %d ulp at x=%v | logCosh max %d ulp at x=%v",
		maxT, atT, maxL, atL)
}

// TestZZDiagTanhSeam asks whether the asm or the Go kernel is closer to truth
// where tanh's drift is worst.
func TestZZDiagTanhSeam(t *testing.T) {
	requireAVX2FMA(t)

	const n = 400001
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(0.60 + 0.10*float64(i)/float64(n-1))
	}

	gotT, _ := asmHyp(t, src)
	wantT, _ := goHyp(src)

	var asmWorst, goWorst int64
	var asmSum, goSum float64
	var asmBetter, goBetter, tie int

	for i, x := range src {
		ref := float32(math.Tanh(float64(x)))
		a, g := ulpDiff32(gotT[i], ref), ulpDiff32(wantT[i], ref)
		asmWorst, goWorst = max(asmWorst, a), max(goWorst, g)
		asmSum += float64(a)
		goSum += float64(g)
		switch {
		case a < g:
			asmBetter++
		case g < a:
			goBetter++
		default:
			tie++
		}
	}

	t.Logf("tanh over [0.60,0.70]: asm worst %d (mean %.4f) | go worst %d (mean %.4f)",
		asmWorst, asmSum/n, goWorst, goSum/n)
	t.Logf("  asm closer: %d  go closer: %d  equal: %d", asmBetter, goBetter, tie)
}
