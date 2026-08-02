//go:build amd64 && !purego

package simd

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/cwbudde/algo-approx/internal/cpu"
)

// requireAVX2FMA skips the test unless the host really has AVX2 and FMA3.
//
// The tests in this file call expBatch32AVX2 directly, bypassing the dispatch
// in expBatch32. Without the guard they would execute FMA opcodes on a host
// that lacks them and die with SIGILL rather than fail.
func requireAVX2FMA(tb testing.TB) {
	tb.Helper()

	features := cpu.DetectFeatures()
	if !features.HasAVX2 || !features.HasFMA {
		tb.Skip("host lacks AVX2+FMA")
	}
}

// asmExp runs the assembly kernel over a copy of inputs.
func asmExp(tb testing.TB, inputs []float32) []float32 {
	tb.Helper()

	dst := make([]float32, len(inputs))
	if !expBatch32AVX2(dst, inputs) {
		tb.Fatal("expBatch32AVX2 declined")
	}

	return dst
}

// goExp runs the pure-Go kernel over a copy of inputs.
func goExp(inputs []float32) []float32 {
	dst := make([]float32, len(inputs))
	expBatch32Go(dst, inputs)

	return dst
}

// TestExpAVX2MatchesGo is the differential test: the assembly kernel and the
// pure-Go kernel must agree to within 1 ulp on every input.
//
// They are not bit-identical, and cannot be. The Go kernel's multiply-adds are
// separate MULSS/ADDSS pairs — the amd64 compiler does not contract them,
// because the amd64 baseline has no FMA — while the assembly kernel uses
// VFMADD/VFNMADD throughout. The fused forms keep the product to full width, so
// the two disagree in the last bit of the polynomial correction on a minority
// of inputs. Measured over the whole float32 domain the disagreement never
// exceeds one representable step, and roughly 98% of inputs still agree
// bit-for-bit.
func TestExpAVX2MatchesGo(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	rng := rand.New(rand.NewSource(20240802)) //nolint:gosec // deterministic test input.

	const n = 1 << 14

	inputs := make([]float32, n)
	for i := range inputs {
		// Spread across the whole useful domain, overflow and flush-to-zero
		// edges included.
		inputs[i] = float32(rng.Float64()*240 - 120)
	}

	got := asmExp(t, inputs)
	want := goExp(inputs)

	for i, x := range inputs {
		if diff := ulpDiff32(got[i], want[i]); diff > 1 {
			t.Fatalf("exp(%v): asm %v, go %v, %d ulp apart", x, got[i], want[i], diff)
		}
	}
}

// TestExpAVX2MatchesGoExponentSweep walks one input from every float32 binade,
// positive and negative, so the comparison covers the subnormal tail and the
// overflow edge rather than only the range a uniform random draw reaches.
func TestExpAVX2MatchesGoExponentSweep(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	var inputs []float32

	for exp := -149; exp <= 127; exp++ {
		mag := float32(math.Ldexp(1, exp))
		inputs = append(inputs, mag, -mag, mag*1.5, -mag*1.5)
	}

	// Plus a fine grid over the region where the reduction actually does work.
	for x := float32(-105); x <= 90; x += 0.0009765625 {
		inputs = append(inputs, x)
	}

	got := asmExp(t, inputs)
	want := goExp(inputs)

	for i, x := range inputs {
		if diff := ulpDiff32(got[i], want[i]); diff > 1 {
			t.Fatalf("exp(%v): asm %v, go %v, %d ulp apart", x, got[i], want[i], diff)
		}
	}
}

// TestExpAVX2Accuracy pins the assembly kernel against the float64 reference
// directly, so a bug that happens to be shared with the Go kernel still fails.
func TestExpAVX2Accuracy(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	rng := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic test input.

	const n = 1 << 14

	inputs := make([]float32, n)
	for i := range inputs {
		inputs[i] = float32(rng.Float64()*176 - 88)
	}

	got := asmExp(t, inputs)

	for i, x := range inputs {
		want := wantExp32(x)
		if diff := ulpDiff32(got[i], want); diff > 1 {
			t.Fatalf("exp(%v) = %v, want %v (%d ulp)", x, got[i], want, diff)
		}
	}
}

// TestExpAVX2SpecialValues covers everything the clamp and the two-step scaling
// exist for.
func TestExpAVX2SpecialValues(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	inf32 := float32(math.Inf(1))
	nan32 := float32(math.NaN())

	tests := []struct {
		name  string
		input float32
		check func(t *testing.T, got float32)
	}{
		{"+Inf", inf32, wantBits(math.Float32bits(inf32))},
		{"-Inf", -inf32, wantBits(0)},
		{"NaN", nan32, wantNaN()},
		{"+0", 0, wantBits(math.Float32bits(1))},
		{"-0", float32(math.Copysign(0, -1)), wantBits(math.Float32bits(1))},
		{"-104", -104, wantBits(0)},
		{"-1e30", -1e30, wantBits(0)},
		{"-1e38", -1e38, wantBits(0)},

		// The case that proves the two-step 2^k reconstruction. Here k = 128,
		// so a single (k+127)<<23 would build the biased exponent 255 — the
		// Inf/NaN encoding — and return +Inf for a result that is perfectly
		// representable just under MaxFloat32.
		{"88.5 stays finite", 88.5, wantFiniteNear(88.5)},

		// Just above the clamp, exp really must be +Inf.
		{"88.8 overflows", 88.8, wantBits(math.Float32bits(inf32))},

		// The subnormal band must survive: clamping at -87.34 instead of -104
		// would flush all of this to zero.
		{"-100 subnormal", -100, wantFiniteNear(-100)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Run at length 1 so the masked tail path handles it, and at
			// length 8 so the main loop does.
			for _, n := range []int{1, 8} {
				inputs := make([]float32, n)
				for i := range inputs {
					inputs[i] = tt.input
				}

				got := asmExp(t, inputs)
				for i := range got {
					tt.check(t, got[i])
				}
			}
		})
	}
}

// wantFiniteNear asserts the result is neither +Inf nor a flushed zero, and is
// within 1 ulp of the float64 reference.
//
// Both failure modes it rules out are silent: exp(88.5) returning +Inf means
// the 2^k reconstruction was collapsed to a single (k+127)<<23, and exp(-100)
// returning zero means the low clamp was raised above -104 and the subnormal
// band was thrown away.
func wantFiniteNear(x float32) func(t *testing.T, got float32) {
	return func(t *testing.T, got float32) {
		t.Helper()

		if math.IsInf(float64(got), 1) {
			t.Fatalf("exp(%v) overflowed to +Inf: the two-step 2^k scaling is broken", x)
		}

		if got == 0 {
			t.Fatalf("exp(%v) flushed to zero: the low clamp swallowed the subnormal band", x)
		}

		want := wantExp32(x)
		if diff := ulpDiff32(got, want); diff > 1 {
			t.Fatalf("exp(%v) = %v, want %v (%d ulp)", x, got, want, diff)
		}
	}
}

func wantBits(bits uint32) func(t *testing.T, got float32) {
	return func(t *testing.T, got float32) {
		t.Helper()

		if math.Float32bits(got) != bits {
			t.Fatalf("got %v (%#08x), want bit pattern %#08x", got, math.Float32bits(got), bits)
		}
	}
}

func wantNaN() func(t *testing.T, got float32) {
	return func(t *testing.T, got float32) {
		t.Helper()

		// The payload differs between the kernels (the assembly clamp returns
		// the hardware's default quiet NaN), so only NaN-ness is asserted.
		if !math.IsNaN(float64(got)) {
			t.Fatalf("got %v, want NaN", got)
		}
	}
}

// TestExpAVX2Tails walks every length from 0 to 25, so each of the eight mask
// rows is exercised at least three times, at three different block counts.
func TestExpAVX2Tails(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	const guard = 0x7fedcba9 // sentinel bit pattern, no float meaning

	for n := range 26 {
		inputs := make([]float32, n)
		for i := range inputs {
			inputs[i] = float32(i)*0.75 - 9
		}

		// Pad the destination and check the padding is untouched: a masked
		// store that leaked would clobber it.
		dst := make([]float32, n+8)
		for i := range dst {
			dst[i] = math.Float32frombits(guard)
		}

		if !expBatch32AVX2(dst[:n], inputs) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		want := goExp(inputs)

		for i := range n {
			if diff := ulpDiff32(dst[i], want[i]); diff > 1 {
				t.Fatalf("n=%d i=%d: asm %v, go %v, %d ulp apart", n, i, dst[i], want[i], diff)
			}
		}

		for i := n; i < len(dst); i++ {
			if math.Float32bits(dst[i]) != guard {
				t.Fatalf("n=%d: masked store wrote past the end at index %d", n, i)
			}
		}
	}
}

// TestExpAVX2InPlace checks dst == src, which the package documents as
// supported.
func TestExpAVX2InPlace(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	for n := range 26 {
		inputs := make([]float32, n)
		for i := range inputs {
			inputs[i] = float32(i)*0.5 - 6
		}

		want := goExp(inputs)

		buf := make([]float32, n)
		copy(buf, inputs)

		if !expBatch32AVX2(buf, buf) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		for i := range n {
			if math.Float32bits(buf[i]) != math.Float32bits(want[i]) {
				t.Fatalf("n=%d i=%d: in-place %v, out-of-place %v", n, i, buf[i], want[i])
			}
		}
	}
}

// TestExpAVX2Empty checks the n == 0 early exit for both empty and nil slices.
func TestExpAVX2Empty(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	if !expBatch32AVX2(nil, nil) {
		t.Fatal("nil, nil: kernel declined")
	}

	if !expBatch32AVX2([]float32{}, []float32{}) {
		t.Fatal("empty, empty: kernel declined")
	}

	// A shorter source must bound the work; the destination tail stays put.
	dst := []float32{7, 7, 7, 7}
	if !expBatch32AVX2(dst, []float32{0}) {
		t.Fatal("declined")
	}

	if dst[0] != 1 {
		t.Fatalf("dst[0] = %v, want 1", dst[0])
	}

	for i := 1; i < len(dst); i++ {
		if dst[i] != 7 {
			t.Fatalf("dst[%d] = %v, want 7 (untouched)", i, dst[i])
		}
	}
}

// TestExpFloat32DispatchMatchesGo exercises the exported wrapper, so the
// dispatch wiring itself is covered rather than only the raw kernel.
func TestExpFloat32DispatchMatchesGo(t *testing.T) {
	t.Parallel()

	inputs := make([]float32, 1000)
	for i := range inputs {
		inputs[i] = float32(i)*0.17 - 90
	}

	got := make([]float32, len(inputs))
	ExpFloat32(got, inputs)

	want := goExp(inputs)

	for i, x := range inputs {
		if diff := ulpDiff32(got[i], want[i]); diff > 1 {
			t.Fatalf("exp(%v): dispatch %v, go %v, %d ulp apart", x, got[i], want[i], diff)
		}
	}
}

// TestExpAVX2FullDomainDifferential compares the assembly kernel against the
// pure-Go kernel on every one of the 2^32 float32 bit patterns.
//
// It takes a few minutes, so it is opt-in: set ALGO_APPROX_EXHAUSTIVE=1. The
// bounded sweeps above run by default and cover the same shapes; this one is
// the proof that nothing hides between the sampled points.
func TestExpAVX2FullDomainDifferential(t *testing.T) {
	t.Parallel()
	requireAVX2FMA(t)

	if os.Getenv("ALGO_APPROX_EXHAUSTIVE") != "1" {
		t.Skip("set ALGO_APPROX_EXHAUSTIVE=1 to sweep all 2^32 float32 inputs")
	}

	const block = 1 << 12

	var (
		buf     = make([]float32, block)
		gotBuf  = make([]float32, block)
		wantBuf = make([]float32, block)
		maxULP  int64
		exact   uint64
		total   uint64
	)

	for base := uint64(0); base < 1<<32; base += block {
		for j := range block {
			buf[j] = math.Float32frombits(uint32(base + uint64(j))) //nolint:gosec // base+j < 2^32 by the loop bound.
		}

		expBatch32AVX2(gotBuf, buf)
		expBatch32Go(wantBuf, buf)

		for j := range block {
			if math.IsNaN(float64(buf[j])) {
				// Both kernels return NaN; the payloads differ by design.
				if !math.IsNaN(float64(gotBuf[j])) {
					t.Fatalf("exp(NaN %#08x) = %v, want NaN", math.Float32bits(buf[j]), gotBuf[j])
				}

				continue
			}

			total++

			if math.Float32bits(gotBuf[j]) == math.Float32bits(wantBuf[j]) {
				exact++

				continue
			}

			diff := ulpDiff32(gotBuf[j], wantBuf[j])
			if diff > 1 {
				t.Fatalf("exp(%v): asm %v, go %v, %d ulp apart", buf[j], gotBuf[j], wantBuf[j], diff)
			}

			maxULP = max(maxULP, diff)
		}
	}

	t.Logf("full domain: max %d ulp, %d/%d (%.2f%%) bit-identical",
		maxULP, exact, total, 100*float64(exact)/float64(total))
}
