package simd

import (
	"math"
	"testing"
)

// ulpDiff32 returns the distance between two float32 values measured in
// representable steps. Both must be finite and non-NaN. The monotone-integer
// mapping used here treats the whole float32 line, subnormals included, as a
// single evenly spaced grid, which is exactly what "ulp" should mean when the
// result of interest (exp(-90), say) is subnormal.
func ulpDiff32(a, b float32) int64 {
	keyA := orderedKey32(a)
	keyB := orderedKey32(b)

	if keyA > keyB {
		return keyA - keyB
	}

	return keyB - keyA
}

// orderedKey32 maps float32 bit patterns onto a monotone int64 key.
func orderedKey32(f float32) int64 {
	bits := math.Float32bits(f)
	if bits&signMask32 != 0 {
		return -int64(bits &^ signMask32)
	}

	return int64(bits)
}

// wantExp32 is the reference: math.Exp in float64, rounded once to float32.
func wantExp32(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

// runExp is a convenience wrapper: one input, one output.
func runExp(x float32) float32 {
	dst := make([]float32, 1)
	ExpFloat32(dst, []float32{x})

	return dst[0]
}

// runTanhLogCosh is a convenience wrapper: one input, two outputs.
func runTanhLogCosh(x float32) (float32, float32) {
	gotTanh := make([]float32, 1)
	gotLogCosh := make([]float32, 1)
	TanhLogCoshFloat32(gotTanh, gotLogCosh, []float32{x})

	return gotTanh[0], gotLogCosh[0]
}

func TestExpConstantBits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		got  float32
		want uint32
	}{
		{"log2e", expLog2e, 0x3fb8aa3b},
		{"C1", expC1, 0x3f318000},
		{"C2", expC2, 0xb95e8083},
		{"P4", expP4, 0x3ab51233},
		{"P3", expP3, 0x3c091ceb},
		{"P2", expP2, 0x3d2aac79},
		{"P1", expP1, 0x3e2aaa49},
		{"P0", expP0, 0x3efffffe},
		{"HI", expHI, 0x42b17218},
		{"LO", expLO, 0xc2d00000},
		{"roundMagic", expRoundMagic, 0x4b400000},
		{"ln2f", ln2f, 0x3f317218},
		{"tanhBranch", tanhBranch32, 0x3f200000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := math.Float32bits(tt.got); got != tt.want {
				t.Errorf("%s = 0x%08x, want 0x%08x", tt.name, got, tt.want)
			}
		})
	}
}

// TestExpCodyWaiteStep1Exact pins the property the whole range reduction rests
// on: fx*C1 is exact in float32 for every fx the kernel can generate.
func TestExpCodyWaiteStep1Exact(t *testing.T) {
	t.Parallel()

	for k := -150; k <= 128; k++ {
		fx := float32(k)
		got := float64(fx * expC1)
		want := float64(fx) * float64(float32(expC1))

		if got != want {
			t.Fatalf("fx=%v: fx*C1 rounded (%v != %v)", fx, got, want)
		}
	}
}

func TestExpAccuracy(t *testing.T) {
	t.Parallel()

	const (
		samples  = 200000
		maxAllow = 2
	)

	src := make([]float32, samples)
	for i := range src {
		// Sweep the entire usable domain, including the subnormal tail.
		src[i] = float32(expLO) + float32(i)*float32(expHI-expLO)/float32(samples-1)
	}

	dst := make([]float32, samples)
	ExpFloat32(dst, src)

	var (
		worst    int64
		worstAt  float32
		worstGot float32
	)

	for i, x := range src {
		want := wantExp32(x)
		if diff := ulpDiff32(dst[i], want); diff > worst {
			worst, worstAt, worstGot = diff, x, dst[i]
		}
	}

	t.Logf("exp: worst error %d ulp at x=%v (got %v, want %v)", worst, worstAt, worstGot, wantExp32(worstAt))

	if worst > maxAllow {
		t.Errorf("exp worst error %d ulp exceeds %d", worst, maxAllow)
	}
}

// TestExpAccuracyNearOverflow walks the top of the range one float32 at a
// time, where a single-step 2^k reconstruction would return +Inf.
func TestExpAccuracyNearOverflow(t *testing.T) {
	t.Parallel()

	const maxAllow = 2

	var worst int64

	for bits := math.Float32bits(88.02); bits <= math.Float32bits(expHI); bits++ {
		x := math.Float32frombits(bits)

		got := runExp(x)
		if math.IsInf(float64(got), 1) && x < expHI {
			t.Fatalf("exp(%v) = +Inf, want finite", x)
		}

		if diff := ulpDiff32(got, wantExp32(x)); diff > worst && x < expHI {
			worst = diff
		}
	}

	t.Logf("exp near overflow: worst error %d ulp", worst)

	if worst > maxAllow {
		t.Errorf("worst error %d ulp exceeds %d", worst, maxAllow)
	}
}

// checkPlusInf, checkExactZero, checkNaN and checkExactly are pulled out of the
// tables below so the tables stay readable and the test functions stay short.
func checkPlusInf(t *testing.T, got float32) {
	t.Helper()

	if !math.IsInf(float64(got), 1) {
		t.Errorf("got %v, want +Inf", got)
	}
}

func checkExactZero(t *testing.T, got float32) {
	t.Helper()

	if math.Float32bits(got) != 0 {
		t.Errorf("got %v (0x%08x), want exactly +0", got, math.Float32bits(got))
	}
}

func checkNaN(t *testing.T, got float32) {
	t.Helper()

	if !math.IsNaN(float64(got)) {
		t.Errorf("got %v, want NaN", got)
	}
}

func checkBits(want uint32) func(*testing.T, float32) {
	return func(t *testing.T, got float32) {
		t.Helper()

		if math.Float32bits(got) != want {
			t.Errorf("got %v (0x%08x), want 0x%08x", got, math.Float32bits(got), want)
		}
	}
}

func TestExpSpecialValues(t *testing.T) {
	t.Parallel()

	one := math.Float32bits(1)

	tests := []struct {
		name  string
		in    float32
		check func(*testing.T, float32)
	}{
		{"+Inf", float32(math.Inf(1)), checkPlusInf},
		{"-Inf", float32(math.Inf(-1)), checkExactZero},
		{"NaN", float32(math.NaN()), checkNaN},
		{"+0", 0, checkBits(one)},
		{"-0", float32(math.Copysign(0, -1)), checkBits(one)},
		{"LO", expLO, checkExactZero},
		{"-1e30", -1e30, checkExactZero},
		{"1e30", 1e30, checkPlusInf},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.check(t, runExp(tt.in))
		})
	}
}

// TestExpSubnormalBand is the counter-test to clamping at -87.34: every input
// between the smallest normal and the true zero threshold must still produce
// its correct subnormal, not a flushed zero.
func TestExpSubnormalBand(t *testing.T) {
	t.Parallel()

	const maxAllow = 2

	for x := float32(-87.0); x >= -104; x -= 0.125 {
		want := wantExp32(x)
		got := runExp(x)

		if diff := ulpDiff32(got, want); diff > maxAllow {
			t.Errorf("exp(%v) = %v (0x%08x), want %v (0x%08x): %d ulp",
				x, got, math.Float32bits(got), want, math.Float32bits(want), diff)
		}
	}

	// The specific case named in the design notes.
	if got, want := runExp(-90), wantExp32(-90); got != want {
		t.Errorf("exp(-90) = %v (0x%08x), want %v (0x%08x)",
			got, math.Float32bits(got), want, math.Float32bits(want))
	}

	if want := wantExp32(-90); want == 0 || want >= math.SmallestNonzeroFloat32*(1<<24) {
		t.Errorf("exp(-90) reference %v is not in the subnormal band; test is not testing what it claims", want)
	}
}

func TestTanhAccuracy(t *testing.T) {
	t.Parallel()

	const (
		samples  = 200000
		maxAllow = 2
	)

	src := make([]float32, samples)
	for i := range src {
		src[i] = -12 + float32(i)*24/float32(samples-1)
	}

	gotTanh := make([]float32, samples)
	gotLogCosh := make([]float32, samples)
	TanhLogCoshFloat32(gotTanh, gotLogCosh, src)

	var (
		worst   int64
		worstAt float32
	)

	for i, x := range src {
		want := float32(math.Tanh(float64(x)))
		if diff := ulpDiff32(gotTanh[i], want); diff > worst {
			worst, worstAt = diff, x
		}
	}

	t.Logf("tanh: worst error %d ulp at x=%v", worst, worstAt)

	if worst > maxAllow {
		t.Errorf("tanh worst error %d ulp exceeds %d", worst, maxAllow)
	}
}

// refLogCosh64 is an accurate float64 log(cosh(x)).
//
// The obvious reference, |x| - ln2 + log1p(exp(-2|x|)), cannot be used near
// zero: it cancels catastrophically there, and at x = 1e-8 it is wrong by
// hundreds of millions of float32 ulp while the kernel is right. The series is
// used below the branch point for the same reason the kernel uses one.
func refLogCosh64(x float64) float64 {
	a := math.Abs(x)
	if a < tanhBranch32 {
		z := a * a

		return z * (0.5 + z*(-1.0/12.0+z*(1.0/45.0+z*(-17.0/2520.0+z*(31.0/14175.0+
			z*(-691.0/935550.0+z*(10922.0/42567525.0+z*(-929569.0/10216206000.0+
				z*(3202291.0/97692469875.0+z*(-221930581.0/18561569276250.0))))))))))
	}

	return a - math.Ln2 + math.Log1p(math.Exp(-2*a))
}

// TestLogCoshAccuracy sweeps several ranges separately so the seam at the
// branch point, which is the worst region, is sampled densely rather than
// being lost in a coarse sweep of the whole domain.
func TestLogCoshAccuracy(t *testing.T) {
	t.Parallel()

	// 4 ulp is the seam just above |x| = 0.625, where a - ln2 cancels against
	// log1p(u); everywhere else the kernel is within 2.
	const (
		samples  = 100001
		maxAllow = 4
	)

	ranges := []struct {
		name   string
		lo, hi float64
	}{
		{"tiny", -1e-4, 1e-4},
		{"small", -0.625, 0.625},
		{"seam", 0.6, 0.7},
		{"mid", -12, 12},
		{"wide", -100, 100},
	}

	for _, rng := range ranges {
		t.Run(rng.name, func(t *testing.T) {
			t.Parallel()

			src := make([]float32, samples)
			for i := range src {
				src[i] = float32(rng.lo + (rng.hi-rng.lo)*float64(i)/float64(samples-1))
			}

			gotTanh := make([]float32, samples)
			gotLogCosh := make([]float32, samples)
			TanhLogCoshFloat32(gotTanh, gotLogCosh, src)

			var (
				worst   int64
				worstAt float32
			)

			for i, x := range src {
				want := float32(refLogCosh64(float64(x)))
				if diff := ulpDiff32(gotLogCosh[i], want); diff > worst {
					worst, worstAt = diff, x
				}
			}

			t.Logf("logCosh %s: worst error %d ulp at x=%v", rng.name, worst, worstAt)

			if worst > maxAllow {
				t.Errorf("logCosh worst error %d ulp exceeds %d", worst, maxAllow)
			}
		})
	}
}

// TestTanhOddSymmetryBitExact is the strong form: not "close to symmetric" but
// identical down to the bit, over the whole interesting range.
func TestTanhOddSymmetryBitExact(t *testing.T) {
	t.Parallel()

	const samples = 20000

	pos := make([]float32, samples)
	neg := make([]float32, samples)

	for i := range pos {
		x := float32(i+1) * 25 / float32(samples)
		pos[i] = x
		neg[i] = -x
	}

	dtPos := make([]float32, samples)
	dlPos := make([]float32, samples)
	dtNeg := make([]float32, samples)
	dlNeg := make([]float32, samples)

	TanhLogCoshFloat32(dtPos, dlPos, pos)
	TanhLogCoshFloat32(dtNeg, dlNeg, neg)

	for i := range pos {
		if got, want := math.Float32bits(dtNeg[i]), math.Float32bits(dtPos[i])^signMask32; got != want {
			t.Fatalf("tanh(%v): bits 0x%08x, want 0x%08x (negation of tanh(%v))",
				neg[i], got, want, pos[i])
		}

		if got, want := math.Float32bits(dlNeg[i]), math.Float32bits(dlPos[i]); got != want {
			t.Fatalf("logCosh(%v): bits 0x%08x, want 0x%08x (even)", neg[i], got, want)
		}
	}
}

// TestTanhSaturation documents the float32 crossover empirically rather than
// trusting the float64 constant of 19.0625.
func TestTanhSaturation(t *testing.T) {
	t.Parallel()

	const (
		lastBelow = 9.010913 // 0x41102cb3, tanh still 0.99999994
		firstAt   = 9.010914 // 0x41102cb4, tanh rounds to 1
	)

	if got := float32(math.Tanh(lastBelow)); got == 1 {
		t.Fatalf("reference tanh(%v) = %v, expected below 1; the documented crossover moved", float32(lastBelow), got)
	}

	if got := float32(math.Tanh(firstAt)); got != 1 {
		t.Fatalf("reference tanh(%v) = %v, expected exactly 1; the documented crossover moved", float32(firstAt), got)
	}

	if got, _ := runTanhLogCosh(firstAt); got != 1 {
		t.Errorf("kernel tanh(%v) = %v, want exactly 1", float32(firstAt), got)
	}

	// And it must stay saturated all the way out, without any explicit
	// saturation constant.
	for x := float32(9.1); x < 200; x *= 1.01 {
		if got, _ := runTanhLogCosh(x); got != 1 {
			t.Fatalf("kernel tanh(%v) = %v, want exactly 1", x, got)
		}
	}
}

func TestTanhLogCoshSpecialValues(t *testing.T) {
	t.Parallel()

	var (
		one      = math.Float32bits(1)
		minusOne = math.Float32bits(-1)
		negZero  = uint32(signMask32)
	)

	tests := []struct {
		name        string
		in          float32
		checkTanh   func(*testing.T, float32)
		checkLogCos func(*testing.T, float32)
	}{
		{"NaN", float32(math.NaN()), checkNaN, checkNaN},
		{"+Inf", float32(math.Inf(1)), checkBits(one), checkPlusInf},
		{"-Inf", float32(math.Inf(-1)), checkBits(minusOne), checkPlusInf},
		{"+0", 0, checkExactZero, checkExactZero},
		{"-0", float32(math.Copysign(0, -1)), checkBits(negZero), checkExactZero},
		// log(cosh(1e30)) stays finite because cosh is never formed.
		{"1e30", 1e30, checkBits(one), checkBits(math.Float32bits(float32(1e30 - math.Ln2)))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			gotTanh, gotLogCosh := runTanhLogCosh(tt.in)
			tt.checkTanh(t, gotTanh)
			tt.checkLogCos(t, gotLogCosh)
		})
	}
}

// TestBlockAndTailAgree checks that the eight-wide block body and the scalar
// tail compute the same thing, at every length up to and past two blocks.
func TestBlockAndTailAgree(t *testing.T) {
	t.Parallel()

	const maxLen = 3*unroll + 1

	src := make([]float32, maxLen)
	for i := range src {
		src[i] = float32(i)*0.37 - 4
	}

	for n := range maxLen + 1 {
		dst := make([]float32, n)
		ExpFloat32(dst, src[:n])

		gotTanh := make([]float32, n)
		gotLogCosh := make([]float32, n)
		TanhLogCoshFloat32(gotTanh, gotLogCosh, src[:n])

		for i := range n {
			if want := expKernel32(src[i]); dst[i] != want {
				t.Errorf("n=%d i=%d: exp = %v, want %v", n, i, dst[i], want)
			}

			wantT, wantL := tanhLogCoshKernel32(src[i])
			if gotTanh[i] != wantT || gotLogCosh[i] != wantL {
				t.Errorf("n=%d i=%d: (tanh, logCosh) = (%v, %v), want (%v, %v)",
					n, i, gotTanh[i], gotLogCosh[i], wantT, wantL)
			}
		}
	}
}

func TestEmptySlices(t *testing.T) {
	t.Parallel()

	ExpFloat32(nil, nil)
	ExpFloat32([]float32{}, []float32{})
	TanhLogCoshFloat32(nil, nil, nil)
	TanhLogCoshFloat32([]float32{}, []float32{}, []float32{})

	// A non-empty destination with an empty source must be left alone.
	dst := []float32{7, 7}
	ExpFloat32(dst, nil)

	if dst[0] != 7 || dst[1] != 7 {
		t.Errorf("dst = %v, want untouched", dst)
	}
}

func TestInPlace(t *testing.T) {
	t.Parallel()

	const n = 3*unroll + 5

	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)*0.21 - 3
	}

	want := make([]float32, n)
	ExpFloat32(want, src)

	buf := make([]float32, n)
	copy(buf, src)
	ExpFloat32(buf, buf)

	for i := range n {
		if buf[i] != want[i] {
			t.Fatalf("in-place exp i=%d: got %v, want %v", i, buf[i], want[i])
		}
	}

	wantT := make([]float32, n)
	wantL := make([]float32, n)
	TanhLogCoshFloat32(wantT, wantL, src)

	bufT := make([]float32, n)
	copy(bufT, src)

	bufL := make([]float32, n)
	TanhLogCoshFloat32(bufT, bufL, bufT)

	for i := range n {
		if bufT[i] != wantT[i] || bufL[i] != wantL[i] {
			t.Fatalf("in-place tanh/logCosh i=%d: got (%v, %v), want (%v, %v)",
				i, bufT[i], bufL[i], wantT[i], wantL[i])
		}
	}
}

func TestShortDestinationPanics(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		call func()
	}{
		{"exp", func() { ExpFloat32(make([]float32, 3), make([]float32, 4)) }},
		{"tanh dst", func() {
			TanhLogCoshFloat32(make([]float32, 3), make([]float32, 4), make([]float32, 4))
		}},
		{"logCosh dst", func() {
			TanhLogCoshFloat32(make([]float32, 4), make([]float32, 3), make([]float32, 4))
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			defer func() {
				if recover() == nil {
					t.Error("expected a panic, got none")
				}
			}()

			tt.call()
		})
	}
}

// TestLongerDestinationIsUntouched pins the documented contract that exactly
// len(src) elements are written.
func TestLongerDestinationIsUntouched(t *testing.T) {
	t.Parallel()

	dst := make([]float32, unroll+4)
	for i := range dst {
		dst[i] = -99
	}

	ExpFloat32(dst, make([]float32, unroll))

	for i := unroll; i < len(dst); i++ {
		if dst[i] != -99 {
			t.Errorf("dst[%d] = %v, want -99 (untouched)", i, dst[i])
		}
	}
}

func BenchmarkExpFloat32(b *testing.B) {
	const n = 4096

	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)*(20.0/n) - 10
	}

	dst := make([]float32, n)

	b.SetBytes(n * 4)
	b.ResetTimer()

	for b.Loop() {
		ExpFloat32(dst, src)
	}
}

func BenchmarkTanhLogCoshFloat32(b *testing.B) {
	const n = 4096

	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)*(20.0/n) - 10
	}

	dstTanh := make([]float32, n)
	dstLogCosh := make([]float32, n)

	b.SetBytes(n * 4)
	b.ResetTimer()

	for b.Loop() {
		TanhLogCoshFloat32(dstTanh, dstLogCosh, src)
	}
}
