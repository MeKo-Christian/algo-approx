//go:build arm64 && !purego

package simd

import (
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/algo-approx/internal/cpu"
)

// requireNEON skips the test unless the host really reports Advanced SIMD.
//
// In practice this never skips: ASIMD is mandatory in ARMv8-A and every
// instruction the kernel uses is base ASIMD with no optional extension behind
// it. The guard is here so that a host whose feature detection says otherwise
// — an emulator, or a kernel that masks the HWCAP bit — skips rather than
// producing a confusing failure.
func requireNEON(tb testing.TB) {
	tb.Helper()

	if !cpu.DetectFeatures().HasNEON {
		tb.Skip("host does not report NEON")
	}
}

// asmExp runs the assembly kernel over a copy of inputs.
func asmExp(tb testing.TB, inputs []float32) []float32 {
	tb.Helper()

	dst := make([]float32, len(inputs))
	if !expBatch32NEON(dst, inputs) {
		tb.Fatal("expBatch32NEON declined")
	}

	return dst
}

// goExp runs the pure-Go kernel over a copy of inputs.
func goExp(inputs []float32) []float32 {
	dst := make([]float32, len(inputs))
	expBatch32Go(dst, inputs)

	return dst
}

// TestExpNEONMatchesGo is the differential test: the assembly kernel and the
// pure-Go kernel must agree to within 1 ulp on every input.
//
// The tolerance is 1 ulp rather than 0 for a different reason than on amd64.
// There, the two kernels differ because the compiler refuses to contract
// multiply-adds and the assembly fuses them. On arm64 the compiler *does*
// contract, so the polynomial matches instruction for instruction. What is
// left is the range reduction: expRint32 rounds via the add-magic-constant
// trick, which on arm64 becomes a fused fma(x, log2e, magic) and therefore
// rounds the product exactly once, while the kernel here uses FRINTN on the
// already-rounded product. The two can land on opposite sides of a tie, and an
// fx that differs by one puts the polynomial slightly outside |r| <= ln2/2.
// See the full-domain test at the bottom of this file for how often that
// actually happens.
func TestExpNEONMatchesGo(t *testing.T) {
	t.Parallel()
	requireNEON(t)

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

// TestExpNEONMatchesGoExponentSweep walks one input from every float32 binade,
// positive and negative, so the comparison covers the subnormal tail and the
// overflow edge rather than only the range a uniform random draw reaches.
func TestExpNEONMatchesGoExponentSweep(t *testing.T) {
	t.Parallel()
	requireNEON(t)

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

// TestExpNEONAccuracy pins the assembly kernel against the float64 reference
// directly, so a bug that happens to be shared with the Go kernel still fails.
func TestExpNEONAccuracy(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	const (
		samples  = 200000
		maxAllow = 2
	)

	inputs := make([]float32, samples)
	for i := range inputs {
		inputs[i] = float32(expLO) + float32(i)*float32(expHI-expLO)/float32(samples-1)
	}

	got := asmExp(t, inputs)

	var (
		worst   int64
		worstAt float32
	)

	for i, x := range inputs {
		if diff := ulpDiff32(got[i], wantExp32(x)); diff > worst {
			worst, worstAt = diff, x
		}
	}

	t.Logf("neon exp: worst error %d ulp at x=%v", worst, worstAt)

	if worst > maxAllow {
		t.Errorf("neon exp worst error %d ulp exceeds %d", worst, maxAllow)
	}
}

// TestExpNEONSpecialValues covers the inputs where the two-step 2^k
// reconstruction and the clamps are what produce the answer.
func TestExpNEONSpecialValues(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	inf32 := float32(math.Inf(1))
	nan32 := float32(math.NaN())

	inputs := []float32{
		0, -0,
		1, -1,
		88.7228391,  // ln(MaxFloat32): the largest finite result
		88.8,        // just past it: must overflow to +Inf
		-87.33654,   // the smallest normal result
		-103.972084, // the smallest subnormal result
		-104, -120,  // flush to zero
		inf32, -inf32, nan32,
	}

	got := asmExp(t, inputs)

	for i, x := range inputs {
		switch {
		case math.IsNaN(float64(x)):
			// AArch64's FMIN/FMAX return the default quiet NaN rather than the
			// operand, so only NaN-ness is asserted, not the payload.
			if !math.IsNaN(float64(got[i])) {
				t.Errorf("exp(NaN) = %v, want NaN", got[i])
			}
		case math.IsInf(float64(x), 1):
			if !math.IsInf(float64(got[i]), 1) {
				t.Errorf("exp(+Inf) = %v, want +Inf", got[i])
			}
		case math.IsInf(float64(x), -1):
			if got[i] != 0 {
				t.Errorf("exp(-Inf) = %v, want 0", got[i])
			}
		default:
			if diff := ulpDiff32(got[i], wantExp32(x)); diff > 2 {
				t.Errorf("exp(%v) = %v, want %v, %d ulp apart", x, got[i], wantExp32(x), diff)
			}
		}
	}
}

// TestExpNEONTails walks every length from 0 to 25 and checks that the
// destination past the end is untouched.
//
// This is the highest-risk test in the file. The AVX2 kernel gets its tail from
// VMASKMOVPS, which cannot write a masked-off lane by construction; the NEON
// kernel has no masked store and stages the 1..3 element tail through a scratch
// buffer in its own frame, copying elements out one at a time. A miscounted
// copy loop would write one float past the slice, and the guard pattern below
// is what catches it.
func TestExpNEONTails(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	const guard = 0x7fedcba9 // sentinel bit pattern, no float meaning

	for n := range 26 {
		inputs := make([]float32, n)
		for i := range inputs {
			inputs[i] = float32(i)*0.75 - 9
		}

		dst := make([]float32, n+8)
		for i := range dst {
			dst[i] = math.Float32frombits(guard)
		}

		if !expBatch32NEON(dst[:n], inputs) {
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
				t.Fatalf("n=%d: kernel wrote past the end at index %d", n, i)
			}
		}
	}
}

// TestExpNEONInPlace checks dst == src, which the package documents as
// supported.
//
// It is worth more here than on amd64. The cheap alternative to a scratch
// buffer for the tail is to reprocess a full vector ending at n, which is
// correct for disjoint slices and silently returns exp(exp(x)) for up to three
// elements when dst == src. This test is what makes that shortcut fail.
func TestExpNEONInPlace(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	for n := range 26 {
		inputs := make([]float32, n)
		for i := range inputs {
			inputs[i] = float32(i)*0.5 - 6
		}

		want := asmExp(t, inputs)

		buf := make([]float32, n)
		copy(buf, inputs)

		if !expBatch32NEON(buf, buf) {
			t.Fatalf("n=%d: kernel declined", n)
		}

		for i := range n {
			if math.Float32bits(buf[i]) != math.Float32bits(want[i]) {
				t.Fatalf("n=%d i=%d: in-place %v, out-of-place %v", n, i, buf[i], want[i])
			}
		}
	}
}

// TestExpNEONEmpty checks the n == 0 early exit for both empty and nil slices.
func TestExpNEONEmpty(t *testing.T) {
	t.Parallel()
	requireNEON(t)

	if !expBatch32NEON(nil, nil) {
		t.Fatal("nil, nil: kernel declined")
	}

	if !expBatch32NEON([]float32{}, []float32{}) {
		t.Fatal("empty, empty: kernel declined")
	}

	// A shorter source must bound the work; the destination tail stays put.
	dst := []float32{7, 7, 7, 7}
	if !expBatch32NEON(dst, []float32{0}) {
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

// TestDispatchSelectsNEONOnCapableHost asserts that the dispatch flag really
// resolved to the assembly, so the suite cannot pass green while the kernel
// never executes.
func TestDispatchSelectsNEONOnCapableHost(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	if !features.HasNEON || features.ForceGeneric {
		t.Skip("host does not select the NEON path")
	}

	if !expUseNEON {
		t.Fatal("host reports NEON but expBatch32 dispatches to the Go kernel")
	}
}

// TestNEONWordEncodings disassembles this package's own kernel and asserts that
// the hand-encoded instructions decode to the mnemonics they are supposed to.
//
// Every arithmetic instruction in EXPBODY is a WORD, because Go's arm64
// assembler has no vector floating-point arithmetic beyond VFMLA/VFMLS (see
// neon_arm64.h). A WORD is opaque to the assembler and to `go vet`: a mistyped
// hex digit produces a different instruction, or a reserved encoding, with no
// diagnostic at all. Nothing else in the suite can distinguish "the encoding is
// right" from "the encoding is wrong in a way the test grid does not reach".
//
// go tool objdump decodes via golang.org/x/arch/arm64, an implementation of the
// encoding independent of the assembler that produced the bytes, so agreement
// between the two is real evidence rather than a tautology.
func TestNEONWordEncodings(t *testing.T) {
	t.Parallel()

	_, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go tool not on PATH")
	}

	// The test compiles a binary of its own rather than disassembling
	// os.Executable(). Under `go test` the running binary lives in the build
	// work directory, and objdump reads no symbols out of it — it exits 0 with
	// empty output, which would make this test silently vacuous. Building
	// explicitly costs a second on a cold cache and nothing afterwards.
	bin := filepath.Join(t.TempDir(), "kernel.test")

	out, err := exec.Command("go", "test", "-c", "-o", bin, ".").CombinedOutput()
	if err != nil {
		t.Skipf("could not build a binary to disassemble (%v): %s", err, out)
	}

	out, err = exec.Command("go", "tool", "objdump", "-s", "expBatch32NEON", bin).CombinedOutput()
	if err != nil {
		t.Skipf("objdump failed (%v): %s", err, out)
	}

	text := string(out)
	if !strings.Contains(text, "TEXT") {
		t.Fatalf("objdump found no expBatch32NEON in %s; output was %q", bin, text)
	}

	// One occurrence per EXPBODY expansion, and EXPBODY is expanded twice: once
	// for the main loop and once for the scratch-buffer tail. Asserting "at
	// least twice" rather than an exact count keeps the test from breaking on
	// an unrelated edit to the body while still proving each encoding decodes.
	for _, mnemonic := range []string{
		"FMAX", "FMIN", "FMUL", "FADD", "FRINTN", "FCVTZS", "VSSHR",
	} {
		// Matching on a leading tab pins the mnemonic to the instruction
		// column, so "FMUL" cannot be satisfied by a mention inside a symbol
		// name or a file path. A trailing space rather than " V" is deliberate:
		// VSSHR's first operand is an immediate, not a register.
		if n := strings.Count(text, "\t"+mnemonic+" "); n < 2 {
			t.Errorf("disassembly contains %d %s instructions, want at least 2:\n%s",
				n, mnemonic, text)
		}
	}

	// An encoding that is not a valid instruction at all disassembles to "?".
	//
	// Only the body is checked. The linker pads the end of the function to an
	// alignment boundary with zero words, and a zero word is not a valid AArch64
	// instruction, so it disassembles to "?" too — scanning the whole listing
	// would fail on every correct kernel.
	body := text
	if cut := strings.LastIndex(body, "\tRET"); cut >= 0 {
		body = body[:cut]
	}

	if strings.Contains(body, "\t?\t") {
		t.Errorf("disassembly contains an undecodable word:\n%s", body)
	}
}

// TestExpNEONFullDomainDifferential compares the assembly kernel against the
// pure-Go kernel on every one of the 2^32 float32 bit patterns.
//
// It takes a few minutes, so it is opt-in: set ALGO_APPROX_EXHAUSTIVE=1. The
// bounded sweeps above run by default and cover the same shapes; this one is
// the proof that nothing hides between the sampled points.
func TestExpNEONFullDomainDifferential(t *testing.T) {
	t.Parallel()
	requireNEON(t)

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

		expBatch32NEON(gotBuf, buf)
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
