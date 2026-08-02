// Shared AVX2 float32 exp machinery: the constants, the eight-lane tail mask
// table, and the EXPBODY macro.
//
// This header exists because the fused tanh/log(cosh) kernel needs exactly the
// same exponential as expBatch32AVX2 does. tanh and log(cosh) are both built
// from u = exp(-2|x|), and the whole reason the fused kernel is worth writing
// is that the two outputs share that one evaluation. Keeping the body in one
// macro means the two .s files cannot drift into computing subtly different
// exponentials, which is the kind of divergence that shows up only as a
// handful of ulp in one function and is very hard to attribute later.
//
// The symbols below are all `<>`-scoped, i.e. file-static. Including this
// header from two .s files in the same package therefore gives each file its
// own private copy of the RODATA rather than a duplicate-symbol error. The
// cost is roughly 600 duplicated bytes; the benefit is one definition of the
// arithmetic.
//
// Three things in here are load-bearing and easy to get wrong:
//
//  1. VMINPS/VMAXPS return SRC2 when either input is NaN. SRC2 is the *first*
//     written Plan 9 operand, so the data register must come first. Writing
//     the constant first still passes every finite-grid accuracy test and
//     silently turns NaN into 88.72.
//
//  2. The 2^k reconstruction must be split into two half exponents. A single
//     (k+127)<<23 produces the Inf/NaN encoding at k = 128 and a sign-flipped
//     garbage pattern at k = -150.
//
//  3. Every instruction is VEX-encoded and every RET is preceded by
//     VZEROUPPER. A single legacy-SSE instruction in a VEX function costs an
//     AVX-SSE transition penalty on every iteration.

// # Register footprint
//
// EXPBODY deliberately keeps only TWO constants in registers. Everything else
// is read straight out of RODATA as a memory operand, which on this class of
// core is a fused load-op costing no extra throughput and, unlike a register,
// costs nothing to keep live. The point is that EXPBODY then occupies only
// Y0..Y4 plus Y6..Y7, leaving Y5 and Y8..Y15 to the caller. The fused
// tanh/log(cosh) kernel needs exactly that headroom: it computes four branch
// bodies around this one, and an EXPBODY that parked ten constants in
// registers would have had them overwritten between loop iterations.
//
// The clamps are the exception and MUST keep their constants in registers.
// VMINPS/VMAXPS return SRC2 when either input is NaN, and SRC2 is the same
// operand slot that a memory reference would have to occupy. Feeding the
// constant from memory would therefore make every NaN come back as 88.72,
// which no accuracy test over a finite grid can see.
DATA expHiF32<>(SB)/4, $0x42b17218 // 88.7228391
GLOBL expHiF32<>(SB), RODATA|NOPTR, $4

DATA expLoF32<>(SB)/4, $0xc2d00000 // -104.0
GLOBL expLoF32<>(SB), RODATA|NOPTR, $4

// Vector constants used as memory operands; VEX memory operands need no
// alignment, so these are ordinary 32-byte blobs.

// expLog2eV = 1/ln2
DATA expLog2eV<>+0x00(SB)/8, $0x3fb8aa3b3fb8aa3b
DATA expLog2eV<>+0x08(SB)/8, $0x3fb8aa3b3fb8aa3b
DATA expLog2eV<>+0x10(SB)/8, $0x3fb8aa3b3fb8aa3b
DATA expLog2eV<>+0x18(SB)/8, $0x3fb8aa3b3fb8aa3b
GLOBL expLog2eV<>(SB), RODATA|NOPTR, $32

// expC1V = 0.693359375, the 9-bit head of ln2
DATA expC1V<>+0x00(SB)/8, $0x3f3180003f318000
DATA expC1V<>+0x08(SB)/8, $0x3f3180003f318000
DATA expC1V<>+0x10(SB)/8, $0x3f3180003f318000
DATA expC1V<>+0x18(SB)/8, $0x3f3180003f318000
GLOBL expC1V<>(SB), RODATA|NOPTR, $32

// expC2V = -2.12194440e-4, the tail of ln2
DATA expC2V<>+0x00(SB)/8, $0xb95e8083b95e8083
DATA expC2V<>+0x08(SB)/8, $0xb95e8083b95e8083
DATA expC2V<>+0x10(SB)/8, $0xb95e8083b95e8083
DATA expC2V<>+0x18(SB)/8, $0xb95e8083b95e8083
GLOBL expC2V<>(SB), RODATA|NOPTR, $32

// expP4V .. expP0V are the degree-6 minimax coefficients.
DATA expP4V<>+0x00(SB)/8, $0x3ab512333ab51233
DATA expP4V<>+0x08(SB)/8, $0x3ab512333ab51233
DATA expP4V<>+0x10(SB)/8, $0x3ab512333ab51233
DATA expP4V<>+0x18(SB)/8, $0x3ab512333ab51233
GLOBL expP4V<>(SB), RODATA|NOPTR, $32

DATA expP3V<>+0x00(SB)/8, $0x3c091ceb3c091ceb
DATA expP3V<>+0x08(SB)/8, $0x3c091ceb3c091ceb
DATA expP3V<>+0x10(SB)/8, $0x3c091ceb3c091ceb
DATA expP3V<>+0x18(SB)/8, $0x3c091ceb3c091ceb
GLOBL expP3V<>(SB), RODATA|NOPTR, $32

DATA expP2V<>+0x00(SB)/8, $0x3d2aac793d2aac79
DATA expP2V<>+0x08(SB)/8, $0x3d2aac793d2aac79
DATA expP2V<>+0x10(SB)/8, $0x3d2aac793d2aac79
DATA expP2V<>+0x18(SB)/8, $0x3d2aac793d2aac79
GLOBL expP2V<>(SB), RODATA|NOPTR, $32

DATA expP1V<>+0x00(SB)/8, $0x3e2aaa493e2aaa49
DATA expP1V<>+0x08(SB)/8, $0x3e2aaa493e2aaa49
DATA expP1V<>+0x10(SB)/8, $0x3e2aaa493e2aaa49
DATA expP1V<>+0x18(SB)/8, $0x3e2aaa493e2aaa49
GLOBL expP1V<>(SB), RODATA|NOPTR, $32

DATA expP0V<>+0x00(SB)/8, $0x3efffffe3efffffe
DATA expP0V<>+0x08(SB)/8, $0x3efffffe3efffffe
DATA expP0V<>+0x10(SB)/8, $0x3efffffe3efffffe
DATA expP0V<>+0x18(SB)/8, $0x3efffffe3efffffe
GLOBL expP0V<>(SB), RODATA|NOPTR, $32

DATA expOneF32<>+0x00(SB)/8, $0x3f8000003f800000
DATA expOneF32<>+0x08(SB)/8, $0x3f8000003f800000
DATA expOneF32<>+0x10(SB)/8, $0x3f8000003f800000
DATA expOneF32<>+0x18(SB)/8, $0x3f8000003f800000
GLOBL expOneF32<>(SB), RODATA|NOPTR, $32

DATA expBiasI32<>+0x00(SB)/8, $0x0000007f0000007f
DATA expBiasI32<>+0x08(SB)/8, $0x0000007f0000007f
DATA expBiasI32<>+0x10(SB)/8, $0x0000007f0000007f
DATA expBiasI32<>+0x18(SB)/8, $0x0000007f0000007f
GLOBL expBiasI32<>(SB), RODATA|NOPTR, $32

// expMaskTab is nine 32-byte rows; row j has j leading all-ones lanes. Used
// with VMASKMOVPS for the 1..7 element tail. Masked-off lanes neither fault on
// load nor write on store, so reading and writing past the end of the slice is
// safe; those lanes just compute exp(0) into nothing.
DATA expMaskTab<>+0x000(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x008(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x010(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x018(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x020(SB)/8, $0x00000000ffffffff
DATA expMaskTab<>+0x028(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x030(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x038(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x040(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x048(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x050(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x058(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x060(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x068(SB)/8, $0x00000000ffffffff
DATA expMaskTab<>+0x070(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x078(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x080(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x088(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x090(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x098(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x0a0(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0a8(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0b0(SB)/8, $0x00000000ffffffff
DATA expMaskTab<>+0x0b8(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x0c0(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0c8(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0d0(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0d8(SB)/8, $0x0000000000000000
DATA expMaskTab<>+0x0e0(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0e8(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0f0(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x0f8(SB)/8, $0x00000000ffffffff
DATA expMaskTab<>+0x100(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x108(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x110(SB)/8, $0xffffffffffffffff
DATA expMaskTab<>+0x118(SB)/8, $0xffffffffffffffff
GLOBL expMaskTab<>(SB), RODATA|NOPTR, $288

// EXPLOADCONSTS puts the two clamp constants into Y6 and Y7. Callers must do
// this once, outside the loop; see the register-footprint note above for why
// these two and only these two live in registers.
#define EXPLOADCONSTS \
	VBROADCASTSS expHiF32<>(SB), Y6 \
	VBROADCASTSS expLoF32<>(SB), Y7

// EXPBODY turns the eight lanes in Y0 into exp(Y0) in Y3.
//
// Clobbers Y0, Y1, Y2, Y3, Y4. Reads the clamp constants in Y6 and Y7. Y5 and
// Y8..Y15 are untouched, which is what lets the fused kernel keep the original
// x live across the call and still have a working set of its own.
#define EXPBODY \
	VMINPS       Y0, Y6, Y0            \ // clamp high; data operand FIRST for NaN
	VMAXPS       Y0, Y7, Y0            \ // clamp low;  data operand FIRST for NaN
	VMULPS       expLog2eV<>(SB), Y0, Y1 \ // x * log2e
	VROUNDPS     $0x08, Y1, Y1         \ // fx = rint(x*log2e), nearest-even
	VFNMADD231PS expC1V<>(SB), Y1, Y0  \ // r  = x - fx*C1   (exact)
	VFNMADD231PS expC2V<>(SB), Y1, Y0  \ // r -= fx*C2
	VMULPS       Y0, Y0, Y2            \ // z = r*r
	VMOVUPS      expP4V<>(SB), Y3      \ // P4
	VFMADD213PS  expP3V<>(SB), Y0, Y3  \ // *r + P3
	VFMADD213PS  expP2V<>(SB), Y0, Y3  \ // *r + P2
	VFMADD213PS  expP1V<>(SB), Y0, Y3  \ // *r + P1
	VFMADD213PS  expP0V<>(SB), Y0, Y3  \ // *r + P0 = P(r)
	VFMADD213PS  Y0,  Y2, Y3           \ // P*z + r
	VADDPS       expOneF32<>(SB), Y3, Y3 \ // + 1 => exp(r)
	VCVTPS2DQ    Y1, Y1                \ // k = int32(fx)
	VPSRAD       $1, Y1, Y4            \ // k1 = k >> 1
	VPSUBD       Y4, Y1, Y1            \ // k2 = k - k1
	VPADDD       expBiasI32<>(SB), Y4, Y4 \
	VPADDD       expBiasI32<>(SB), Y1, Y1 \
	VPSLLD       $23, Y4, Y4           \ // 2^k1
	VPSLLD       $23, Y1, Y1           \ // 2^k2
	VMULPS       Y4, Y3, Y3            \
	VMULPS       Y1, Y3, Y3
