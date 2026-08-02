// AVX2+FMA float32 batch exp.
//
// This is a transliteration of expBatch32Go (see exp32.go) into eight-wide
// vector form. Every constant, the Cody-Waite split, the degree-6 minimax
// polynomial and the two-step 2^k reconstruction are the same; only the width
// and the use of hardware FMA differ. The FMA contractions are the sole reason
// the two kernels are not bit-identical, and they cost at most 1 ulp.
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

//go:build amd64 && !purego

#include "textflag.h"

// Scalar constants, broadcast into a register once before the loop.
DATA expLog2eF32<>(SB)/4, $0x3fb8aa3b // 1/ln2
GLOBL expLog2eF32<>(SB), RODATA|NOPTR, $4

DATA expC1F32<>(SB)/4, $0x3f318000 // 0.693359375, the 9-bit head of ln2
GLOBL expC1F32<>(SB), RODATA|NOPTR, $4

DATA expC2F32<>(SB)/4, $0xb95e8083 // -2.12194440e-4, the tail of ln2
GLOBL expC2F32<>(SB), RODATA|NOPTR, $4

DATA expP4F32<>(SB)/4, $0x3ab51233
GLOBL expP4F32<>(SB), RODATA|NOPTR, $4

DATA expP3F32<>(SB)/4, $0x3c091ceb
GLOBL expP3F32<>(SB), RODATA|NOPTR, $4

DATA expP2F32<>(SB)/4, $0x3d2aac79
GLOBL expP2F32<>(SB), RODATA|NOPTR, $4

DATA expP1F32<>(SB)/4, $0x3e2aaa49
GLOBL expP1F32<>(SB), RODATA|NOPTR, $4

DATA expP0F32<>(SB)/4, $0x3efffffe
GLOBL expP0F32<>(SB), RODATA|NOPTR, $4

DATA expHiF32<>(SB)/4, $0x42b17218 // 88.7228391
GLOBL expHiF32<>(SB), RODATA|NOPTR, $4

DATA expLoF32<>(SB)/4, $0xc2d00000 // -104.0
GLOBL expLoF32<>(SB), RODATA|NOPTR, $4

// Vector constants used as memory operands; VEX memory operands need no
// alignment, so these are ordinary 32-byte blobs.
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

// EXPBODY turns the eight lanes in Y0 into exp(Y0) in Y3.
//
// Clobbers Y0, Y1, Y2, Y3, Y4. Reads the constants held in Y6..Y15.
#define EXPBODY \
	VMINPS       Y0, Y6, Y0            \ // clamp high; data operand FIRST for NaN
	VMAXPS       Y0, Y7, Y0            \ // clamp low;  data operand FIRST for NaN
	VMULPS       Y8, Y0, Y1            \ // x * log2e
	VROUNDPS     $0x08, Y1, Y1         \ // fx = rint(x*log2e), nearest-even
	VFNMADD231PS Y9,  Y1, Y0           \ // r  = x - fx*C1   (exact)
	VFNMADD231PS Y10, Y1, Y0           \ // r -= fx*C2
	VMULPS       Y0, Y0, Y2            \ // z = r*r
	VMOVAPS      Y11, Y3               \ // P4
	VFMADD213PS  Y12, Y0, Y3           \ // *r + P3
	VFMADD213PS  Y13, Y0, Y3           \ // *r + P2
	VFMADD213PS  Y14, Y0, Y3           \ // *r + P1
	VFMADD213PS  Y15, Y0, Y3           \ // *r + P0 = P(r)
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

// func expBatch32AVX2(dst, src []float32) bool
TEXT ·expBatch32AVX2(SB), NOSPLIT, $0-49
	MOVB $1, ret+48(FP) // this kernel never declines

	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), AX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), BX

	// n = min(len(dst), len(src))
	CMPQ AX, BX
	JGE  lenok
	MOVQ AX, BX

lenok:
	TESTQ BX, BX
	JEQ   empty

	// Broadcast the scalar constants once.
	VBROADCASTSS expHiF32<>(SB), Y6
	VBROADCASTSS expLoF32<>(SB), Y7
	VBROADCASTSS expLog2eF32<>(SB), Y8
	VBROADCASTSS expC1F32<>(SB), Y9
	VBROADCASTSS expC2F32<>(SB), Y10
	VBROADCASTSS expP4F32<>(SB), Y11
	VBROADCASTSS expP3F32<>(SB), Y12
	VBROADCASTSS expP2F32<>(SB), Y13
	VBROADCASTSS expP1F32<>(SB), Y14
	VBROADCASTSS expP0F32<>(SB), Y15

	// CX = byte length of the whole-vector part, R9 = running byte offset.
	MOVQ BX, CX
	SHRQ $3, CX
	SHLQ $5, CX
	XORQ R9, R9

	TESTQ CX, CX
	JEQ   tail

loop:
	VMOVUPS (SI)(R9*1), Y0
	EXPBODY
	VMOVUPS Y3, (DI)(R9*1)

	ADDQ $32, R9
	CMPQ R9, CX
	JLT  loop

tail:
	MOVQ  BX, DX
	ANDQ  $7, DX
	TESTQ DX, DX
	JEQ   done

	// Y5 = row DX of the mask table: DX leading all-ones lanes.
	LEAQ expMaskTab<>(SB), R11
	SHLQ $5, DX
	VMOVUPS (R11)(DX*1), Y5

	VMASKMOVPS (SI)(R9*1), Y5, Y0
	EXPBODY
	VMASKMOVPS Y3, Y5, (DI)(R9*1)

done:
	VZEROUPPER
	RET

empty:
	VZEROUPPER
	RET
