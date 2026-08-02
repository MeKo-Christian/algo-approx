// AVX2+FMA float32 batch exp.
//
// This is a transliteration of expBatch32Go (see exp32.go) into eight-wide
// vector form. Every constant, the Cody-Waite split, the degree-6 minimax
// polynomial and the two-step 2^k reconstruction are the same; only the width
// and the use of hardware FMA differ. The FMA contractions are the sole reason
// the two kernels are not bit-identical, and they cost at most 1 ulp.
//
// The arithmetic itself, and the three traps that come with it, live in
// exp32_amd64.h, which the fused tanh/log(cosh) kernel includes as well.

//go:build amd64 && !purego

#include "textflag.h"
#include "exp32_amd64.h"

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
	EXPLOADCONSTS

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
