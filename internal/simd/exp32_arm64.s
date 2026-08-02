// NEON float32 batch exp.
//
// This is a transliteration of expBatch32Go (see exp32.go) into four-wide
// vector form, and the direct counterpart of expBatch32AVX2. Every constant,
// the Cody-Waite split, the degree-6 minimax polynomial and the two-step 2^k
// reconstruction are the same; only the width and the use of hardware FMA
// differ.
//
// The arithmetic lives in exp32_arm64.h, which the fused tanh/log(cosh) kernel
// includes as well. The hand-encoded instructions it uses live in neon_arm64.h,
// which explains why they have to be hand-encoded at all.
//
// # The tail
//
// The AVX2 kernel handles its 1..7 element tail with VMASKMOVPS, which neither
// faults on the masked-off loads nor writes on the masked-off stores. NEON has
// no masked load or store, so the 1..3 element tail is staged through a
// 16-byte scratch buffer in the frame instead.
//
// The obvious cheaper trick — process a final full vector starting at n-4,
// recomputing up to three elements — is wrong here and is worth naming so it
// does not get "optimised" back in. dst == src is a supported mode, and the
// overlapping lanes have already been written by the previous iteration, so
// that variant would read its own output and return exp(exp(x)) for up to
// three elements at the end of every in-place call. No test over disjoint
// slices can see it.
//
// Staging through the buffer keeps every element on exactly the same
// instruction path, which is the property the differential tests rely on: a
// tail computed by some other route could differ from the body by an ulp for
// reasons that have nothing to do with the kernel.

//go:build arm64 && !purego

#include "textflag.h"
#include "neon_arm64.h"
#include "exp32_arm64.h"

// func expBatch32NEON(dst, src []float32) bool
TEXT ·expBatch32NEON(SB), NOSPLIT, $16-49
	MOVD $1, R9
	MOVB R9, ret+48(FP) // this kernel never declines

	MOVD dst_base+0(FP), R0
	MOVD dst_len+8(FP), R1
	MOVD src_base+24(FP), R2
	MOVD src_len+32(FP), R3

	// n = min(len(dst), len(src))
	CMP  R3, R1
	CSEL GT, R3, R1, R1
	CBZ  R1, done

	EXPLOADCONSTS

	AND $3, R1, R5 // R5 = tail element count, 0..3
	LSR $2, R1, R4
	LSL $2, R4, R4 // R4 = whole-vector element count
	CBZ R4, tail

loop:
	VLD1.P 16(R2), [V0.S4]
	EXPBODY
	VST1.P [V3.S4], 16(R0)

	SUB  $4, R4, R4
	CBNZ R4, loop

tail:
	CBZ R5, done

	// Zero the scratch buffer so the unused lanes hold exp(0) rather than
	// whatever was on the stack. Nothing reads those lanes, but a stack-
	// dependent computation is a stack-dependent bug waiting to happen.
	MOVD $buf-16(SP), R8
	VEOR V0.B16, V0.B16, V0.B16
	VST1 [V0.S4], (R8)

	MOVD R8, R9
	MOVD R5, R6

copyin:
	MOVWU.P 4(R2), R7
	MOVW.P  R7, 4(R9)
	SUB     $1, R6, R6
	CBNZ    R6, copyin

	VLD1 (R8), [V0.S4]
	EXPBODY
	VST1 [V3.S4], (R8)

	MOVD R8, R9

copyout:
	MOVWU.P 4(R9), R7
	MOVW.P  R7, 4(R0)
	SUB     $1, R5, R5
	CBNZ    R5, copyout

done:
	RET
