// NEON fused float32 batch tanh / log(cosh).
//
// A transliteration of tanhLogCoshBatch32Go (see hyperbolic32.go) into
// four-wide vector form, and the direct counterpart of tanhLogCoshBatch32AVX2:
// same branch point, same rational core, same series, same shared
// u = exp(-2|x|). The exponential comes from EXPBODY in exp32_arm64.h, so this
// kernel and expBatch32NEON cannot drift apart.
//
// # Why this kernel is the interesting one
//
// expBatch32NEON is branch-free, so vectorising it is mechanical. This one is
// not: the scalar code has a data-dependent branch at |x| = 0.625, and a vector
// kernel cannot take a branch per lane. It therefore evaluates BOTH sides for
// every lane and blends. That is real, unavoidable extra work — roughly
// nineteen instructions per element that a scalar version would skip for
// whichever branch it did not take — and whether the four-fold width still pays
// after absorbing it is the question this file answers.
//
// # Constants live in memory, not in registers
//
// The AVX2 version reads most of its constants as memory operands, which x86
// fuses into the arithmetic instruction at no extra cost. NEON has no memory
// operands at all, so every constant must reach a register first. There are
// twenty-six of them here and, with EXPBODY holding twelve in V16..V27, there
// are nowhere near twenty-six free registers.
//
// The way out is that Horner evaluation wants a fresh register for each
// coefficient anyway: VFMLA accumulates into its destination, so every step
// needs its coefficient loaded somewhere before it can be used. Loading that
// coefficient from the table with FMOVQ costs exactly what copying it out of a
// register would have cost — one instruction — so parking the polynomial
// coefficients in registers would buy nothing even if there were room. Only the
// four constants used more than once, or used outside a Horner step, are worth
// a register: they are in V28..V31.
//
// R10 holds the table base for the whole loop. FMOVQ's immediate offset covers
// 64 KiB, far more than the 416 bytes here, so no address arithmetic is needed.
//
// # Register budget
//
//	V16..V27  EXPBODY's constants; live for the whole loop
//	V28..V31  absMask, signMask, 1.0, 2.0; live for the whole loop
//	V5        x, the only value carried across EXPBODY
//	V0..V4    EXPBODY's scratch, then a / sign / mask / u / z
//	V8..V13   the four branch bodies; V8 ends as tanh, V11 as log(cosh)
//	V6, V7, V14, V15  free
//
// a, sign, z and the branch mask are rebuilt from V5 after EXPBODY rather than
// kept across it, because EXPBODY owns V0..V4. That is five instructions per
// block of four and is cheaper than spilling to the stack.
//
// # Divisions
//
// There are three FDIVs per block: the rational core's P/Q, tanh's 2u/(1+u),
// and log1p's w = u/(2+u). The same VRCPPS-plus-Newton question was asked of
// the AVX2 kernel and answered by measurement: the divider was busy 42% of
// cycles but removing it entirely recovered only 6%, because the out-of-order
// engine already hides most of it behind the surrounding FMA work. See PLAN.md
// §6.2.2. That measurement was taken on Cascade Lake and says nothing directly
// about this core — FRECPE plus Newton steps is a different trade here — but it
// does say that divider *occupancy* is not the same as critical path, and that
// the thing to do first is measure rather than write the reciprocal sequence.

//go:build arm64 && !purego

#include "textflag.h"
#include "neon_arm64.h"
#include "exp32_arm64.h"

// Offsets into hypConstsNEON. Each entry is one float32 (or int32) replicated
// across four lanes, so every offset is a multiple of 16.
#define HYP_ABSMASK    0x000
#define HYP_SIGNMASK   0x010
#define HYP_BRANCHBITS 0x020
#define HYP_NEGTWO     0x030
#define HYP_ONE        0x040
#define HYP_TWO        0x050
#define HYP_TP0        0x060
#define HYP_TP1        0x070
#define HYP_TP2        0x080
#define HYP_TQ0        0x090
#define HYP_TQ1        0x0a0
#define HYP_TQ2        0x0b0
#define HYP_LC0        0x0c0
#define HYP_LC1        0x0d0
#define HYP_LC2        0x0e0
#define HYP_LC3        0x0f0
#define HYP_LC4        0x100
#define HYP_LC5        0x110
#define HYP_LC6        0x120
#define HYP_LC7        0x130
#define HYP_LC8        0x140
#define HYP_LN2        0x150
#define HYP_R3         0x160
#define HYP_R5         0x170
#define HYP_R7         0x180
#define HYP_R9         0x190

DATA hypConstsNEON<>+0x000(SB)/8, $0x7fffffff7fffffff // |x| mask
DATA hypConstsNEON<>+0x008(SB)/8, $0x7fffffff7fffffff
DATA hypConstsNEON<>+0x010(SB)/8, $0x8000000080000000 // sign mask
DATA hypConstsNEON<>+0x018(SB)/8, $0x8000000080000000
DATA hypConstsNEON<>+0x020(SB)/8, $0x3f2000003f200000 // bits(0.625), compared as int32
DATA hypConstsNEON<>+0x028(SB)/8, $0x3f2000003f200000
DATA hypConstsNEON<>+0x030(SB)/8, $0xc0000000c0000000 // -2, forms the exp argument -2|x|
DATA hypConstsNEON<>+0x038(SB)/8, $0xc0000000c0000000
DATA hypConstsNEON<>+0x040(SB)/8, $0x3f8000003f800000 // 1
DATA hypConstsNEON<>+0x048(SB)/8, $0x3f8000003f800000
DATA hypConstsNEON<>+0x050(SB)/8, $0x4000000040000000 // 2
DATA hypConstsNEON<>+0x058(SB)/8, $0x4000000040000000
DATA hypConstsNEON<>+0x060(SB)/8, $0xbf76e2ddbf76e2dd // tanh rational, numerator
DATA hypConstsNEON<>+0x068(SB)/8, $0xbf76e2ddbf76e2dd //   -0.9643992
DATA hypConstsNEON<>+0x070(SB)/8, $0xc2c69350c2c69350 //  -99.28772
DATA hypConstsNEON<>+0x078(SB)/8, $0xc2c69350c2c69350
DATA hypConstsNEON<>+0x080(SB)/8, $0xc4c9d602c4c9d602 // -1614.6877
DATA hypConstsNEON<>+0x088(SB)/8, $0xc4c9d602c4c9d602
DATA hypConstsNEON<>+0x090(SB)/8, $0x42e19f9442e19f94 // tanh rational, denominator
DATA hypConstsNEON<>+0x098(SB)/8, $0x42e19f9442e19f94 //  112.81168
DATA hypConstsNEON<>+0x0a0(SB)/8, $0x450bb7d0450bb7d0 // 2235.4883
DATA hypConstsNEON<>+0x0a8(SB)/8, $0x450bb7d0450bb7d0
DATA hypConstsNEON<>+0x0b0(SB)/8, $0x4597608145976081 // 4844.063
DATA hypConstsNEON<>+0x0b8(SB)/8, $0x4597608145976081
DATA hypConstsNEON<>+0x0c0(SB)/8, $0x3f0000003f000000 // log(cosh) series in z = x*x
DATA hypConstsNEON<>+0x0c8(SB)/8, $0x3f0000003f000000 //  0.5
DATA hypConstsNEON<>+0x0d0(SB)/8, $0xbdaaaaabbdaaaaab // -0.08333334
DATA hypConstsNEON<>+0x0d8(SB)/8, $0xbdaaaaabbdaaaaab
DATA hypConstsNEON<>+0x0e0(SB)/8, $0x3cb60b613cb60b61 //  0.022222223
DATA hypConstsNEON<>+0x0e8(SB)/8, $0x3cb60b613cb60b61
DATA hypConstsNEON<>+0x0f0(SB)/8, $0xbbdd0dd1bbdd0dd1 // -0.006746032
DATA hypConstsNEON<>+0x0f8(SB)/8, $0xbbdd0dd1bbdd0dd1
DATA hypConstsNEON<>+0x100(SB)/8, $0x3b0f52ea3b0f52ea //  0.0021869488
DATA hypConstsNEON<>+0x108(SB)/8, $0x3b0f52ea3b0f52ea
DATA hypConstsNEON<>+0x110(SB)/8, $0xba419eceba419ece // -0.00073855303
DATA hypConstsNEON<>+0x118(SB)/8, $0xba419eceba419ece
DATA hypConstsNEON<>+0x120(SB)/8, $0x398685a9398685a9 //  0.00025660123
DATA hypConstsNEON<>+0x128(SB)/8, $0x398685a9398685a9
DATA hypConstsNEON<>+0x130(SB)/8, $0xb8bed1b2b8bed1b2 // -9.098576e-05
DATA hypConstsNEON<>+0x138(SB)/8, $0xb8bed1b2b8bed1b2
DATA hypConstsNEON<>+0x140(SB)/8, $0x38097c8238097c82 //  3.278177e-05
DATA hypConstsNEON<>+0x148(SB)/8, $0x38097c8238097c82
DATA hypConstsNEON<>+0x150(SB)/8, $0x3f3172183f317218 // ln2 rounded to float32
DATA hypConstsNEON<>+0x158(SB)/8, $0x3f3172183f317218
DATA hypConstsNEON<>+0x160(SB)/8, $0x3eaaaaab3eaaaaab // atanh series for log1p
DATA hypConstsNEON<>+0x168(SB)/8, $0x3eaaaaab3eaaaaab //  0.33333334
DATA hypConstsNEON<>+0x170(SB)/8, $0x3e4ccccd3e4ccccd //  0.2
DATA hypConstsNEON<>+0x178(SB)/8, $0x3e4ccccd3e4ccccd
DATA hypConstsNEON<>+0x180(SB)/8, $0x3e1249253e124925 //  0.14285715
DATA hypConstsNEON<>+0x188(SB)/8, $0x3e1249253e124925
DATA hypConstsNEON<>+0x190(SB)/8, $0x3de38e393de38e39 //  0.11111111
DATA hypConstsNEON<>+0x198(SB)/8, $0x3de38e393de38e39
GLOBL hypConstsNEON<>(SB), RODATA|NOPTR, $416

// HYPLOADCONSTS fills V28..V31 with the four constants that are not Horner
// coefficients, and points R10 at the table. Callers must do this once, outside
// the loop, and must call EXPLOADCONSTS as well.
#define HYPLOADCONSTS                     \
	MOVD  $hypConstsNEON<>(SB), R10   \
	FMOVQ HYP_ABSMASK(R10), F28       \
	FMOVQ HYP_SIGNMASK(R10), F29      \
	FMOVQ HYP_ONE(R10), F30           \
	FMOVQ HYP_TWO(R10), F31

// HYPPRE turns the four raw inputs in V5 into the exp argument -2|x| in V0,
// ready for EXPBODY. Clobbers V0 and V1.
#define HYPPRE                            \
	VAND  V28.B16, V5.B16, V0.B16     \
	FMOVQ HYP_NEGTWO(R10), F1         \
	NFMUL(1, 0, 0)

// HYPBODY consumes x in V5 and u = exp(-2|x|) in V3, and leaves tanh(x) in V8
// and log(cosh(x)) in V11. Clobbers V0..V2, V4, V8..V13.
//
// V16..V27 are NOT touched: they hold EXPBODY's constants and must survive to
// the next iteration of the loop. V6, V7, V14 and V15 are left free.
//
// # One place where this kernel must NOT copy the AVX2 one
//
// logCoshLarge32 ends with `a - ln2f + 2*w*(...)`, and the last step here is a
// VFMLA that accumulates 2w*series into a - ln2 with a single rounding. The
// AVX2 kernel deliberately does the opposite — a separate VMULPS then VADDPS —
// and both are faithful to their own compiler: amd64 never contracts a
// multiply-add, so the Go kernel it is checked against rounds the product
// first, while arm64 always contracts, so the Go kernel here does not.
//
// Getting this backwards is not a hypothetical. The first version of this file
// mirrored the AVX2 sequence and disagreed with the Go kernel by 1 ulp of
// log(cosh) at x = 0.81 — one element in twenty-five, caught by
// TestBlockAndTailAgree, and invisible in the tanh output. A differential test
// only means something when both sides are computing the same expression; this
// is where "transliterate the AVX2 kernel" stops being the right instruction.
#define HYPBODY                                                                 \
	VAND  V28.B16, V5.B16, V0.B16      /* a = |x|                          */ \
	VAND  V29.B16, V5.B16, V1.B16      /* the sign bit, reattached below   */ \
	FMOVQ HYP_BRANCHBITS(R10), F2                                             \
	VSUB  V2.S4, V0.S4, V2.S4          /* integer compare on bit patterns: */ \
	NSSHR(31, 2, 2)                    /* all-ones iff a < 0.625, NaN high */ \
	NFMUL(0, 0, 4)                     /* z = a*a                          */ \
	                                                                          \
	FMOVQ HYP_TP0(R10), F9             /* -- tanh, a < 0.625: a + a*z*P/Q  */ \
	FMOVQ HYP_TP1(R10), F10                                                   \
	VFMLA V4.S4, V9.S4, V10.S4                                                \
	FMOVQ HYP_TP2(R10), F9                                                    \
	VFMLA V4.S4, V10.S4, V9.S4         /* V9 = P(z)                        */ \
	FMOVQ HYP_TQ0(R10), F10                                                   \
	NFADD(4, 10, 10)                                                          \
	FMOVQ HYP_TQ1(R10), F11                                                   \
	VFMLA V4.S4, V10.S4, V11.S4                                               \
	FMOVQ HYP_TQ2(R10), F10                                                   \
	VFMLA V4.S4, V11.S4, V10.S4        /* V10 = Q(z)                       */ \
	NFDIV(10, 9, 9)                    /* V9 = P/Q                         */ \
	NFMUL(4, 0, 10)                    /* a*z                              */ \
	VMOV  V0.B16, V11.B16              /* a                                */ \
	VFMLA V10.S4, V9.S4, V11.S4        /* V11 = a + a*z*(P/Q)              */ \
	                                                                          \
	NFADD(30, 3, 12)                   /* -- tanh, a >= 0.625: 1-2u/(1+u)  */ \
	NFADD(3, 3, 13)                    /* 2u                               */ \
	NFDIV(12, 13, 13)                  /* 2u/(1+u)                         */ \
	NFSUB(13, 30, 8)                   /* V8 = 1 - 2u/(1+u)                */ \
	                                                                          \
	VBIT  V2.B16, V11.B16, V8.B16      /* mask set -> rational core        */ \
	VEOR  V1.B16, V8.B16, V8.B16       /* sign: odd symmetry is BIT-exact  */ \
	                                                                          \
	FMOVQ HYP_LC8(R10), F9             /* -- log(cosh), a < 0.625: z*S(z)  */ \
	FMOVQ HYP_LC7(R10), F10                                                   \
	VFMLA V4.S4, V9.S4, V10.S4                                                \
	FMOVQ HYP_LC6(R10), F9                                                    \
	VFMLA V4.S4, V10.S4, V9.S4                                                \
	FMOVQ HYP_LC5(R10), F10                                                   \
	VFMLA V4.S4, V9.S4, V10.S4                                                \
	FMOVQ HYP_LC4(R10), F9                                                    \
	VFMLA V4.S4, V10.S4, V9.S4                                                \
	FMOVQ HYP_LC3(R10), F10                                                   \
	VFMLA V4.S4, V9.S4, V10.S4                                                \
	FMOVQ HYP_LC2(R10), F9                                                    \
	VFMLA V4.S4, V10.S4, V9.S4                                                \
	FMOVQ HYP_LC1(R10), F10                                                   \
	VFMLA V4.S4, V9.S4, V10.S4                                                \
	FMOVQ HYP_LC0(R10), F9                                                    \
	VFMLA V4.S4, V10.S4, V9.S4         /* V9 = S(z)                        */ \
	NFMUL(4, 9, 9)                     /* V9 = z*S(z)                      */ \
	                                                                          \
	NFADD(31, 3, 10)                   /* -- log(cosh), a >= 0.625:        */ \
	NFDIV(10, 3, 10)                   /* w = u/(2+u), log1p = 2*atanh(w)  */ \
	NFMUL(10, 10, 11)                  /* w2                               */ \
	FMOVQ HYP_R9(R10), F12                                                    \
	FMOVQ HYP_R7(R10), F13                                                    \
	VFMLA V11.S4, V12.S4, V13.S4                                              \
	FMOVQ HYP_R5(R10), F12                                                    \
	VFMLA V11.S4, V13.S4, V12.S4                                              \
	FMOVQ HYP_R3(R10), F13                                                    \
	VFMLA V11.S4, V12.S4, V13.S4                                              \
	VMOV  V30.B16, V12.B16             /* 1                                */ \
	VFMLA V11.S4, V13.S4, V12.S4       /* V12 = atanh series               */ \
	NFADD(10, 10, 13)                  /* V13 = 2w                         */ \
	FMOVQ HYP_LN2(R10), F11                                                   \
	NFSUB(11, 0, 11)                   /* V11 = a - ln2                    */ \
	VFMLA V12.S4, V13.S4, V11.S4       /* V11 = (a-ln2) + 2w*series        */ \
	                                                                          \
	VBIT  V2.B16, V9.B16, V11.B16      /* mask set -> series               */

// func tanhLogCoshBatch32NEON(dstTanh, dstLogCosh, src []float32) bool
TEXT ·tanhLogCoshBatch32NEON(SB), NOSPLIT, $16-73
	MOVD $1, R9
	MOVB R9, ret+72(FP) // this kernel never declines

	MOVD dstTanh_base+0(FP), R0
	MOVD dstTanh_len+8(FP), R3
	MOVD dstLogCosh_base+24(FP), R1
	MOVD dstLogCosh_len+32(FP), R6
	MOVD src_base+48(FP), R2
	MOVD src_len+56(FP), R7

	// R3 = n = min(len(dstTanh), len(dstLogCosh), len(src))
	CMP  R6, R3
	CSEL GT, R6, R3, R3
	CMP  R7, R3
	CSEL GT, R7, R3, R3
	CBZ  R3, done

	EXPLOADCONSTS
	HYPLOADCONSTS

	AND $3, R3, R5 // R5 = tail element count, 0..3
	LSR $2, R3, R4
	LSL $2, R4, R4 // R4 = whole-vector element count
	CBZ R4, tail

loop:
	VLD1.P 16(R2), [V5.S4]
	HYPPRE
	EXPBODY
	HYPBODY
	VST1.P [V8.S4], 16(R0)
	VST1.P [V11.S4], 16(R1)

	SUB  $4, R4, R4
	CBNZ R4, loop

tail:
	CBZ R5, done

	// The 1..3 element tail is staged through a 16-byte scratch buffer in the
	// frame, for the reasons set out at the top of exp32_arm64.s: NEON has no
	// masked store, and the cheaper-looking overlapping-vector trick silently
	// breaks the in-place case. The buffer is zeroed first, so the unused lanes
	// carry +0 — which takes the rational branch and yields 0 in both outputs —
	// rather than whatever was on the stack.
	MOVD $buf-16(SP), R12
	VEOR V5.B16, V5.B16, V5.B16
	VST1 [V5.S4], (R12)

	MOVD R12, R9
	MOVD R5, R6

copyin:
	MOVWU.P 4(R2), R7
	MOVW.P  R7, 4(R9)
	SUB     $1, R6, R6
	CBNZ    R6, copyin

	VLD1 (R12), [V5.S4]
	HYPPRE
	EXPBODY
	HYPBODY

	// Both destinations are copied out of the same buffer, one after the other,
	// so V8 must survive the log(cosh) copy loop. Nothing between here and the
	// second VST1 touches it.
	VST1 [V8.S4], (R12)
	MOVD R12, R9
	MOVD R5, R6

copyoutTanh:
	MOVWU.P 4(R9), R7
	MOVW.P  R7, 4(R0)
	SUB     $1, R6, R6
	CBNZ    R6, copyoutTanh

	VST1 [V11.S4], (R12)
	MOVD R12, R9

copyoutLogCosh:
	MOVWU.P 4(R9), R7
	MOVW.P  R7, 4(R1)
	SUB     $1, R5, R5
	CBNZ    R5, copyoutLogCosh

done:
	RET
