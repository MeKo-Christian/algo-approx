// AVX2+FMA fused float32 batch tanh / log(cosh).
//
// A transliteration of tanhLogCoshBatch32Go (see hyperbolic32.go) into
// eight-wide vector form: same branch point, same rational core, same series,
// same shared u = exp(-2|x|). The exponential comes from EXPBODY in
// exp32_amd64.h, so this kernel and expBatch32AVX2 cannot drift apart.
//
// # Why this kernel is the interesting one
//
// expBatch32AVX2 is branch-free, so vectorising it is mechanical. This one is
// not: the scalar code has a data-dependent branch at |x| = 0.625, and a
// vector kernel cannot take a branch per lane. It therefore evaluates BOTH
// sides for every lane and blends. That is real, unavoidable extra work -
// roughly nineteen instructions per element that a scalar version would skip
// for whichever branch it did not take - and whether the eight-fold width
// still pays after absorbing it is the entire question this file answers.
//
// # Register budget
//
// This kernel is the reason EXPBODY keeps all but two of its constants in
// RODATA instead of in registers. An earlier version broadcast ten of them
// into Y6..Y15; the four branch bodies below then overwrote those registers,
// so every block after the first computed its exponential from whatever the
// previous block had left behind. It produced correct results for the first
// eight elements and NaN for the rest, which is exactly the shape of bug that
// a spot check at a handful of x values will not find.
//
// The layout now is:
//
//	Y6, Y7   EXPBODY's clamp constants; live for the whole loop
//	Y5       x, the only value carried across EXPBODY
//	Y0..Y4   EXPBODY's scratch, then a / sign / mask / u / z
//	Y8..Y13  the four branch bodies; Y8 ends as tanh, Y9 as log(cosh)
//	Y15      the tail mask, on the tail path only
//
// a, sign, z and the branch mask are rebuilt from Y5 after EXPBODY rather than
// kept across it, because EXPBODY owns Y0..Y4. That is four instructions per
// block of eight and is far cheaper than spilling to the stack.
//
// # Divisions
//
// There are three VDIVPS per block: the rational core's num/den, tanh's
// 2u/(1+u), and log1p's w = u/(2+u). VDIVPS does not pipeline like the rest of
// the kernel, so it is the first thing to look at if the measured ratio
// disappoints. A VRCPPS plus two Newton-Raphson steps was probed on this
// hardware and lands within 1 ulp, i.e. it is a viable swap - but it is five
// instructions against one, so it is only worth it if the divider really is
// the limiter. Measure before switching.

//go:build amd64 && !purego

#include "textflag.h"
#include "exp32_amd64.h"

// hypAbsMask = |x| mask
DATA hypAbsMask<>+0x00(SB)/8, $0x7fffffff7fffffff
DATA hypAbsMask<>+0x08(SB)/8, $0x7fffffff7fffffff
DATA hypAbsMask<>+0x10(SB)/8, $0x7fffffff7fffffff
DATA hypAbsMask<>+0x18(SB)/8, $0x7fffffff7fffffff
GLOBL hypAbsMask<>(SB), RODATA|NOPTR, $32

// hypSignMask = sign mask
DATA hypSignMask<>+0x00(SB)/8, $0x8000000080000000
DATA hypSignMask<>+0x08(SB)/8, $0x8000000080000000
DATA hypSignMask<>+0x10(SB)/8, $0x8000000080000000
DATA hypSignMask<>+0x18(SB)/8, $0x8000000080000000
GLOBL hypSignMask<>(SB), RODATA|NOPTR, $32

// hypBranchBits = bits(0.625), compared as int32
DATA hypBranchBits<>+0x00(SB)/8, $0x3f2000003f200000
DATA hypBranchBits<>+0x08(SB)/8, $0x3f2000003f200000
DATA hypBranchBits<>+0x10(SB)/8, $0x3f2000003f200000
DATA hypBranchBits<>+0x18(SB)/8, $0x3f2000003f200000
GLOBL hypBranchBits<>(SB), RODATA|NOPTR, $32

// hypNegTwo = -2, forms the exp argument -2|x|
DATA hypNegTwo<>+0x00(SB)/8, $0xc0000000c0000000
DATA hypNegTwo<>+0x08(SB)/8, $0xc0000000c0000000
DATA hypNegTwo<>+0x10(SB)/8, $0xc0000000c0000000
DATA hypNegTwo<>+0x18(SB)/8, $0xc0000000c0000000
GLOBL hypNegTwo<>(SB), RODATA|NOPTR, $32

// hypOne = 1
DATA hypOne<>+0x00(SB)/8, $0x3f8000003f800000
DATA hypOne<>+0x08(SB)/8, $0x3f8000003f800000
DATA hypOne<>+0x10(SB)/8, $0x3f8000003f800000
DATA hypOne<>+0x18(SB)/8, $0x3f8000003f800000
GLOBL hypOne<>(SB), RODATA|NOPTR, $32

// hypTwo = 2
DATA hypTwo<>+0x00(SB)/8, $0x4000000040000000
DATA hypTwo<>+0x08(SB)/8, $0x4000000040000000
DATA hypTwo<>+0x10(SB)/8, $0x4000000040000000
DATA hypTwo<>+0x18(SB)/8, $0x4000000040000000
GLOBL hypTwo<>(SB), RODATA|NOPTR, $32

// hypTP0 = -0.9643992, tanh rational, numerator
DATA hypTP0<>+0x00(SB)/8, $0xbf76e2ddbf76e2dd
DATA hypTP0<>+0x08(SB)/8, $0xbf76e2ddbf76e2dd
DATA hypTP0<>+0x10(SB)/8, $0xbf76e2ddbf76e2dd
DATA hypTP0<>+0x18(SB)/8, $0xbf76e2ddbf76e2dd
GLOBL hypTP0<>(SB), RODATA|NOPTR, $32

// hypTP1 = -99.28772
DATA hypTP1<>+0x00(SB)/8, $0xc2c69350c2c69350
DATA hypTP1<>+0x08(SB)/8, $0xc2c69350c2c69350
DATA hypTP1<>+0x10(SB)/8, $0xc2c69350c2c69350
DATA hypTP1<>+0x18(SB)/8, $0xc2c69350c2c69350
GLOBL hypTP1<>(SB), RODATA|NOPTR, $32

// hypTP2 = -1614.6877
DATA hypTP2<>+0x00(SB)/8, $0xc4c9d602c4c9d602
DATA hypTP2<>+0x08(SB)/8, $0xc4c9d602c4c9d602
DATA hypTP2<>+0x10(SB)/8, $0xc4c9d602c4c9d602
DATA hypTP2<>+0x18(SB)/8, $0xc4c9d602c4c9d602
GLOBL hypTP2<>(SB), RODATA|NOPTR, $32

// hypTQ0 = 112.81168, tanh rational, denominator
DATA hypTQ0<>+0x00(SB)/8, $0x42e19f9442e19f94
DATA hypTQ0<>+0x08(SB)/8, $0x42e19f9442e19f94
DATA hypTQ0<>+0x10(SB)/8, $0x42e19f9442e19f94
DATA hypTQ0<>+0x18(SB)/8, $0x42e19f9442e19f94
GLOBL hypTQ0<>(SB), RODATA|NOPTR, $32

// hypTQ1 = 2235.4883
DATA hypTQ1<>+0x00(SB)/8, $0x450bb7d0450bb7d0
DATA hypTQ1<>+0x08(SB)/8, $0x450bb7d0450bb7d0
DATA hypTQ1<>+0x10(SB)/8, $0x450bb7d0450bb7d0
DATA hypTQ1<>+0x18(SB)/8, $0x450bb7d0450bb7d0
GLOBL hypTQ1<>(SB), RODATA|NOPTR, $32

// hypTQ2 = 4844.063
DATA hypTQ2<>+0x00(SB)/8, $0x4597608145976081
DATA hypTQ2<>+0x08(SB)/8, $0x4597608145976081
DATA hypTQ2<>+0x10(SB)/8, $0x4597608145976081
DATA hypTQ2<>+0x18(SB)/8, $0x4597608145976081
GLOBL hypTQ2<>(SB), RODATA|NOPTR, $32

// hypLC0 = 0.5, log(cosh) series in z = x*x
DATA hypLC0<>+0x00(SB)/8, $0x3f0000003f000000
DATA hypLC0<>+0x08(SB)/8, $0x3f0000003f000000
DATA hypLC0<>+0x10(SB)/8, $0x3f0000003f000000
DATA hypLC0<>+0x18(SB)/8, $0x3f0000003f000000
GLOBL hypLC0<>(SB), RODATA|NOPTR, $32

// hypLC1 = -0.08333334
DATA hypLC1<>+0x00(SB)/8, $0xbdaaaaabbdaaaaab
DATA hypLC1<>+0x08(SB)/8, $0xbdaaaaabbdaaaaab
DATA hypLC1<>+0x10(SB)/8, $0xbdaaaaabbdaaaaab
DATA hypLC1<>+0x18(SB)/8, $0xbdaaaaabbdaaaaab
GLOBL hypLC1<>(SB), RODATA|NOPTR, $32

// hypLC2 = 0.022222223
DATA hypLC2<>+0x00(SB)/8, $0x3cb60b613cb60b61
DATA hypLC2<>+0x08(SB)/8, $0x3cb60b613cb60b61
DATA hypLC2<>+0x10(SB)/8, $0x3cb60b613cb60b61
DATA hypLC2<>+0x18(SB)/8, $0x3cb60b613cb60b61
GLOBL hypLC2<>(SB), RODATA|NOPTR, $32

// hypLC3 = -0.006746032
DATA hypLC3<>+0x00(SB)/8, $0xbbdd0dd1bbdd0dd1
DATA hypLC3<>+0x08(SB)/8, $0xbbdd0dd1bbdd0dd1
DATA hypLC3<>+0x10(SB)/8, $0xbbdd0dd1bbdd0dd1
DATA hypLC3<>+0x18(SB)/8, $0xbbdd0dd1bbdd0dd1
GLOBL hypLC3<>(SB), RODATA|NOPTR, $32

// hypLC4 = 0.0021869488
DATA hypLC4<>+0x00(SB)/8, $0x3b0f52ea3b0f52ea
DATA hypLC4<>+0x08(SB)/8, $0x3b0f52ea3b0f52ea
DATA hypLC4<>+0x10(SB)/8, $0x3b0f52ea3b0f52ea
DATA hypLC4<>+0x18(SB)/8, $0x3b0f52ea3b0f52ea
GLOBL hypLC4<>(SB), RODATA|NOPTR, $32

// hypLC5 = -0.00073855303
DATA hypLC5<>+0x00(SB)/8, $0xba419eceba419ece
DATA hypLC5<>+0x08(SB)/8, $0xba419eceba419ece
DATA hypLC5<>+0x10(SB)/8, $0xba419eceba419ece
DATA hypLC5<>+0x18(SB)/8, $0xba419eceba419ece
GLOBL hypLC5<>(SB), RODATA|NOPTR, $32

// hypLC6 = 0.00025660123
DATA hypLC6<>+0x00(SB)/8, $0x398685a9398685a9
DATA hypLC6<>+0x08(SB)/8, $0x398685a9398685a9
DATA hypLC6<>+0x10(SB)/8, $0x398685a9398685a9
DATA hypLC6<>+0x18(SB)/8, $0x398685a9398685a9
GLOBL hypLC6<>(SB), RODATA|NOPTR, $32

// hypLC7 = -9.098576e-05
DATA hypLC7<>+0x00(SB)/8, $0xb8bed1b2b8bed1b2
DATA hypLC7<>+0x08(SB)/8, $0xb8bed1b2b8bed1b2
DATA hypLC7<>+0x10(SB)/8, $0xb8bed1b2b8bed1b2
DATA hypLC7<>+0x18(SB)/8, $0xb8bed1b2b8bed1b2
GLOBL hypLC7<>(SB), RODATA|NOPTR, $32

// hypLC8 = 3.278177e-05
DATA hypLC8<>+0x00(SB)/8, $0x38097c8238097c82
DATA hypLC8<>+0x08(SB)/8, $0x38097c8238097c82
DATA hypLC8<>+0x10(SB)/8, $0x38097c8238097c82
DATA hypLC8<>+0x18(SB)/8, $0x38097c8238097c82
GLOBL hypLC8<>(SB), RODATA|NOPTR, $32

// hypLn2 = 0.6931472, ln2 rounded to float32
DATA hypLn2<>+0x00(SB)/8, $0x3f3172183f317218
DATA hypLn2<>+0x08(SB)/8, $0x3f3172183f317218
DATA hypLn2<>+0x10(SB)/8, $0x3f3172183f317218
DATA hypLn2<>+0x18(SB)/8, $0x3f3172183f317218
GLOBL hypLn2<>(SB), RODATA|NOPTR, $32

// hypR3 = 0.33333334, atanh series for log1p
DATA hypR3<>+0x00(SB)/8, $0x3eaaaaab3eaaaaab
DATA hypR3<>+0x08(SB)/8, $0x3eaaaaab3eaaaaab
DATA hypR3<>+0x10(SB)/8, $0x3eaaaaab3eaaaaab
DATA hypR3<>+0x18(SB)/8, $0x3eaaaaab3eaaaaab
GLOBL hypR3<>(SB), RODATA|NOPTR, $32

// hypR5 = 0.2
DATA hypR5<>+0x00(SB)/8, $0x3e4ccccd3e4ccccd
DATA hypR5<>+0x08(SB)/8, $0x3e4ccccd3e4ccccd
DATA hypR5<>+0x10(SB)/8, $0x3e4ccccd3e4ccccd
DATA hypR5<>+0x18(SB)/8, $0x3e4ccccd3e4ccccd
GLOBL hypR5<>(SB), RODATA|NOPTR, $32

// hypR7 = 0.14285715
DATA hypR7<>+0x00(SB)/8, $0x3e1249253e124925
DATA hypR7<>+0x08(SB)/8, $0x3e1249253e124925
DATA hypR7<>+0x10(SB)/8, $0x3e1249253e124925
DATA hypR7<>+0x18(SB)/8, $0x3e1249253e124925
GLOBL hypR7<>(SB), RODATA|NOPTR, $32

// hypR9 = 0.11111111
DATA hypR9<>+0x00(SB)/8, $0x3de38e393de38e39
DATA hypR9<>+0x08(SB)/8, $0x3de38e393de38e39
DATA hypR9<>+0x10(SB)/8, $0x3de38e393de38e39
DATA hypR9<>+0x18(SB)/8, $0x3de38e393de38e39
GLOBL hypR9<>(SB), RODATA|NOPTR, $32

// HYPPRE turns the eight raw inputs in Y5 into the exp argument -2|x| in Y0,
// ready for EXPBODY.
#define HYPPRE \
	VANDPS hypAbsMask<>(SB), Y5, Y0 \
	VMULPS hypNegTwo<>(SB), Y0, Y0

// HYPBODY consumes x in Y5 and u = exp(-2|x|) in Y3, and leaves tanh(x) in Y8
// and log(cosh(x)) in Y9. Clobbers Y0..Y2, Y4, Y8..Y13.
//
// Y6 and Y7 are NOT touched: they hold EXPBODY's clamp constants and must
// survive to the next iteration of the loop. Y14 and Y15 are left free for the
// caller's tail mask.
//
// a, sign, the branch mask and z are rebuilt from Y5 here rather than kept
// across EXPBODY, which owns Y0..Y4. Four instructions per block is much
// cheaper than spilling to the stack.
#define HYPBODY \
	VANDPS  hypAbsMask<>(SB), Y5, Y0    \ // a = |x|
	VANDPS  hypSignMask<>(SB), Y5, Y1   \ // the sign bit, reattached at the end
	VPSUBD  hypBranchBits<>(SB), Y0, Y2 \ // integer compare on the bit patterns:
	VPSRAD  $31, Y2, Y2                 \ // all-ones iff a < 0.625; NaN sorts high
	VMULPS  Y0, Y0, Y4                  \ // z = a*a
	                                    \
	VMOVUPS     hypTP0<>(SB), Y8        \ // --- tanh, |a| < 0.625: a + a*z*P(z)/Q(z)
	VFMADD213PS hypTP1<>(SB), Y4, Y8    \
	VFMADD213PS hypTP2<>(SB), Y4, Y8    \ // Y8 = P(z)
	VADDPS      hypTQ0<>(SB), Y4, Y9    \
	VFMADD213PS hypTQ1<>(SB), Y4, Y9    \
	VFMADD213PS hypTQ2<>(SB), Y4, Y9    \ // Y9 = Q(z)
	VDIVPS      Y9, Y8, Y8              \ // Y8 = P/Q
	VMULPS      Y4, Y0, Y10             \ // a*z
	VFMADD213PS Y0, Y10, Y8             \ // Y8 = a*z*(P/Q) + a
	                                    \
	VADDPS  hypOne<>(SB), Y3, Y9        \ // --- tanh, |a| >= 0.625: 1 - 2u/(1+u)
	VADDPS  Y3, Y3, Y10                 \ // 2u
	VDIVPS  Y9, Y10, Y10                \ // 2u/(1+u)
	VMOVUPS hypOne<>(SB), Y11           \
	VSUBPS  Y10, Y11, Y9                \ // Y9 = 1 - 2u/(1+u)
	                                    \
	VBLENDVPS Y2, Y8, Y9, Y8            \ // mask set -> rational core
	VXORPS    Y1, Y8, Y8                \ // reattach sign: odd symmetry is BIT-exact
	                                    \
	VMOVUPS     hypLC8<>(SB), Y9        \ // --- log(cosh), |a| < 0.625: z*S(z)
	VFMADD213PS hypLC7<>(SB), Y4, Y9    \
	VFMADD213PS hypLC6<>(SB), Y4, Y9    \
	VFMADD213PS hypLC5<>(SB), Y4, Y9    \
	VFMADD213PS hypLC4<>(SB), Y4, Y9    \
	VFMADD213PS hypLC3<>(SB), Y4, Y9    \
	VFMADD213PS hypLC2<>(SB), Y4, Y9    \
	VFMADD213PS hypLC1<>(SB), Y4, Y9    \
	VFMADD213PS hypLC0<>(SB), Y4, Y9    \
	VMULPS      Y4, Y9, Y9              \ // Y9 = z*S(z)
	                                    \
	VADDPS      hypTwo<>(SB), Y3, Y10   \ // --- log(cosh), |a| >= 0.625:
	VDIVPS      Y10, Y3, Y10            \ // w = u/(2+u), so log1p(u) = 2*atanh(w)
	VMULPS      Y10, Y10, Y11           \ // w2
	VMOVUPS     hypR9<>(SB), Y12        \
	VFMADD213PS hypR7<>(SB), Y11, Y12   \
	VFMADD213PS hypR5<>(SB), Y11, Y12   \
	VFMADD213PS hypR3<>(SB), Y11, Y12   \
	VFMADD213PS hypOne<>(SB), Y11, Y12  \
	VADDPS      Y10, Y10, Y13           \ // 2w
	VMULPS      Y13, Y12, Y12           \ // 2w * atanh series
	VSUBPS      hypLn2<>(SB), Y0, Y13   \ // a - ln2
	VADDPS      Y13, Y12, Y10           \ // Y10 = a - ln2 + log1p(u)
	                                    \
	VBLENDVPS Y2, Y9, Y10, Y9             // mask set -> series

// func tanhLogCoshBatch32AVX2(dstTanh, dstLogCosh, src []float32) bool
TEXT ·tanhLogCoshBatch32AVX2(SB), NOSPLIT, $0-73
	MOVB $1, ret+72(FP) // this kernel never declines

	MOVQ dstTanh_base+0(FP), DI
	MOVQ dstTanh_len+8(FP), AX
	MOVQ dstLogCosh_base+24(FP), R8
	MOVQ dstLogCosh_len+32(FP), BX
	MOVQ src_base+48(FP), SI
	MOVQ src_len+56(FP), DX

	// BX = n = min(len(dstTanh), len(dstLogCosh), len(src))
	CMPQ AX, BX
	JGE  min2
	MOVQ AX, BX

min2:
	CMPQ DX, BX
	JGE  lenok
	MOVQ DX, BX

lenok:
	TESTQ BX, BX
	JEQ   empty

	EXPLOADCONSTS

	// CX = byte length of the whole-vector part, R9 = running byte offset.
	MOVQ BX, CX
	SHRQ $3, CX
	SHLQ $5, CX
	XORQ R9, R9

	TESTQ CX, CX
	JEQ   tail

loop:
	VMOVUPS (SI)(R9*1), Y5
	HYPPRE
	EXPBODY
	HYPBODY
	VMOVUPS Y8, (DI)(R9*1)
	VMOVUPS Y9, (R8)(R9*1)

	ADDQ $32, R9
	CMPQ R9, CX
	JLT  loop

tail:
	MOVQ  BX, DX
	ANDQ  $7, DX
	TESTQ DX, DX
	JEQ   done

	// Y15 holds the tail mask: row DX of the table has DX leading all-ones
	// lanes. Y15 rather than Y5, which is where x has to live, and rather
	// than anything below Y14, all of which HYPBODY uses.
	LEAQ expMaskTab<>(SB), R11
	SHLQ $5, DX
	VMOVUPS (R11)(DX*1), Y15

	// Masked-off lanes read as +0, which takes the rational branch and
	// produces 0 in both outputs. They fault on neither the load nor either
	// store, so running past the end of the slice is safe.
	VMASKMOVPS (SI)(R9*1), Y15, Y5
	HYPPRE
	EXPBODY
	HYPBODY
	VMASKMOVPS Y8, Y15, (DI)(R9*1)
	VMASKMOVPS Y9, Y15, (R8)(R9*1)

done:
	VZEROUPPER
	RET

empty:
	VZEROUPPER
	RET
