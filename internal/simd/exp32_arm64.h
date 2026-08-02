// Shared NEON float32 exp machinery: the constants and the EXPBODY macro.
//
// This is the arm64 counterpart of exp32_amd64.h and exists for the same
// reason: the fused tanh/log(cosh) kernel needs exactly the same exponential
// that expBatch32NEON computes, because both of its outputs are built from
// u = exp(-2|x|). One definition of the arithmetic means the two .s files
// cannot drift into computing subtly different exponentials.
//
// The symbols are `<>`-scoped, i.e. file-static, so including this header from
// two .s files in one package gives each its own copy of the RODATA rather
// than a duplicate-symbol error.
//
// # How this differs from the AVX2 version
//
// Four lanes rather than eight, so the same work covers half as many elements
// per iteration. That is the whole of the architectural difference; NEON has
// no 256-bit mode and Go's assembler has no SVE, so four is what is available.
//
// Two of the three traps documented in exp32_amd64.h do not apply here:
//
//   - There is no VMINPS/VMAXPS NaN asymmetry. AArch64's FMIN and FMAX are
//     symmetric, so the clamp constants do not have to be kept in a particular
//     operand slot. They live in registers anyway, because unlike x86 NEON has
//     no memory operands at all — every input to an arithmetic instruction is
//     a register, which is also why this macro parks twelve constants in
//     V16..V27 where the AVX2 version parks two.
//
//   - There is no VZEROUPPER and no AVX-SSE transition to avoid.
//
// The third trap is unchanged and is the one that matters: the 2^k
// reconstruction must be split into two half exponents. A single (k+127)<<23
// produces the Inf/NaN encoding at k = 128 and a sign-flipped garbage pattern
// at k = -150. See expScaleApply32 in exp32.go.
//
// # Register footprint
//
// EXPBODY reads V16..V27 and clobbers V0..V4. It leaves V5..V15 and V28..V31
// untouched, which is the headroom the fused tanh/log(cosh) kernel needs: it
// evaluates four branch bodies around this one and has to keep the original x
// live across it.
//
//	V16 expHi   V20 expC2   V24 expP1
//	V17 expLo   V21 expP4   V25 expP0
//	V18 log2e   V22 expP3   V26 1.0
//	V19 expC1   V23 expP2   V27 int32(127)

// The constants are laid out as twelve consecutive 16-byte blocks, each the
// float32 (or int32) value replicated across four lanes, so EXPLOADCONSTS can
// pull them out four registers at a time. NEON's multi-register loads require
// consecutive destinations, which is why the register assignment above is in
// table order rather than in order of use.
DATA expConstsNEON<>+0x00(SB)/8, $0x42b1721842b17218 //  88.7228391   = ln(MaxFloat32)
DATA expConstsNEON<>+0x08(SB)/8, $0x42b1721842b17218
DATA expConstsNEON<>+0x10(SB)/8, $0xc2d00000c2d00000 // -104.0
DATA expConstsNEON<>+0x18(SB)/8, $0xc2d00000c2d00000
DATA expConstsNEON<>+0x20(SB)/8, $0x3fb8aa3b3fb8aa3b //  1/ln2
DATA expConstsNEON<>+0x28(SB)/8, $0x3fb8aa3b3fb8aa3b
DATA expConstsNEON<>+0x30(SB)/8, $0x3f3180003f318000 //  0.693359375, the 9-bit head of ln2
DATA expConstsNEON<>+0x38(SB)/8, $0x3f3180003f318000
DATA expConstsNEON<>+0x40(SB)/8, $0xb95e8083b95e8083 // -2.12194440e-4, the tail of ln2
DATA expConstsNEON<>+0x48(SB)/8, $0xb95e8083b95e8083
DATA expConstsNEON<>+0x50(SB)/8, $0x3ab512333ab51233 //  expP4
DATA expConstsNEON<>+0x58(SB)/8, $0x3ab512333ab51233
DATA expConstsNEON<>+0x60(SB)/8, $0x3c091ceb3c091ceb //  expP3
DATA expConstsNEON<>+0x68(SB)/8, $0x3c091ceb3c091ceb
DATA expConstsNEON<>+0x70(SB)/8, $0x3d2aac793d2aac79 //  expP2
DATA expConstsNEON<>+0x78(SB)/8, $0x3d2aac793d2aac79
DATA expConstsNEON<>+0x80(SB)/8, $0x3e2aaa493e2aaa49 //  expP1
DATA expConstsNEON<>+0x88(SB)/8, $0x3e2aaa493e2aaa49
DATA expConstsNEON<>+0x90(SB)/8, $0x3efffffe3efffffe //  expP0
DATA expConstsNEON<>+0x98(SB)/8, $0x3efffffe3efffffe
DATA expConstsNEON<>+0xa0(SB)/8, $0x3f8000003f800000 //  1.0
DATA expConstsNEON<>+0xa8(SB)/8, $0x3f8000003f800000
DATA expConstsNEON<>+0xb0(SB)/8, $0x0000007f0000007f //  float32 exponent bias
DATA expConstsNEON<>+0xb8(SB)/8, $0x0000007f0000007f
GLOBL expConstsNEON<>(SB), RODATA|NOPTR, $192

// EXPLOADCONSTS fills V16..V27. Callers must do this once, outside the loop.
// It clobbers R8, which it walks across the table with post-increment loads.
#define EXPLOADCONSTS                                 \
	MOVD   $expConstsNEON<>(SB), R8               \
	VLD1.P 64(R8), [V16.S4, V17.S4, V18.S4, V19.S4] \
	VLD1.P 64(R8), [V20.S4, V21.S4, V22.S4, V23.S4] \
	VLD1.P 64(R8), [V24.S4, V25.S4, V26.S4, V27.S4]

// EXPBODY turns the four lanes in V0 into exp(V0) in V3.
//
// Clobbers V0, V1, V2, V3, V4 and reads V16..V27. The VMOV pairs in the
// polynomial are not redundant: VFMLA accumulates into its destination, so
// each Horner step needs its coefficient copied into a fresh register first.
// The copies alternate between V3 and V4 so no step reads a register it is
// also writing.
#define EXPBODY                                  \
	NFMAX(17, 0, 0)              /* x = max(x, -104)     */ \
	NFMIN(16, 0, 0)              /* x = min(x, 88.72)    */ \
	NFMUL(18, 0, 1)              /* fx = x * log2e       */ \
	NFRINTN(1, 1)                /* fx = rint(fx)        */ \
	VFMLS V19.S4, V1.S4, V0.S4   /* r = x - fx*C1 (exact)*/ \
	VFMLS V20.S4, V1.S4, V0.S4   /* r -= fx*C2           */ \
	NFMUL(0, 0, 2)               /* z = r*r              */ \
	VMOV  V22.B16, V3.B16        /* P3                   */ \
	VFMLA V21.S4, V0.S4, V3.S4   /* P4*r + P3            */ \
	VMOV  V23.B16, V4.B16        /* P2                   */ \
	VFMLA V3.S4, V0.S4, V4.S4    /* *r + P2              */ \
	VMOV  V24.B16, V3.B16        /* P1                   */ \
	VFMLA V4.S4, V0.S4, V3.S4    /* *r + P1              */ \
	VMOV  V25.B16, V4.B16        /* P0                   */ \
	VFMLA V3.S4, V0.S4, V4.S4    /* *r + P0 = P(r)       */ \
	VMOV  V0.B16, V3.B16         /* r                    */ \
	VFMLA V4.S4, V2.S4, V3.S4    /* P*z + r              */ \
	NFADD(26, 3, 3)              /* + 1 => exp(r)        */ \
	NFCVTZS(1, 1)                /* k = int32(fx)        */ \
	NSSHR(1, 1, 4)               /* k1 = k >> 1          */ \
	VSUB  V4.S4, V1.S4, V1.S4    /* k2 = k - k1          */ \
	VADD  V27.S4, V4.S4, V4.S4   /* k1 + bias            */ \
	VADD  V27.S4, V1.S4, V1.S4   /* k2 + bias            */ \
	VSHL  $23, V4.S4, V4.S4      /* 2^k1                 */ \
	VSHL  $23, V1.S4, V1.S4      /* 2^k2                 */ \
	NFMUL(4, 3, 3)                                          \
	NFMUL(1, 3, 3)
