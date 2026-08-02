// NEON floating-point vector instructions that Go's arm64 assembler does not
// provide, encoded by hand.
//
// # Why this file has to exist
//
// Go's arm64 assembler recognises almost no vector floating-point arithmetic.
// The complete set of vector mnemonics it accepts is
//
//	VADD VADDP VADDV VAND VBCAX VBIF VBIT VBSL VCMEQ VCMTST VCNT VDUP VEOR
//	VEOR3 VEXT VFMLA VFMLS VLD1 VLD1R VLD2 VLD2R VLD3 VLD3R VLD4 VLD4R VMOV
//	VMOVD VMOVI VMOVQ VMOVS VORR VPMULL VPMULL2 VRAX1 VRBIT VREV16 VREV32
//	VREV64 VSHL VSLI VSRI VST1 VST2 VST3 VST4 VSUB VTBL VTBX VTRN1 VTRN2
//	VUADDLV VUADDW VUADDW2 VUMAX VUMIN VUSHLL VUSHLL2 VUSHR VUSRA VUXTL
//	VUXTL2 VUZP1 VUZP2 VXAR VZIP1 VZIP2
//
// VADD, VSUB, VUMAX and VUMIN are integer-only, and VFMLA / VFMLS are the only
// floating-point arithmetic in the list. There is no vector FMUL, FADD, FSUB,
// FDIV, FMIN, FMAX, FABS, FCVTZS, SCVTF or arithmetic right shift. FMULS,
// FADDS and friends exist but are scalar, one lane at a time, which defeats the
// purpose. So every kernel in this package that does more than multiply-add has
// to reach the hardware through WORD.
//
// # How to read a macro
//
// Arguments are register NUMBERS, not register names, because WORD takes an
// integer expression and the assembler's preprocessor cannot turn V7 into 7.
// The argument order is (m, n, d) so that an invocation reads in the same
// Plan 9 order as the native mnemonics beside it — destination last:
//
//	VFMLA   V2.S4, V1.S4, V0.S4   // V0 += V1 * V2   (native)
//	NFMUL(2, 1, 0)                // V0  = V1 * V2   (this file)
//
// Every call site carries a comment naming the registers, because the numbers
// alone are unreadable and `go vet` cannot check a WORD.
//
// # Why these encodings can be trusted
//
// Each one was assembled and then disassembled with `go tool objdump`, which
// decodes them via golang.org/x/arch/arm64 — an independent implementation of
// the encoding, not the one that produced the bytes. TestNEONWordEncodings
// re-runs that check as part of the suite: it disassembles this package's own
// text and asserts the expected mnemonics appear, so a mistyped hex digit
// cannot reach a release just because the arithmetic happened to still look
// plausible on a test grid.
//
// All operate on the .S4 arrangement: four 32-bit lanes, the full 128-bit
// register. There is no .S2 variant here; nothing in this package wants one.

// Advanced SIMD three-same, size = single precision.
//
//	0 Q U 01110 size 1 Rm opcode 1 Rn Rd
#define NFADD(m, n, d)  WORD $(0x4E20D400 | ((m)<<16) | ((n)<<5) | (d)) // d = n + m
#define NFSUB(m, n, d)  WORD $(0x4EA0D400 | ((m)<<16) | ((n)<<5) | (d)) // d = n - m
#define NFMUL(m, n, d)  WORD $(0x6E20DC00 | ((m)<<16) | ((n)<<5) | (d)) // d = n * m
#define NFDIV(m, n, d)  WORD $(0x6E20FC00 | ((m)<<16) | ((n)<<5) | (d)) // d = n / m

// NFMAX / NFMIN are the NaN-propagating pair (FMAX/FMIN, not FMAXNM/FMINNM).
// If either operand is a NaN the result is a NaN, which is what Go's min and
// max do and therefore what the pure-Go kernels these transliterate do.
//
// Unlike x86's VMINPS/VMAXPS there is no operand-order trap here: AArch64's
// FMIN and FMAX are symmetric in their NaN handling, so a swapped pair is
// merely wrong about which number it returns, not silently wrong about NaN.
// The payload is not preserved — the result is the default quiet NaN rather
// than the input — and the tests assert NaN-ness rather than a bit pattern for
// exactly that reason.
#define NFMAX(m, n, d)  WORD $(0x4E20F400 | ((m)<<16) | ((n)<<5) | (d)) // d = max(n, m)
#define NFMIN(m, n, d)  WORD $(0x4EA0F400 | ((m)<<16) | ((n)<<5) | (d)) // d = min(n, m)

// Vector compares produce an all-ones / all-zeros lane mask, the shape VBSL
// wants as its selector.
#define NFCMGE(m, n, d) WORD $(0x6E20E400 | ((m)<<16) | ((n)<<5) | (d)) // d = n >= m
#define NFCMGT(m, n, d) WORD $(0x6EA0E400 | ((m)<<16) | ((n)<<5) | (d)) // d = n > m

// Advanced SIMD two-register misc.
//
//	0 Q U 01110 size 10000 opcode 10 Rn Rd
#define NFABS(n, d)     WORD $(0x4EA0F800 | ((n)<<5) | (d)) // d = |n|
#define NFNEG(n, d)     WORD $(0x6EA0F800 | ((n)<<5) | (d)) // d = -n

// NFRINTN rounds to nearest with ties to even, the same mode as x86's
// VROUNDPS $0x08 and as Go's own round-to-nearest float conversions.
#define NFRINTN(n, d)   WORD $(0x4E218800 | ((n)<<5) | (d)) // d = rint(n)

#define NFCVTZS(n, d)   WORD $(0x4EA1B800 | ((n)<<5) | (d)) // d = int32(n), toward zero
#define NSCVTF(n, d)    WORD $(0x4E21D800 | ((n)<<5) | (d)) // d = float32(int32 n)

// NSSHR is an ARITHMETIC right shift, which VUSHR is not. Both users need the
// sign: EXPBODY computes k>>1 for k as low as -150, where a logical shift
// yields a large positive exponent instead of a small negative one, and
// HYPBODY broadcasts a comparison's sign bit across the lane with a shift of 31.
//
// The immediate is encoded as immh:immb = 64 - shift, which for any shift in
// 1..32 leaves immh in 0b0100..0b0111 — the range that selects 32-bit
// elements. Shifts outside that range would silently change the element size,
// so this macro is only valid for shift in 1..32.
#define NSSHR(sh, n, d) WORD $(0x4F000400 | ((64-(sh))<<16) | ((n)<<5) | (d)) // d = n >> sh, signed
