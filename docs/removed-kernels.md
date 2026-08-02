# Removed kernels

`FastSqrt`, `FastInvSqrt` and `FastRecip` (and their internal kernels) were
removed from this library. They were measured at roughly 6.9x, 4.3x and 9x
*slower* than the hardware they were meant to replace: `math.Sqrt` compiles to
`SQRTSD` and `1.0/x` to `DIVSD`, and Go's compiler intrinsifies `math.Sqrt` on
every `GOARCH` it supports. There is no target on which the approximations win,
so they were deleted outright rather than deprecated.

`FastRoot(x, 2)` now delegates to `math.Sqrt`, which makes it both faster and
more accurate than it was.

What follows is preserved from the doc comment on the deleted reciprocal kernel
(`internal/approx/recip.go`). The convergence analysis is the reason the seed
needed a cubic polish, and it is worth keeping even though the code is gone.

> Recip returns an approximate 1/x.
>
> Go has no reciprocal intrinsic: the expression 1/x lowers to a true DIVSD on
> amd64 (and FDIV on arm64). This routine replaces that with a bit-trick seed
> plus Newton-Raphson, which trades one long-latency instruction for a chain
> of short ones. Whether that is a win is entirely a property of the caller's
> dependency structure; see the benchmark discussion in README.md before
> reaching for it.
>
> Precision selects the number of quadratic Newton steps applied on top of the
> seed:
>
> ```
> PrecisionFast      1 step   ~1.7e-8  relative
> PrecisionBalanced  2 steps  ~3e-16   relative (full float64 in practice)
> PrecisionHigh      3 steps  <=1 ulp
> ```
>
> The seed itself is the magic-constant estimate (5.05e-2) polished by one
> cubic step, y <- y*(1 + r + r^2) with r = 1 - m*y, giving ~1.3e-4. Without
> that polish the requested step counts cannot reach the accuracies above: a
> bare magic-constant seed converges 5.05e-2 -> 2.6e-3 -> 6.6e-6 -> 4.4e-11,
> so two Newton steps would land five orders short of full precision.
>
> Edge cases match 1/x exactly: NaN -> NaN, +/-0 -> +/-Inf, +/-Inf -> +/-0,
> and subnormal inputs are normalized before the seed so they do not fall off
> the bottom of the exponent field.
