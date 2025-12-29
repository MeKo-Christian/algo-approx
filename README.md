# algo-approx

Fast, allocation-free mathematical approximations for Go.

## Status

Phase 1 MVP is implemented: `sqrt`, `invsqrt`, `log` (ln), and `exp` with `float32`/`float64` generics and a `Precision` knob.

## Install

```bash
go get github.com/meko-christian/algo-approx
```

## Usage

```go
package main

import (
	"fmt"

	"github.com/meko-christian/algo-approx"
)

func main() {
	fmt.Println(approx.FastSqrt(16.0))
	fmt.Println(approx.FastLog(10.0))
	fmt.Println(approx.FastExp(2.0))

	// Precision control
	fmt.Println(approx.FastSqrtPrec(16.0, approx.PrecisionHigh))
}
```

## Benchmarks (2025-12-28)

Run:

```bash
go test -bench=. -benchmem -run=^$ ./...
```

Results below are from Linux/amd64 on an Intel i7-1255U.

| Operation  | approx ns/op | math ns/op | approx vs math |
| ---------- | -----------: | ---------: | -------------: |
| `Sqrt`     |        11.09 |      1.942 |   5.71× slower |
| `InvSqrt`  |        12.83 |      5.887 |   2.18× slower |
| `Log` (ln) |        6.398 |      10.83 |   1.69× faster |
| `Exp`      |        7.269 |      13.56 |   1.87× faster |

These numbers are expected to vary across CPUs/Go versions. Right now the focus is correctness + a stable API; performance tuning is still pending.

## Accuracy

See [ACCURACY.md](ACCURACY.md) for measured error metrics on representative ranges.

## License

MIT
