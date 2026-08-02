package consumerbench

import (
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

// TestCrossModuleInlining is the regression gate for the defect this module
// exists to catch.
//
// The library's arithmetic used to live inside generic function bodies. Go
// compiles a generic body once per gcshape and reaches it through a runtime
// dictionary, and it will not inline such a body across a package boundary --
// so every call from a real consumer paid a frame that no caller could remove.
// The symptom was a library that won its own in-package benchmarks and lost
// every consumer's. The fix was structural: each algorithm moved into a
// non-generic float64 kernel, with the generic function reduced to a shim
// small enough for the compiler to inline into the caller.
//
// That property is *static*, so it can be asserted deterministically instead of
// being inferred from timings. `go build -gcflags=-m` prints the compiler's
// inlining decisions, and because generic instantiations are compiled into the
// *importing* package, building this module reports the decisions made for
// algo-approx's shims at this module's own call sites -- exactly the
// cross-module configuration that matters and the only one in which the
// property is observable.
//
// This is the same technique the Go project uses on itself
// (cmd/compile/internal/test/inl_test.go, TestIntendedInlining). It is
// deterministic: the compiler's inlining budget is a fixed cost model, not a
// measurement, so this test either passes or fails identically on every run and
// on every machine. Contrast the timings, which cannot be gated in CI at all --
// see the note in .github/workflows/test-bench.yaml.
func TestCrossModuleInlining(t *testing.T) {
	t.Parallel()

	if runtime.Compiler != "gc" {
		t.Skipf("inlining diagnostics are gc-specific; compiler is %q", runtime.Compiler)
	}

	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skipf("no go tool in PATH: %v", err)
	}

	// A non-main package produces no output file, so this only populates the
	// build cache. -gcflags=-m applies to the package named on the command
	// line, which is what we want: the instantiations reported below are the
	// ones compiled into this consumer module.
	cmd := exec.Command(goTool, "build", "-gcflags=-m", ".")

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build -gcflags=-m: %v\n%s", err, out)
	}

	lines := strings.Split(string(out), "\n")

	// Every algorithm that was converted to the kernel-plus-shim structure.
	for _, name := range []string{"Log", "Exp", "Tanh", "LogCosh"} {
		// The internal generic shim must still be small enough to inline. If
		// the arithmetic moves back into the generic body, this is the line
		// that disappears.
		want := "can inline approx." + name + "[go.shape.float64]"
		if !hasLineWith(lines, "internal/approx/", want) {
			t.Errorf("internal shim no longer inlinable: no %q from internal/approx\n%s", want, out)
		}

		// ...and the public entry points must actually be inlined into this
		// module's call sites in callsites.go, both the generic one and the
		// concrete float64 one.
		for _, call := range []string{
			"inlining call to approx.Fast" + name + "[go.shape.float64]",
			"inlining call to approx.Fast" + name + "64",
		} {
			if !hasLineWith(lines, "callsites.go", call) {
				t.Errorf("not inlined into the consumer: no %q at a callsites.go call site\n%s", call, out)
			}
		}
	}
}

func hasLineWith(lines []string, substrs ...string) bool {
	for _, line := range lines {
		matched := true

		for _, s := range substrs {
			if !strings.Contains(line, s) {
				matched = false

				break
			}
		}

		if matched {
			return true
		}
	}

	return false
}
