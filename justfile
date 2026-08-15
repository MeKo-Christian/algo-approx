# Build the library
build:
    go build -v ./...

# Run all tests
test:
    go test -v -race -count=1 ./...

# Build, vet and test the nested consumerbench module.
# `go build ./...` and `go test ./...` at the root do NOT descend into nested
# modules, so this recipe is the only thing that covers consumerbench/ at all.
# It carries TestCrossModuleInlining -- the static gate on the cross-module
# inlining property -- and finishes with a -benchtime=1x smoke run that proves
# the benchmarks still compile and execute without measuring anything.
# Mirrors the test-consumer job in .github/workflows/test-unit.yaml.
test-consumer:
    cd consumerbench && go build -v ./...
    cd consumerbench && go vet ./...
    cd consumerbench && go test -v -race -count=1 ./...
    cd consumerbench && go test -run=^$ -bench=. -benchtime=1x ./...

# Run benchmarks
bench:
    go test -bench=. -benchmem -run=^$ ./...

# Run the in-package benchmarks the way the published table was measured.
# GOMAXPROCS=1 and -count>=4 are part of the method, not decoration.
bench-published:
    GOMAXPROCS=1 go test -run=^$ -bench=. -benchtime=400ms -count=4 .

# Run the cross-module consumer benchmarks. These are the numbers a real
# caller sees: consumerbench/ is its own module and imports algo-approx by
# module path, so nothing gets the same-package inlining that flatters the
# in-package benchmarks.
bench-consumer:
    cd consumerbench && GOMAXPROCS=1 go test -run=^$ -bench=. -benchtime=400ms -count=6 .

# Run linters.
# The second line is not redundant: consumerbench/ is a nested module and
# golangci-lint, like `go test ./...`, does not descend into one. Run from
# inside it, golangci-lint walks up and finds the same .golangci.toml.
lint:
    golangci-lint run
    cd consumerbench && golangci-lint run

# Run linters and fix issues
lint-fix:
    golangci-lint run --fix
    cd consumerbench && golangci-lint run --fix

# Format code using treefmt
fmt:
    treefmt . --allow-missing-formatter

# Generate coverage report
cover:
    go test -coverprofile=coverage.txt -covermode=atomic ./...
    go tool cover -html=coverage.txt -o coverage.html

# Clean build artifacts
clean:
    rm -f coverage.txt coverage.html

# Run all checks (test, nested consumer module, lint, coverage).
# test-consumer is not optional here: it is the local mirror of the CI job that
# covers consumerbench/, and `just check` and the GitHub workflow must not
# disagree about what is gated.
check: test test-consumer lint cover check-deps

# Cross-compile for ARM64
build-arm64:
    GOOS=linux GOARCH=arm64 go build -v ./...

# Run tests on ARM64 using QEMU (requires qemu-user-static)
test-arm64:
    #!/usr/bin/env bash
    if ! command -v qemu-aarch64-static &> /dev/null; then
        echo "Error: qemu-aarch64-static not found"
        echo "Install with: sudo apt-get install qemu-user-static binfmt-support"
        exit 1
    fi
    GOOS=linux GOARCH=arm64 go test -exec="qemu-aarch64-static" -v -count=1 ./...

# Run benchmarks on ARM64 using QEMU (NOTE: performance not representative, correctness only)
bench-arm64:
    #!/usr/bin/env bash
    if ! command -v qemu-aarch64-static &> /dev/null; then
        echo "Error: qemu-aarch64-static not found"
        echo "Install with: sudo apt-get install qemu-user-static binfmt-support"
        exit 1
    fi
    @echo "NOTE: QEMU benchmarks are for correctness validation only, not performance measurement"
    GOOS=linux GOARCH=arm64 go test -exec="qemu-aarch64-static" -bench=. -benchmem -run=^$ ./...

# Build for both amd64 and arm64
build-all: build build-arm64
    @echo "Built for amd64 and arm64"

# Test on both amd64 and arm64
test-all: test test-arm64
    @echo "Tests passed on both architectures"

# Run all checks on both architectures
check-all: check test-arm64
    @echo "All checks passed on amd64 and arm64"

# Default target
default: build

fix:
    just lint-fix
    just fmt

# Are all github.com/cwbudde/* dependencies at their latest tags?
check-deps:
    ./scripts/release-guard.sh deps

# How much work is sitting on main past the latest tag?
check-unreleased:
    ./scripts/release-guard.sh unreleased

# Check every release precondition for VERSION without tagging anything.
release-check VERSION:
    ./scripts/release-guard.sh gate {{VERSION}}

# Tag VERSION: run the full gate, then create and push the annotated tag.
# Refuses on a dirty tree, stale siblings, a missing CHANGELOG section, or an
# incompatible API change the version does not signal. See AGENTS.md.
tag-release VERSION:
    ./scripts/release-guard.sh tag {{VERSION}}
