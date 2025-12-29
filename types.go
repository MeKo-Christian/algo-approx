package approx

// Float is the type constraint for floating-point inputs supported by this library.
//
// The tilde (~) allows user-defined named types whose underlying type is float32
// or float64.
type Float interface {
	~float32 | ~float64
}
