//go:build !goexperiment.simd

package tfidf

// dotProduct is the portable implementation used when the simd experiment is
// disabled.
func dotProduct(a, b []float32) float64 {
	return dotProductScalar(a, b)
}
