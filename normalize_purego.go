//go:build !goexperiment.simd

package tfidf

// scaleVector multiplies every element of v by scale. This is the portable
// implementation used when the simd experiment is disabled.
func scaleVector(v []float32, scale float32) {
	for i := range v {
		v[i] *= scale
	}
}
