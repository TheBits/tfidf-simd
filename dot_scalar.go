package tfidf

// dotProductScalar is the portable fallback. It uses four independent
// accumulators to make the loop auto-vectorizable and to reduce the dependency
// chain.
func dotProductScalar(a, b []float32) float64 {
	var s0, s1, s2, s3 float32

	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}

	return float64((s0 + s1) + (s2 + s3))
}
