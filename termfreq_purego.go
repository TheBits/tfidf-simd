//go:build !goexperiment.simd

package tfidf

// countEqual returns the number of elements in v equal to x. This is the
// portable fallback used when the simd experiment is disabled.
func countEqual(v []uint32, x uint32) int {
	total := 0
	for _, y := range v {
		if y == x {
			total++
		}
	}
	return total
}
