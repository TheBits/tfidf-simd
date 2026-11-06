//go:build goexperiment.simd && !amd64

package tfidf

import "simd"

// dotProduct follows the inner-product example from the Go SIMD blog. It uses
// the vector's own length instead of a hard-coded or queried width, so the same
// code works with any portable simd vector size. Four accumulators hide FMA
// latency.
func dotProduct(a, b []float32) float64 {
	var s0, s1, s2, s3 simd.Float32s
	lanes := s0.Len()

	i := 0
	for ; i+4*lanes <= len(a); i += 4 * lanes {
		s0 = simd.LoadFloat32s(a[i:]).MulAdd(simd.LoadFloat32s(b[i:]), s0)
		s1 = simd.LoadFloat32s(a[i+lanes:]).MulAdd(simd.LoadFloat32s(b[i+lanes:]), s1)
		s2 = simd.LoadFloat32s(a[i+2*lanes:]).MulAdd(simd.LoadFloat32s(b[i+2*lanes:]), s2)
		s3 = simd.LoadFloat32s(a[i+3*lanes:]).MulAdd(simd.LoadFloat32s(b[i+3*lanes:]), s3)
	}
	for ; i+lanes <= len(a); i += lanes {
		s0 = simd.LoadFloat32s(a[i:]).MulAdd(simd.LoadFloat32s(b[i:]), s0)
	}

	acc := s0.Add(s1).Add(s2).Add(s3)

	// LoadFloat32sPart zero-fills the unused lanes, so the short tail can use
	// the same multiply-add operation as the main loop.
	if i < len(a) {
		u, _ := simd.LoadFloat32sPart(a[i:])
		v, _ := simd.LoadFloat32sPart(b[i:])
		acc = u.MulAdd(v, acc)
	}

	return float64(sumFloat32s(acc))
}

// sumFloat32s reduces a vector to a scalar. Go 1.27 does not yet provide
// ReduceSum; using Len keeps the buffer correct for any vector width.
func sumFloat32s(v simd.Float32s) float32 {
	var small [16]float32
	buf := small[:]
	if n := v.Len(); n > len(buf) {
		buf = make([]float32, n)
	}
	n := v.StorePart(buf)

	var sum float32
	for _, x := range buf[:n] {
		sum += x
	}
	return sum
}
