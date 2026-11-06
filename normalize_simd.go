//go:build goexperiment.simd && !amd64

package tfidf

import "simd"

// scaleVector multiplies every element of v by scale. The vector width comes
// from the simd type itself, and LoadFloat32sPart/StorePart handle the tail
// without a scalar loop.
func scaleVector(v []float32, scale float32) {
	var x simd.Float32s
	lanes := x.Len()
	s := simd.BroadcastFloat32s(scale)

	i := 0
	for ; i+lanes <= len(v); i += lanes {
		simd.LoadFloat32s(v[i:]).Mul(s).Store(v[i:])
	}
	if i < len(v) {
		u, _ := simd.LoadFloat32sPart(v[i:])
		u.Mul(s).StorePart(v[i:])
	}
}
