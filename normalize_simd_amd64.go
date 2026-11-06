//go:build goexperiment.simd && amd64

package tfidf

import (
	"simd"
	"simd/archsimd"
)

// scaleVector dispatches through the simd bridge type, mirroring dotProduct.
// The compiler specializes this function for the active vector width and
// removes the type switch, so there is no dispatch in the hot loop.
func scaleVector(v []float32, scale float32) {
	var x simd.Float32s
	switch x.ToArch().(type) {
	case archsimd.Float32x16:
		scaleVectorAVX512(v, scale)
	case archsimd.Float32x8:
		scaleVectorAVX8(v, scale)
	case archsimd.Float32x4:
		scaleVectorAVX4(v, scale)
	default:
		// Emulation, for example GODEBUG=simd=0.
		scaleVectorScalar(v, scale)
	}
}

// scaleVectorAVX512 multiplies v by scale with 512-bit AVX-512 vectors.
func scaleVectorAVX512(v []float32, scale float32) {
	s := archsimd.BroadcastFloat32x16(scale)

	i := 0
	for ; i+16 <= len(v); i += 16 {
		archsimd.LoadFloat32x16(v[i:]).Mul(s).Store(v[i:])
	}
	if i < len(v) {
		u, _ := archsimd.LoadFloat32x16Part(v[i:])
		u.Mul(s).StorePart(v[i:])
	}
}

// scaleVectorAVX8 multiplies v by scale with 256-bit AVX vectors.
func scaleVectorAVX8(v []float32, scale float32) {
	s := archsimd.BroadcastFloat32x8(scale)

	i := 0
	for ; i+8 <= len(v); i += 8 {
		archsimd.LoadFloat32x8(v[i:]).Mul(s).Store(v[i:])
	}
	if i < len(v) {
		u, _ := archsimd.LoadFloat32x8Part(v[i:])
		u.Mul(s).StorePart(v[i:])
	}
}

// scaleVectorAVX4 multiplies v by scale with 128-bit AVX vectors.
func scaleVectorAVX4(v []float32, scale float32) {
	s := archsimd.BroadcastFloat32x4(scale)

	i := 0
	for ; i+4 <= len(v); i += 4 {
		archsimd.LoadFloat32x4(v[i:]).Mul(s).Store(v[i:])
	}
	if i < len(v) {
		u, _ := archsimd.LoadFloat32x4Part(v[i:])
		u.Mul(s).StorePart(v[i:])
	}
}

// scaleVectorScalar is the emulation fallback.
func scaleVectorScalar(v []float32, scale float32) {
	for i := range v {
		v[i] *= scale
	}
}
