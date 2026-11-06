//go:build goexperiment.simd && amd64

package tfidf

import (
	"simd"
	"simd/archsimd"
)

// dotProduct dispatches through the simd bridge type, the pattern recommended
// by the Go SIMD blog. The compiler specializes this function for the active
// vector width and removes the type switch, so there is no dispatch in the hot
// loop. GODEBUG=simd=... is honored by the simd package.
func dotProduct(a, b []float32) float64 {
	var v simd.Float32s
	switch v.ToArch().(type) {
	case archsimd.Float32x16:
		return dotProductAVX512(a, b)
	case archsimd.Float32x8:
		if archsimd.X86.FMA() {
			return dotProductAVX8FMA(a, b)
		}
		return dotProductAVX8(a, b)
	case archsimd.Float32x4:
		if archsimd.X86.FMA() {
			return dotProductAVX4FMA(a, b)
		}
		return dotProductAVX4(a, b)
	default:
		// Emulation, for example GODEBUG=simd=0.
		return dotProductScalar(a, b)
	}
}

// dotProductAVX512 uses 512-bit AVX-512 vectors with fused multiply-add.
func dotProductAVX512(a, b []float32) float64 {
	var s0, s1, s2, s3 archsimd.Float32x16

	i := 0
	for ; i+64 <= len(a); i += 64 {
		s0 = archsimd.LoadFloat32x16(a[i:]).MulAdd(archsimd.LoadFloat32x16(b[i:]), s0)
		s1 = archsimd.LoadFloat32x16(a[i+16:]).MulAdd(archsimd.LoadFloat32x16(b[i+16:]), s1)
		s2 = archsimd.LoadFloat32x16(a[i+32:]).MulAdd(archsimd.LoadFloat32x16(b[i+32:]), s2)
		s3 = archsimd.LoadFloat32x16(a[i+48:]).MulAdd(archsimd.LoadFloat32x16(b[i+48:]), s3)
	}
	for ; i+16 <= len(a); i += 16 {
		s0 = archsimd.LoadFloat32x16(a[i:]).MulAdd(archsimd.LoadFloat32x16(b[i:]), s0)
	}
	if i < len(a) {
		u, _ := archsimd.LoadFloat32x16Part(a[i:])
		v, _ := archsimd.LoadFloat32x16Part(b[i:])
		s1 = u.MulAdd(v, s1)
	}

	var lanes [16]float32
	s0.Add(s1).Add(s2).Add(s3).StoreArray(&lanes)

	var sum float32
	for _, x := range lanes {
		sum += x
	}
	return float64(sum)
}

// dotProductAVX8FMA uses 256-bit AVX vectors with fused multiply-add.
func dotProductAVX8FMA(a, b []float32) float64 {
	var s0, s1, s2, s3 archsimd.Float32x8

	i := 0
	for ; i+32 <= len(a); i += 32 {
		s0 = archsimd.LoadFloat32x8(a[i:]).MulAdd(archsimd.LoadFloat32x8(b[i:]), s0)
		s1 = archsimd.LoadFloat32x8(a[i+8:]).MulAdd(archsimd.LoadFloat32x8(b[i+8:]), s1)
		s2 = archsimd.LoadFloat32x8(a[i+16:]).MulAdd(archsimd.LoadFloat32x8(b[i+16:]), s2)
		s3 = archsimd.LoadFloat32x8(a[i+24:]).MulAdd(archsimd.LoadFloat32x8(b[i+24:]), s3)
	}
	for ; i+8 <= len(a); i += 8 {
		s0 = archsimd.LoadFloat32x8(a[i:]).MulAdd(archsimd.LoadFloat32x8(b[i:]), s0)
	}
	if i < len(a) {
		u, _ := archsimd.LoadFloat32x8Part(a[i:])
		v, _ := archsimd.LoadFloat32x8Part(b[i:])
		s1 = u.MulAdd(v, s1)
	}

	var lanes [8]float32
	s0.Add(s1).Add(s2).Add(s3).StoreArray(&lanes)

	var sum float32
	for _, x := range lanes {
		sum += x
	}
	return float64(sum)
}

// dotProductAVX8 uses 256-bit AVX vectors without requiring FMA.
func dotProductAVX8(a, b []float32) float64 {
	var s0, s1, s2, s3 archsimd.Float32x8

	i := 0
	for ; i+32 <= len(a); i += 32 {
		s0 = archsimd.LoadFloat32x8(a[i:]).Mul(archsimd.LoadFloat32x8(b[i:])).Add(s0)
		s1 = archsimd.LoadFloat32x8(a[i+8:]).Mul(archsimd.LoadFloat32x8(b[i+8:])).Add(s1)
		s2 = archsimd.LoadFloat32x8(a[i+16:]).Mul(archsimd.LoadFloat32x8(b[i+16:])).Add(s2)
		s3 = archsimd.LoadFloat32x8(a[i+24:]).Mul(archsimd.LoadFloat32x8(b[i+24:])).Add(s3)
	}
	for ; i+8 <= len(a); i += 8 {
		s0 = archsimd.LoadFloat32x8(a[i:]).Mul(archsimd.LoadFloat32x8(b[i:])).Add(s0)
	}
	if i < len(a) {
		u, _ := archsimd.LoadFloat32x8Part(a[i:])
		v, _ := archsimd.LoadFloat32x8Part(b[i:])
		s1 = u.Mul(v).Add(s1)
	}

	var lanes [8]float32
	s0.Add(s1).Add(s2).Add(s3).StoreArray(&lanes)

	var sum float32
	for _, x := range lanes {
		sum += x
	}
	return float64(sum)
}

// dotProductAVX4FMA uses 128-bit AVX vectors with fused multiply-add.
func dotProductAVX4FMA(a, b []float32) float64 {
	var s0, s1, s2, s3 archsimd.Float32x4

	i := 0
	for ; i+16 <= len(a); i += 16 {
		s0 = archsimd.LoadFloat32x4(a[i:]).MulAdd(archsimd.LoadFloat32x4(b[i:]), s0)
		s1 = archsimd.LoadFloat32x4(a[i+4:]).MulAdd(archsimd.LoadFloat32x4(b[i+4:]), s1)
		s2 = archsimd.LoadFloat32x4(a[i+8:]).MulAdd(archsimd.LoadFloat32x4(b[i+8:]), s2)
		s3 = archsimd.LoadFloat32x4(a[i+12:]).MulAdd(archsimd.LoadFloat32x4(b[i+12:]), s3)
	}
	for ; i+4 <= len(a); i += 4 {
		s0 = archsimd.LoadFloat32x4(a[i:]).MulAdd(archsimd.LoadFloat32x4(b[i:]), s0)
	}
	if i < len(a) {
		u, _ := archsimd.LoadFloat32x4Part(a[i:])
		v, _ := archsimd.LoadFloat32x4Part(b[i:])
		s1 = u.MulAdd(v, s1)
	}

	var lanes [4]float32
	s0.Add(s1).Add(s2).Add(s3).StoreArray(&lanes)

	var sum float32
	for _, x := range lanes {
		sum += x
	}
	return float64(sum)
}

// dotProductAVX4 uses 128-bit AVX vectors without requiring FMA.
func dotProductAVX4(a, b []float32) float64 {
	var s0, s1, s2, s3 archsimd.Float32x4

	i := 0
	for ; i+16 <= len(a); i += 16 {
		s0 = archsimd.LoadFloat32x4(a[i:]).Mul(archsimd.LoadFloat32x4(b[i:])).Add(s0)
		s1 = archsimd.LoadFloat32x4(a[i+4:]).Mul(archsimd.LoadFloat32x4(b[i+4:])).Add(s1)
		s2 = archsimd.LoadFloat32x4(a[i+8:]).Mul(archsimd.LoadFloat32x4(b[i+8:])).Add(s2)
		s3 = archsimd.LoadFloat32x4(a[i+12:]).Mul(archsimd.LoadFloat32x4(b[i+12:])).Add(s3)
	}
	for ; i+4 <= len(a); i += 4 {
		s0 = archsimd.LoadFloat32x4(a[i:]).Mul(archsimd.LoadFloat32x4(b[i:])).Add(s0)
	}
	if i < len(a) {
		u, _ := archsimd.LoadFloat32x4Part(a[i:])
		v, _ := archsimd.LoadFloat32x4Part(b[i:])
		s1 = u.Mul(v).Add(s1)
	}

	var lanes [4]float32
	s0.Add(s1).Add(s2).Add(s3).StoreArray(&lanes)

	var sum float32
	for _, x := range lanes {
		sum += x
	}
	return float64(sum)
}
