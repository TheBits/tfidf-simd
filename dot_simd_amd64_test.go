//go:build goexperiment.simd && amd64

package tfidf

import (
	"math"
	"testing"

	"simd/archsimd"
)

func TestAMD64SIMDDotProductPaths(t *testing.T) {
	sizes := []int{0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000}

	for _, n := range sizes {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			a[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
			b[i] = float32(math.Cos(float64(i)*0.19)) * 0.5
		}
		want := dotProductScalar(a, b)

		check := func(name string, got float64) {
			if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Errorf("%s, length %d: got %v, want %v", name, n, got, want)
			}
		}

		check("dispatch", dotProduct(a, b))

		if !archsimd.X86.AVX() {
			continue
		}
		check("avx4", dotProductAVX4(a, b))
		if archsimd.X86.FMA() {
			check("avx4-fma", dotProductAVX4FMA(a, b))
		}
		if archsimd.X86.AVX2() {
			check("avx8", dotProductAVX8(a, b))
			if archsimd.X86.FMA() {
				check("avx8-fma", dotProductAVX8FMA(a, b))
			}
		}
		if archsimd.X86.AVX512() {
			check("avx512", dotProductAVX512(a, b))
		}
	}
}
