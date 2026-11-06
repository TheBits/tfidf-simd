//go:build goexperiment.simd && amd64

package tfidf

import (
	"math"
	"testing"

	"simd/archsimd"
)

func TestAMD64SIMDScaleVectorPaths(t *testing.T) {
	sizes := []int{0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000}
	const scale = 0.75

	for _, n := range sizes {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
		}

		want := append([]float32(nil), src...)
		scaleVectorScalar(want, scale)

		check := func(name string, fn func([]float32, float32)) {
			got := append([]float32(nil), src...)
			fn(got, scale)
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("%s, length %d, index %d: got %v, want %v", name, n, i, got[i], want[i])
					return
				}
			}
		}

		check("dispatch", scaleVector)

		if !archsimd.X86.AVX() {
			continue
		}
		check("avx4", scaleVectorAVX4)
		if archsimd.X86.AVX2() {
			check("avx8", scaleVectorAVX8)
		}
		if archsimd.X86.AVX512() {
			check("avx512", scaleVectorAVX512)
		}
	}
}
