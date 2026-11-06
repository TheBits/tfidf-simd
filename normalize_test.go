package tfidf

import (
	"math"
	"testing"
)

func TestNormalizeSizes(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 1000, 4096}

	for _, n := range sizes {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
			// Make sure the vector is non-zero even for small lengths where
			// sin(0) is zero.
			if v[i] == 0 {
				v[i] = 0.5
			}
		}

		normalize(v)

		if n == 0 {
			continue
		}
		got := norm(v)
		if math.Abs(got-1) > 1e-5 {
			t.Errorf("length %d: L2 norm = %v, want 1", n, got)
		}
	}
}

func TestNormalizeZeroVector(t *testing.T) {
	v := make([]float32, 33)
	normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Fatalf("normalize of zero vector changed v[%d] to %v", i, x)
		}
	}
}
