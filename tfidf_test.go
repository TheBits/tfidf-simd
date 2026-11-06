package tfidf

import (
	"hash/maphash"
	"math"
	"sync"
	"testing"
)

func almostEqual(a, b float64) bool {
	d := math.Abs(a - b)
	return d <= 1e-6 || d <= 1e-6*math.Max(math.Abs(a), math.Abs(b))
}

func norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func TestNew(t *testing.T) {
	if got := New(0).Dimensions(); got != DefaultDimensions {
		t.Fatalf("default dimensions = %d, want %d", got, DefaultDimensions)
	}
	if got := New(-1).Dimensions(); got != DefaultDimensions {
		t.Fatalf("negative dimensions = %d, want %d", got, DefaultDimensions)
	}
	if got := New(17).Dimensions(); got != 17 {
		t.Fatalf("dimensions = %d, want 17", got)
	}

	ix := New(64)
	if got := ix.Len(); got != 0 {
		t.Fatalf("initial Len = %d, want 0", got)
	}
	if got := ix.GetVector(0); got != nil {
		t.Fatalf("vector of missing document = %v, want nil", got)
	}
	if got := ix.GetVector(-1); got != nil {
		t.Fatalf("vector of negative index = %v, want nil", got)
	}
}

func TestAddDocumentAndRawVector(t *testing.T) {
	const dims = 8192
	ix := New(dims, WithNormalize(false))

	tokens := []string{"go", "go", "simd", "tfidf"}
	idx := ix.AddDocument(tokens)
	if idx != 0 {
		t.Fatalf("first document index = %d, want 0", idx)
	}
	if got := ix.Len(); got != 1 {
		t.Fatalf("Len = %d, want 1", got)
	}

	wantCounts := make(map[uint32]float64)
	for _, token := range tokens {
		wantCounts[ix.bucket(token)]++
	}

	vec := ix.GetVector(idx)
	if len(vec) != dims {
		t.Fatalf("vector length = %d, want %d", len(vec), dims)
	}

	for bucket, count := range wantCounts {
		idf := math.Log(2.0/(1.0+float64(ix.df[bucket]))) + 1
		want := count * idf
		if got := float64(vec[bucket]); !almostEqual(got, want) {
			t.Errorf("bucket %d: got %v, want %v", bucket, got, want)
		}
	}

	nonZero := 0
	for _, x := range vec {
		if x != 0 {
			nonZero++
		}
	}
	if nonZero != len(wantCounts) {
		t.Fatalf("non-zero entries = %d, want %d", nonZero, len(wantCounts))
	}

	if alias := ix.Vector(idx); len(alias) != dims {
		t.Fatalf("Vector alias length = %d, want %d", len(alias), dims)
	}
}

func TestBucketPowerOfTwo(t *testing.T) {
	for _, dimensions := range []int{1, 2, 16, 4096} {
		ix := New(dimensions)
		if !ix.powerOfTwo {
			t.Fatalf("dimensions %d: powerOfTwo = false", dimensions)
		}
		for _, token := range []string{"", "a", "go", "some longer token"} {
			h := maphash.String(ix.seed, token)
			want := uint32(h & uint64(dimensions-1))
			if got := ix.bucket(token); got != want {
				t.Errorf("dimensions %d, token %q: bucket = %d, want %d", dimensions, token, got, want)
			}
		}
	}
}

func TestBucketNonPowerOfTwo(t *testing.T) {
	for _, dimensions := range []int{3, 1000, 4095} {
		ix := New(dimensions)
		if ix.powerOfTwo {
			t.Fatalf("dimensions %d: powerOfTwo = true", dimensions)
		}
		for _, token := range []string{"", "a", "go", "some longer token"} {
			h := maphash.String(ix.seed, token)
			want := uint32(h % uint64(dimensions))
			if got := ix.bucket(token); got != want {
				t.Errorf("dimensions %d, token %q: bucket = %d, want %d", dimensions, token, got, want)
			}
		}
	}
}

func TestIDFChangesExistingVectors(t *testing.T) {
	ix := New(8192, WithNormalize(false))
	ix.AddDocument([]string{"background one"})
	ix.AddDocument([]string{"background two"})
	first := ix.AddDocument([]string{"go", "go", "simd"})
	before := ix.GetVector(first)

	second := ix.AddDocument([]string{"go"})
	if second != 3 {
		t.Fatalf("second document index = %d, want 3", second)
	}
	after := ix.GetVector(first)

	if ix.df[ix.bucket("go")] != 2 {
		t.Fatalf("document frequency of go = %d, want 2", ix.df[ix.bucket("go")])
	}

	// The shared term's IDF must decrease after the second document is added.
	goBucket := ix.bucket("go")
	if !(after[goBucket] < before[goBucket]) {
		t.Fatalf("weight of go did not decrease: before=%v after=%v", before[goBucket], after[goBucket])
	}

	want := 2 * (math.Log(5.0/3.0) + 1)
	if got := float64(after[goBucket]); !almostEqual(got, want) {
		t.Fatalf("weight of go = %v, want %v", got, want)
	}
}

func TestNormalization(t *testing.T) {
	ix := New(8192)
	idx := ix.AddDocument([]string{"alpha", "alpha", "beta", "gamma"})
	vec := ix.GetVector(idx)

	if got := norm(vec); math.Abs(got-1) > 1e-5 {
		t.Fatalf("L2 norm = %v, want 1", got)
	}

	empty := ix.AddDocument(nil)
	emptyVec := ix.GetVector(empty)
	if len(emptyVec) != 8192 {
		t.Fatalf("empty document vector length = %d, want 8192", len(emptyVec))
	}
	for i, x := range emptyVec {
		if x != 0 {
			t.Fatalf("empty document vector[%d] = %v, want 0", i, x)
		}
	}
}

func TestQueryVector(t *testing.T) {
	ix := New(4096, WithNormalize(false))

	// With no documents, IDF is log(1/1)+1 = 1.
	query := ix.QueryVector([]string{"x", "x", "y"})
	wantCounts := make(map[uint32]float64)
	for _, token := range []string{"x", "x", "y"} {
		wantCounts[ix.bucket(token)]++
	}
	for bucket, count := range wantCounts {
		if got := float64(query[bucket]); !almostEqual(got, count) {
			t.Errorf("bucket %d: got %v, want %v", bucket, got, count)
		}
	}

	ix.AddDocument([]string{"x"})
	query = ix.QueryVector([]string{"x"})
	xBucket := ix.bucket("x")
	if got := float64(query[xBucket]); !almostEqual(got, 1) {
		t.Fatalf("weight of x after one document = %v, want 1", got)
	}
	ix.AddDocument([]string{"x"})
	query = ix.QueryVector([]string{"x"})
	if got := float64(query[xBucket]); !almostEqual(got, math.Log(3.0/3.0)+1) {
		t.Fatalf("weight of x after two documents = %v, want 1", got)
	}
}

func TestSimilarity(t *testing.T) {
	ix := New(4096)
	a := ix.AddDocument([]string{"go", "simd", "vectors"})
	b := ix.AddDocument([]string{"go", "simd", "vectors"})
	c := ix.AddDocument([]string{"banana"})
	empty := ix.AddDocument(nil)

	if got := ix.Similarity(a, b); math.Abs(got-1) > 1e-5 {
		t.Fatalf("similarity of identical documents = %v, want 1", got)
	}
	if got := ix.Similarity(a, a); math.Abs(got-1) > 1e-5 {
		t.Fatalf("self similarity = %v, want 1", got)
	}
	if got := ix.Similarity(empty, a); got != 0 {
		t.Fatalf("similarity with empty document = %v, want 0", got)
	}
	if got := ix.Similarity(a, 999); got != 0 {
		t.Fatalf("similarity with invalid index = %v, want 0", got)
	}
	if got := ix.Similarity(a, c); got < 0 || got > 1 {
		t.Fatalf("similarity out of range: %v", got)
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"parallel", []float32{1, 2}, []float32{2, 4}, 1},
		{"opposite", []float32{1, 1}, []float32{-1, -1}, -1},
		{"zero", []float32{0, 0}, []float32{1, 1}, 0},
		{"empty", nil, []float32{1}, 0},
		{"different lengths", []float32{1, 2, 3}, []float32{1, 2}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CosineSimilarity(tt.a, tt.b); math.Abs(got-tt.want) > 1e-6 {
				t.Fatalf("CosineSimilarity() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDotProduct(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000}
	for _, n := range sizes {
		a := make([]float32, n)
		b := make([]float32, n)
		var want float64
		for i := 0; i < n; i++ {
			a[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
			b[i] = float32(math.Cos(float64(i)*0.19)) * 0.5
			want += float64(a[i]) * float64(b[i])
		}
		got := DotProduct(a, b)
		if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Errorf("length %d: DotProduct() = %v, want %v", n, got, want)
		}
	}

	if got := DotProduct([]float32{1, 2, 3}, []float32{4, 5}); got != 14 {
		t.Fatalf("DotProduct with different lengths = %v, want 14", got)
	}
}

func TestConcurrentUse(t *testing.T) {
	ix := New(2048)
	const writers = 8
	const perWriter = 50

	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				ix.AddDocument([]string{"writer", "doc", "shared"})
			}
		}(w)
	}
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				ix.GetVector(0)
				ix.QueryVector([]string{"shared"})
				ix.Similarity(0, 1)
			}
		}()
	}
	wg.Wait()

	if got := ix.Len(); got != writers*perWriter {
		t.Fatalf("Len = %d, want %d", got, writers*perWriter)
	}
}
