package tfidf

import (
	"math"
	"strconv"
	"testing"
)

func TestWithNormalizeOption(t *testing.T) {
	raw := New(4096, WithNormalize(false))
	rawIdx := raw.AddDocument([]string{"alpha", "alpha", "beta"})
	rawVec := raw.GetVector(rawIdx)
	if got := norm(rawVec); got <= 1 {
		t.Fatalf("raw vector norm = %v, want > 1", got)
	}

	normalized := New(4096)
	normIdx := normalized.AddDocument([]string{"alpha", "alpha", "beta"})
	normVec := normalized.GetVector(normIdx)
	if got := norm(normVec); math.Abs(got-1) > 1e-5 {
		t.Fatalf("normalized vector norm = %v, want 1", got)
	}
}

func TestVectorIsCopy(t *testing.T) {
	ix := New(2048)
	idx := ix.AddDocument([]string{"go", "simd", "go"})
	first := ix.GetVector(idx)
	want := append([]float32(nil), first...)

	for i := range first {
		first[i] = 12345
	}

	second := ix.GetVector(idx)
	if len(second) != len(want) {
		t.Fatalf("vector length = %d, want %d", len(second), len(want))
	}
	for i := range want {
		if second[i] != want[i] {
			t.Fatalf("vector changed after mutating returned slice at %d: got %v, want %v", i, second[i], want[i])
		}
	}
}

func TestHashCollisionsAreCombined(t *testing.T) {
	ix := New(1, WithNormalize(false))

	first := ix.AddDocument([]string{"a", "b", "a"})
	if got := ix.df[0]; got != 1 {
		t.Fatalf("document frequency = %d, want 1", got)
	}
	if got := float64(ix.GetVector(first)[0]); !almostEqual(got, 3) {
		t.Fatalf("collided term frequency = %v, want 3", got)
	}

	second := ix.AddDocument([]string{"b"})
	if got := ix.df[0]; got != 2 {
		t.Fatalf("document frequency after second document = %d, want 2", got)
	}
	if got := float64(ix.GetVector(first)[0]); !almostEqual(got, 3) {
		t.Fatalf("weight after idf change = %v, want 3", got)
	}
	if got := float64(ix.GetVector(second)[0]); !almostEqual(got, 1) {
		t.Fatalf("second document weight = %v, want 1", got)
	}
}

func TestQueryVectorDoesNotAddDocument(t *testing.T) {
	ix := New(1024)
	ix.AddDocument([]string{"go", "simd"})
	before := ix.Len()

	query := ix.QueryVector([]string{"go", "vectors"})
	if got := ix.Len(); got != before {
		t.Fatalf("Len changed from %d to %d", before, got)
	}
	if got := norm(query); math.Abs(got-1) > 1e-5 {
		t.Fatalf("normalized query norm = %v, want 1", got)
	}
}

func TestEmptyQueryVector(t *testing.T) {
	ix := New(1024)
	query := ix.QueryVector(nil)
	if len(query) != 1024 {
		t.Fatalf("query length = %d, want 1024", len(query))
	}
	for i, x := range query {
		if x != 0 {
			t.Fatalf("query[%d] = %v, want 0", i, x)
		}
	}
}

func TestSimilarityIsSymmetric(t *testing.T) {
	ix := New(4096)
	a := ix.AddDocument([]string{"go", "simd", "vectors"})
	b := ix.AddDocument([]string{"go", "fast", "vectors"})
	c := ix.AddDocument([]string{"completely", "different"})

	for _, pair := range [][2]int{{a, b}, {a, c}, {b, c}} {
		ab := ix.Similarity(pair[0], pair[1])
		ba := ix.Similarity(pair[1], pair[0])
		if math.Abs(ab-ba) > 1e-5 {
			t.Fatalf("Similarity(%d, %d) = %v, Similarity(%d, %d) = %v", pair[0], pair[1], ab, pair[1], pair[0], ba)
		}
	}
}

func BenchmarkAddDocument(b *testing.B) {
	for _, tokenCount := range []int{8, 64, 256} {
		b.Run("tokens="+strconv.Itoa(tokenCount), func(b *testing.B) {
			tokens := benchmarkTokens(tokenCount)
			ix := New(4096)
			b.ReportAllocs()

			for b.Loop() {
				// Keep the index bounded while benchmarking the add path.
				if len(ix.mainDocuments) >= 4096 {
					ix = New(4096)
				}
				benchSinkInt = ix.AddDocument(tokens)
			}
		})
	}
}

func BenchmarkGetVector(b *testing.B) {
	for _, dimensions := range []int{1024, 4096, 16384} {
		b.Run("dims="+strconv.Itoa(dimensions), func(b *testing.B) {
			ix := benchmarkIndex(dimensions, 256, 64)
			b.ReportAllocs()

			for b.Loop() {
				benchSinkVector = ix.GetVector(0)
			}
		})
	}
}

func BenchmarkQueryVector(b *testing.B) {
	ix := benchmarkIndex(4096, 256, 64)
	query := benchmarkTokens(32)
	b.ReportAllocs()

	for b.Loop() {
		benchSinkVector = ix.QueryVector(query)
	}
}

func BenchmarkSimilarity(b *testing.B) {
	ix := benchmarkIndex(4096, 512, 64)
	b.ReportAllocs()

	for b.Loop() {
		benchSinkFloat = ix.Similarity(0, 1)
	}
}

func BenchmarkBucket(b *testing.B) {
	for _, dimensions := range []int{4095, 4096} {
		b.Run("dims="+strconv.Itoa(dimensions), func(b *testing.B) {
			ix := New(dimensions)
			b.ReportAllocs()

			for b.Loop() {
				benchSinkInt = int(ix.bucket("token-17"))
			}
		})
	}
}

func BenchmarkNormalize(b *testing.B) {
	for _, n := range []int{128, 1024, 4096, 16384} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			v := benchmarkVector(n, 0.37)
			b.SetBytes(int64(n * 4))
			b.ReportAllocs()

			for b.Loop() {
				normalize(v)
			}
		})
	}
}

func BenchmarkDotProduct(b *testing.B) {
	for _, n := range []int{128, 1024, 4096, 16384} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			a := benchmarkVector(n, 0.37)
			c := benchmarkVector(n, 0.19)
			b.SetBytes(int64(n * 4))
			b.ReportAllocs()

			for b.Loop() {
				benchSinkFloat = DotProduct(a, c)
			}
		})
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	for _, n := range []int{128, 1024, 4096} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			a := benchmarkVector(n, 0.37)
			c := benchmarkVector(n, 0.19)
			b.SetBytes(int64(n * 4))
			b.ReportAllocs()

			for b.Loop() {
				benchSinkFloat = CosineSimilarity(a, c)
			}
		})
	}
}

func BenchmarkDotProductParallel(b *testing.B) {
	a := benchmarkVector(4096, 0.37)
	c := benchmarkVector(4096, 0.19)
	b.SetBytes(int64(len(a) * 4))
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			benchSinkFloat = DotProduct(a, c)
		}
	})
}

var (
	benchSinkFloat  float64
	benchSinkVector []float32
	benchSinkInt    int
)

func benchmarkTokens(n int) []string {
	tokens := make([]string, n)
	for i := range tokens {
		tokens[i] = "token-" + strconv.Itoa(i%32)
	}
	return tokens
}

func benchmarkIndex(dimensions, docs, tokensPerDoc int) *Index {
	ix := New(dimensions)
	for i := 0; i < docs; i++ {
		tokens := make([]string, tokensPerDoc)
		for j := range tokens {
			// Mix document and position so documents share some terms but are
			// not identical.
			tokens[j] = "term-" + strconv.Itoa((i*17+j*31)%256)
		}
		ix.AddDocument(tokens)
	}
	return ix
}

func benchmarkVector(n int, phase float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(i)*phase)) * 0.5
	}
	return v
}
