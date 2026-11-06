package tfidf

import (
	"strconv"
	"testing"
)

func termFreqInput(ix *Index, n int) []uint32 {
	src := make([]uint32, n)
	for i := range src {
		src[i] = ix.bucket("token-" + strconv.Itoa((i*37)%11))
	}
	return src
}

func termFreqCounts(src []uint32) map[uint32]uint32 {
	want := make(map[uint32]uint32, len(src))
	for _, b := range src {
		want[b]++
	}
	return want
}

func checkTermFreq(t *testing.T, unique, freq []uint32, want map[uint32]uint32) {
	t.Helper()
	if len(unique) != len(want) {
		t.Fatalf("len(unique) = %d, want %d", len(unique), len(want))
	}
	if len(freq) != len(want) {
		t.Fatalf("len(freq) = %d, want %d", len(freq), len(want))
	}

	got := make(map[uint32]uint32, len(want))
	for i, b := range unique {
		if _, seen := got[b]; seen {
			t.Fatalf("bucket %d appears more than once in unique", b)
		}
		got[b] = freq[i]
	}
	for b, count := range want {
		if got[b] != count {
			t.Errorf("bucket %d: count = %d, want %d", b, got[b], count)
		}
	}
}

func TestTermFreq(t *testing.T) {
	ix := New(4096)

	sizes := []int{0, 1, 2, 3, 63, 64, 65, 127, 257}
	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := termFreqInput(ix, n)
			want := termFreqCounts(src)
			unique, freq := ix.termFreq(src)
			checkTermFreq(t, unique, freq, want)
		})
	}
}

func TestTermFreqSortFallback(t *testing.T) {
	ix := New(4096)

	sizes := []int{0, 1, 2, 3, 63, 64, 65, 127, 257}
	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := termFreqInput(ix, n)
			want := termFreqCounts(src)
			unique, freq := sortTermFreq(src)
			checkTermFreq(t, unique, freq, want)
		})
	}
}

func TestTermFreqFallsBackForLargeDimensions(t *testing.T) {
	ix := New(maxDenseHistogramDimensions + 1)
	if ix.hist != nil {
		t.Fatal("dense histogram allocated for large dimensions")
	}

	src := termFreqInput(ix, 100)
	want := termFreqCounts(src)
	unique, freq := ix.termFreq(src)
	checkTermFreq(t, unique, freq, want)
}

func TestTermFreqStrategiesAgree(t *testing.T) {
	ix := New(4096)

	for n := 0; n <= 300; n++ {
		src := termFreqInput(ix, n)
		want := termFreqCounts(src)

		unique, freq := ix.termFreq(src)
		checkTermFreq(t, unique, freq, want)

		unique, freq = sortTermFreq(append([]uint32(nil), src...))
		checkTermFreq(t, unique, freq, want)
	}
}

func TestMainVectorAndDocumentInfo(t *testing.T) {
	ix := New(4096)

	docs := [][]string{
		{"go", "go", "simd", "tfidf"},
		nil,
		{"a", "b", "a", "c", "b", "a"},
		{"go", "simd", "vectors", "go", "fast"},
	}
	for _, tokens := range docs {
		ix.AddDocument(tokens)
	}

	if got, want := len(ix.mainDocuments), len(docs); got != want {
		t.Fatalf("len(mainDocuments) = %d, want %d", got, want)
	}

	totalUnique := 0
	for doc, tokens := range docs {
		want := make(map[uint32]uint32, len(tokens))
		for _, token := range tokens {
			want[ix.bucket(token)]++
		}
		totalUnique += len(want)

		d := ix.mainDocuments[doc]
		if got, want := len(d.docSlice), len(want); got != want {
			t.Fatalf("document %d: len(docSlice) = %d, want %d", doc, got, want)
		}
		if got, want := len(d.termFreq), len(want); got != want {
			t.Fatalf("document %d: len(termFreq) = %d, want %d", doc, got, want)
		}

		got := make(map[uint32]uint32, len(d.docSlice))
		for i, b := range d.docSlice {
			got[b] = d.termFreq[i]
		}
		for b, count := range want {
			if got[b] != count {
				t.Errorf("document %d, bucket %d: count = %d, want %d", doc, b, got[b], count)
			}
		}
	}

	// mainVector only retains the live, unique bucket entries.
	if got, want := len(ix.mainVector), totalUnique; got != want {
		t.Fatalf("len(mainVector) = %d, want %d", got, want)
	}
}

func TestDocumentsRemainValidAfterMainVectorGrowth(t *testing.T) {
	ix := New(8192)
	const docCount = 512

	want := make([]map[uint32]uint32, docCount)
	for i := 0; i < docCount; i++ {
		tokens := make([]string, 12)
		for j := range tokens {
			tokens[j] = "d" + strconv.Itoa(i) + "-t" + strconv.Itoa(j%5)
		}
		ix.AddDocument(tokens)

		counts := make(map[uint32]uint32, len(tokens))
		for _, token := range tokens {
			counts[ix.bucket(token)]++
		}
		want[i] = counts
	}

	for doc, counts := range want {
		d := ix.mainDocuments[doc]
		got := make(map[uint32]uint32, len(d.docSlice))
		for i, b := range d.docSlice {
			got[b] = d.termFreq[i]
		}
		for b, count := range counts {
			if got[b] != count {
				t.Fatalf("document %d, bucket %d: count = %d, want %d", doc, b, got[b], count)
			}
		}
	}
}

func TestCountEqual(t *testing.T) {
	for n := 0; n <= 40; n++ {
		v := make([]uint32, n)
		for i := range v {
			v[i] = uint32(i % 5)
		}
		for x := uint32(0); x < 7; x++ {
			want := 0
			for _, y := range v {
				if y == x {
					want++
				}
			}
			if got := countEqual(v, x); got != want {
				t.Fatalf("countEqual(len=%d, x=%d) = %d, want %d", n, x, got, want)
			}
		}
	}
}
