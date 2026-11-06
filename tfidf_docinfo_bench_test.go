package tfidf

import (
	"strconv"
	"testing"
)

// BenchmarkTermFreq compares the dense, generation-stamped histogram with the
// sort fallback used for very large dimensions.
func BenchmarkTermFreq(b *testing.B) {
	cases := []struct {
		name    string
		tokens  int
		buckets int // zero means every token has a distinct bucket
	}{
		{"tokens=8/unique", 8, 0},
		{"tokens=64/unique", 64, 0},
		{"tokens=65/unique", 65, 0},
		{"tokens=256/unique", 256, 0},
		{"tokens=1024/unique", 1024, 0},
		{"tokens=8/repeated", 8, 2},
		{"tokens=64/repeated", 64, 8},
		{"tokens=65/repeated", 65, 8},
		{"tokens=256/repeated", 256, 8},
		{"tokens=1024/repeated", 1024, 32},
	}

	ix := New(4096)
	for _, tc := range cases {
		template := make([]uint32, tc.tokens)
		for i := range template {
			if tc.buckets == 0 {
				template[i] = uint32(i)
			} else {
				template[i] = uint32(i % tc.buckets)
			}
		}
		buf := make([]uint32, len(template))

		b.Run(tc.name+"/dense", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				copy(buf, template)
				unique, freq := ix.termFreq(buf)
				benchSinkInt = len(unique) + len(freq)
			}
		})
		b.Run(tc.name+"/sort", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				copy(buf, template)
				unique, freq := sortTermFreq(buf)
				benchSinkInt = len(unique) + len(freq)
			}
		})
	}
}

func BenchmarkCountEqual(b *testing.B) {
	for _, n := range []int{8, 64, 256, 1024} {
		v := make([]uint32, n)
		for i := range v {
			v[i] = uint32(i % 32)
		}

		for _, tc := range []struct {
			name string
			x    uint32
		}{
			{"hit", 7},
			{"miss", 1000},
		} {
			b.Run("len="+strconv.Itoa(n)+"/"+tc.name, func(b *testing.B) {
				b.SetBytes(int64(len(v) * 4))
				b.ReportAllocs()

				for b.Loop() {
					benchSinkInt = countEqual(v, tc.x)
				}
			})
		}
	}
}

// BenchmarkAddDocumentUnique complements BenchmarkAddDocument, which uses a
// small repeating token set. This version exercises the unique-bucket path.
func BenchmarkAddDocumentUnique(b *testing.B) {
	for _, tokenCount := range []int{8, 64, 256} {
		b.Run("tokens="+strconv.Itoa(tokenCount), func(b *testing.B) {
			tokens := make([]string, tokenCount)
			for i := range tokens {
				tokens[i] = "unique-token-" + strconv.Itoa(i)
			}

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
