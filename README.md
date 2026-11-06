# tfidf-simd

A pure-Go TF-IDF implementation that uses SIMD instructions. It works on ARM
Neon and amd64 SSE/AVX (including AVX-512 when available). On amd64 it uses
`simd/archsimd` directly and dispatches at runtime to AVX, AVX2/FMA, or AVX-512
code; other architectures use Go's portable `simd` package. No cgo or assembly
is required.

Tokens are hashed into a fixed number of dimensions using `hash/maphash`.
Documents therefore never require a stored vocabulary, and every vector has
the same length, which makes the vectors suitable for SIMD operations.

## Requirements

- Go 1.27 or newer.
- SIMD acceleration is optional. Build with `GOEXPERIMENT=simd` to use SIMD.
  Without the experiment, a portable pure-Go implementation is used.
  - On amd64, `simd/archsimd` is used with runtime dispatch to AVX,
    AVX2/FMA, or AVX-512. CPUs without AVX fall back to the portable scalar
    implementation.
  - On ARM Neon and other supported architectures, the portable `simd`
    package is used.

```sh
go test ./...
GOEXPERIMENT=simd go test ./...
```

## Benchmarks

Run the benchmarks against the portable implementation and the SIMD
implementation to compare them on your machine:

```sh
go test -bench=. -benchmem ./...
GOEXPERIMENT=simd go test -bench=. -benchmem ./...
```

Individual benchmark groups can be selected with `-bench`, for example:

```sh
GOEXPERIMENT=simd go test -bench='DotProduct|Cosine' -benchmem ./...
```

## SIMD implementation notes

The kernels follow the inner-product and bridge-type patterns described in the
[Go SIMD blog post](https://go.dev/blog/simd-experiment):

- Loops use the vector's own `Len()` rather than a hard-coded vector width.
- Short tails use `Load*Part`, which zero-fills inactive lanes.
- amd64 dispatches with a `Float32s.ToArch()` type switch. The compiler
  specializes the function and removes the switch from the hot path.
- AVX/AVX2 paths use FMA when the CPU supports it and plain multiply/add
  otherwise. AVX-512 uses 512-bit FMA vectors.
- L2 normalization reuses the SIMD dot product for the squared norm and applies
  the scale factor with a vectorized multiply pass.
- The portable vector sum uses a fixed-size stack buffer where possible, so
  the dot-product path performs no allocations.
- Term frequencies use a dense, generation-stamped histogram indexed by
  bucket. Counting is O(n) per document with no per-term probing and no
  allocation, and the counts are stored in a flat arena shared by all
  documents. Indexes with more than 65536 dimensions fall back to sorting and
  run-length counting to bound memory use; SIMD compare/scatter kernels do not
  pay off for sparse hashed buckets, so this path is scalar by design.

## Usage

```go
package main

import (
	"fmt"

	tfidf "github.com/thebits/tfidf-simd"
)

func main() {
	ix := tfidf.New(4096) // vector dimensions

	a := ix.AddDocument([]string{"go", "simd", "vectors"})
	b := ix.AddDocument([]string{"go", "vectors", "fast"})

	fmt.Println(ix.GetVector(a))       // []float32 of length 4096
	fmt.Println(ix.Vector(b))          // alias for GetVector
	fmt.Println(ix.Similarity(a, b))   // cosine similarity
	fmt.Println(tfidf.DotProduct(
		ix.GetVector(a),
		ix.GetVector(b),
	)) // SIMD dot product
}
```

Tokens should be produced by your own tokenizer (lowercasing, stemming,
removing stop words, etc.).

## API

```go
func New(dimensions int, opts ...Option) *Index
func WithNormalize(enabled bool) Option

func (ix *Index) AddDocument(tokens []string) int
func (ix *Index) GetVector(n int) []float32
func (ix *Index) Vector(n int) []float32
func (ix *Index) QueryVector(tokens []string) []float32
func (ix *Index) Similarity(i, j int) float64
func (ix *Index) Dimensions() int
func (ix *Index) Len() int

func DotProduct(a, b []float32) float64
func CosineSimilarity(a, b []float32) float64
```

`GetVector` and `Vector` return `nil` for an out-of-range index.

## Weights

For a term with count `tf` in document `d`:

```text
idf = ln((1 + N) / (1 + df)) + 1
weight = tf * idf
```

where `N` is the number of documents and `df` is the number of documents
containing the term's hash bucket.

Vectors are L2-normalized by default so that a SIMD dot product is already the
cosine similarity. Use `WithNormalize(false)` to get raw TF-IDF weights:

```go
ix := tfidf.New(8192, tfidf.WithNormalize(false))
```

Empty documents are valid and produce all-zero vectors.

## Storage

Hashed bucket IDs are kept in one flat `mainVector` slice. Each document has a
`docInfo` entry containing a view into that slice (`docSlice`) and a parallel
list of term frequencies (`termFreq`). Repeated tokens and hash collisions are
merged into one entry, so each document stores only its unique buckets.

## Hashing and persistence

The hash seed comes from `hash/maphash.MakeSeed`, so bucket assignments differ
between process runs. The seed cannot be serialized. This is intentional: the
index is designed as an in-memory structure. Collisions are handled by
combining term frequencies in the same bucket.

When the number of dimensions is a power of two (including the default 4096),
bucket selection uses a bitmask (`h & (n-1)`). Other dimension counts use
`h % n`.

## Concurrency

`Index` is safe for concurrent use. `AddDocument` updates the document
frequencies under a write lock; readers use a read lock. Because adding a
document changes IDF values, previously returned vectors are snapshots and
existing vectors may have different weights after new documents are added.
