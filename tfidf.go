// Package tfidf provides a compact TF-IDF index that maps tokens into
// fixed-width vectors using hash/maphash.
//
// Tokens are hashed into a configurable number of dimensions. This is known as
// feature hashing (or the hashing trick): no vocabulary has to be stored, and
// every document vector has the same length, which makes the vectors suitable
// for SIMD dot products and cosine similarity.
//
// The package has three implementations of its inner dot product:
//
//   - a portable pure-Go implementation, used by default;
//   - on amd64 with GOEXPERIMENT=simd, an implementation using
//     simd/archsimd with runtime AVX, AVX2, and AVX-512 dispatch;
//   - on other architectures with GOEXPERIMENT=simd, an implementation using
//     the portable simd package.
//
// The TF-IDF weights are computed identically by all implementations. L2
// normalization uses the SIMD dot product, so normalized vectors can differ in
// their last bits between implementations.
package tfidf

import (
	"hash/maphash"
	"math"
	"sync"
)

// DefaultDimensions is used when New is called with a non-positive dimension.
const DefaultDimensions = 4096

// Option configures an Index.
type Option func(*Index)

// WithNormalize enables or disables L2 normalization of returned vectors.
// Normalization is enabled by default.
func WithNormalize(enabled bool) Option {
	return func(ix *Index) {
		ix.normalize = enabled
	}
}

// Index is a concurrency-safe TF-IDF index.
//
// The zero value is not usable; create an Index with New.
type Index struct {
	mu         sync.RWMutex
	dimensions int
	seed       maphash.Seed
	normalize  bool

	powerOfTwo bool
	bucketMask uint64

	df []uint32 // number of documents containing each hashed bucket

	// mainVector is the flat storage for the unique hashed bucket IDs of all
	// documents. Bucket IDs are uint32, which is also the index type used by
	// df. A document owns a view into this slice (see docInfo.docSlice).
	mainVector []uint32

	// mainFreq is the flat storage for the term frequencies of all documents,
	// parallel to mainVector. A document owns a view into this slice (see
	// docInfo.termFreq). Keeping both values in flat arenas means adding a
	// document does not allocate one slice per document.
	mainFreq []uint32

	// hist counts term frequencies with a dense, generation-stamped table. It
	// is nil when dimensions is too large for the dense table; documents are
	// then sorted and run-length counted instead.
	hist *histogram

	// mainDocuments holds the per-document metadata. The document index
	// returned by AddDocument is an index into this slice.
	mainDocuments []docInfo
}

// docInfo describes one indexed document.
//
// docSlice is a view into Index.mainVector containing the document's unique
// hashed buckets, in first-occurrence order (or sorted order for the large
// dimensions fallback). termFreq is a view into Index.mainFreq and is parallel
// to docSlice: termFreq[i] is the number of times docSlice[i] occurred in the
// original token stream. Collisions are therefore merged in the same way as a
// map-based term counter.
type docInfo struct {
	docSlice []uint32
	termFreq []uint32
}

// New creates an Index with the given number of dimensions.
//
// A non-positive dimension uses DefaultDimensions. Dimensions must fit in a
// uint32.
func New(dimensions int, opts ...Option) *Index {
	if dimensions <= 0 {
		dimensions = DefaultDimensions
	}
	if uint64(dimensions) > math.MaxUint32 {
		panic("tfidf: dimensions do not fit in uint32")
	}

	ix := &Index{
		dimensions: dimensions,
		seed:       maphash.MakeSeed(),
		normalize:  true,
		df:         make([]uint32, dimensions),
	}
	if dimensions&(dimensions-1) == 0 {
		ix.powerOfTwo = true
		ix.bucketMask = uint64(dimensions - 1)
	}
	if dimensions <= maxDenseHistogramDimensions {
		ix.hist = newHistogram(dimensions)
	}
	for _, opt := range opts {
		if opt != nil {
			opt(ix)
		}
	}
	return ix
}

// Dimensions returns the length of every vector produced by the index.
func (ix *Index) Dimensions() int {
	return ix.dimensions
}

// Len returns the number of indexed documents.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.mainDocuments)
}

// AddDocument adds tokens as a new document and returns its index.
//
// The input is treated as an already-tokenized document. Token frequency is
// preserved; repeated tokens increase the term frequency. An empty document is
// valid and produces an all-zero vector.
//
// AddDocument may be called concurrently with other methods.
func (ix *Index) AddDocument(tokens []string) int {
	buckets := make([]uint32, len(tokens))
	for i, token := range tokens {
		buckets[i] = ix.bucket(token)
	}

	ix.mu.Lock()
	defer ix.mu.Unlock()

	unique, freq := ix.termFreq(buckets)

	start := len(ix.mainVector)
	ix.mainVector = append(ix.mainVector, unique...)
	freqStart := len(ix.mainFreq)
	ix.mainFreq = append(ix.mainFreq, freq...)

	for _, b := range unique {
		ix.df[b]++
	}
	ix.mainDocuments = append(ix.mainDocuments, docInfo{
		docSlice: ix.mainVector[start : start+len(unique)],
		termFreq: ix.mainFreq[freqStart : freqStart+len(freq)],
	})
	return len(ix.mainDocuments) - 1
}

// GetVector returns the TF-IDF vector for document index n.
//
// It returns nil when n is out of range. Vectors are L2-normalized by default;
// use WithNormalize(false) to get raw TF-IDF weights.
func (ix *Index) GetVector(n int) []float32 {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if n < 0 || n >= len(ix.mainDocuments) {
		return nil
	}
	d := ix.mainDocuments[n]
	return ix.vector(d.docSlice, d.termFreq)
}

// Vector is an alias for GetVector.
func (ix *Index) Vector(n int) []float32 {
	return ix.GetVector(n)
}

// QueryVector returns a TF-IDF vector for tokens without adding a document.
//
// The current document frequencies are used for the IDF part. As with
// document vectors, the result is L2-normalized by default.
func (ix *Index) QueryVector(tokens []string) []float32 {
	buckets := make([]uint32, len(tokens))
	for i, token := range tokens {
		buckets[i] = ix.bucket(token)
	}

	// ix.hist is owned by AddDocument under the write lock, so queries use a
	// pooled histogram instead of touching it.
	if ix.hist != nil {
		h := queryHistogramPool.Get().(*histogram)
		defer queryHistogramPool.Put(h)
		h.resize(ix.dimensions)
		unique, freq := h.count(buckets)

		ix.mu.RLock()
		defer ix.mu.RUnlock()
		return ix.vector(unique, freq)
	}

	unique, freq := sortTermFreq(buckets)
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.vector(unique, freq)
}

// Similarity returns the cosine similarity between documents i and j.
// It returns zero if either index is out of range.
func (ix *Index) Similarity(i, j int) float64 {
	ix.mu.RLock()
	if i < 0 || i >= len(ix.mainDocuments) || j < 0 || j >= len(ix.mainDocuments) {
		ix.mu.RUnlock()
		return 0
	}
	di := ix.mainDocuments[i]
	dj := ix.mainDocuments[j]
	a := ix.vector(di.docSlice, di.termFreq)
	b := ix.vector(dj.docSlice, dj.termFreq)
	ix.mu.RUnlock()

	return CosineSimilarity(a, b)
}

// vector builds a TF-IDF vector from unique buckets and their frequencies.
// The caller must hold at least a read lock.
func (ix *Index) vector(buckets, termFreq []uint32) []float32 {
	vec := make([]float32, ix.dimensions)
	n := float64(len(ix.mainDocuments))
	for i, b := range buckets {
		df := float64(ix.df[b])
		idf := math.Log((1+n)/(1+df)) + 1
		vec[b] = float32(float64(termFreq[i]) * idf)
	}
	if ix.normalize {
		normalize(vec)
	}
	return vec
}

func (ix *Index) bucket(token string) uint32 {
	h := maphash.String(ix.seed, token)
	if ix.powerOfTwo {
		return uint32(h & ix.bucketMask)
	}
	return uint32(h % uint64(ix.dimensions))
}

// normalize L2-normalizes v in place. The squared norm is computed with the
// SIMD dot product, and the scaling pass is vectorized as well.
func normalize(v []float32) {
	sum := dotProduct(v, v)
	if sum == 0 {
		return
	}
	scaleVector(v, float32(1/math.Sqrt(sum)))
}

// DotProduct returns the dot product of a and b, using the shorter length when
// they differ.
//
// It is the SIMD-accelerated primitive used by CosineSimilarity.
func DotProduct(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotProduct(a[:n], b[:n])
}

// CosineSimilarity returns the cosine similarity of a and b, using the shorter
// length when they differ. It returns zero if either vector has zero norm.
func CosineSimilarity(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	a = a[:n]
	b = b[:n]

	dot := dotProduct(a, b)
	na := dotProduct(a, a)
	nb := dotProduct(b, b)
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}
