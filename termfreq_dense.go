package tfidf

import (
	"slices"
	"sync"
)

// maxDenseHistogramDimensions bounds the memory used by the dense histogram.
// Each dimension costs 8 bytes, so 1<<16 dimensions use 512 KiB. Indexes with
// more dimensions fall back to sorting, which needs no per-dimension storage.
const maxDenseHistogramDimensions = 1 << 16

// histogram counts term frequencies with a dense table indexed by bucket.
//
// Each slot packs a generation stamp in the low 32 bits and the count in the
// high 32 bits. A document only reinitializes the slots it actually touches;
// the stamp makes stale counts from previous documents invisible without
// clearing the whole table. This keeps counting O(n) with one cache line touch
// per token and no allocation per document.
type histogram struct {
	state   []uint64 // low 32 bits: epoch, high 32 bits: count
	epoch   uint32
	touched []uint32
	freq    []uint32
}

// queryHistogramPool provides scratch histograms for QueryVector, which cannot
// use Index.hist because AddDocument mutates it under the write lock.
var queryHistogramPool = sync.Pool{
	New: func() any { return new(histogram) },
}

func newHistogram(dimensions int) *histogram {
	return &histogram{state: make([]uint64, dimensions)}
}

// resize grows the table to dimensions, resetting the generation stamps when
// the backing storage changes.
func (h *histogram) resize(dimensions int) {
	if len(h.state) >= dimensions {
		return
	}
	h.state = make([]uint64, dimensions)
	h.epoch = 0
}

// count returns the unique buckets in src and their frequencies, in
// first-occurrence order. The returned slices are valid until the next call.
func (h *histogram) count(src []uint32) (unique, freq []uint32) {
	h.epoch++
	if h.epoch == 0 {
		// Stamp wrap: every slot is stale again.
		clear(h.state)
		h.epoch = 1
	}

	h.touched = h.touched[:0]
	for _, b := range src {
		if uint32(h.state[b]) != h.epoch {
			h.state[b] = uint64(h.epoch) | 1<<32
			h.touched = append(h.touched, b)
			continue
		}
		h.state[b] += 1 << 32
	}

	h.freq = h.freq[:0]
	for _, b := range h.touched {
		h.freq = append(h.freq, uint32(h.state[b]>>32))
	}
	return h.touched, h.freq
}

// sortTermFreq sorts src in place and counts equal runs. It is the fallback
// used when dimensions is too large for the dense histogram.
func sortTermFreq(src []uint32) (unique, freq []uint32) {
	slices.Sort(src)

	freq = make([]uint32, 0, len(src))
	n := 0
	for i := 0; i < len(src); {
		j := i + 1
		for j < len(src) && src[j] == src[i] {
			j++
		}
		src[n] = src[i]
		freq = append(freq, uint32(j-i))
		n++
		i = j
	}
	return src[:n], freq
}

// termFreq returns the unique buckets of src and their frequencies.
//
// It uses the dense histogram when available, which is O(n) and independent of
// document size, and does not modify src. The sort fallback reorders src in
// place.
func (ix *Index) termFreq(src []uint32) (unique, freq []uint32) {
	if ix.hist != nil {
		return ix.hist.count(src)
	}
	return sortTermFreq(src)
}
