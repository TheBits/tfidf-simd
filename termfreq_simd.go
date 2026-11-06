//go:build goexperiment.simd

package tfidf

import "simd"

// countEqual returns the number of elements in v equal to x.
//
// It is used by the term-frequency kernel. The vector width comes from the
// simd type itself, so the same code works with every supported architecture
// and vector width. The tail is handled scalar because LoadUint32sPart zero
// fills inactive lanes, which would falsely match x == 0.
func countEqual(v []uint32, x uint32) int {
	var acc simd.Int32s
	ones := simd.BroadcastInt32s(1)
	want := simd.BroadcastUint32s(x)
	lanes := acc.Len()

	i := 0
	for ; i+lanes <= len(v); i += lanes {
		eq := simd.LoadUint32s(v[i:]).Equal(want)
		// Mask true lanes are represented by -1; AND with 1 turns them into
		// a vector of 0/1 counters that can be accumulated.
		acc = acc.Add(eq.ToInt32s().And(ones))
	}

	var small [16]int32
	buf := small[:]
	if n := acc.Len(); n > len(buf) {
		buf = make([]int32, n)
	}
	n := acc.StorePart(buf)

	total := 0
	for _, c := range buf[:n] {
		total += int(c)
	}
	for ; i < len(v); i++ {
		if v[i] == x {
			total++
		}
	}
	return total
}
