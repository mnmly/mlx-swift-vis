// Shared GPU helpers used across DR methods.

import Foundation
import MLX

/// Vectorized binary search on GPU (equivalent to `np.searchsorted`, side="left").
///
/// - Parameters:
///   - sortedArray: 1-D sorted array to search within.
///   - values: 1-D query values.
/// - Returns: Insertion indices, shape `(values.count,)`, dtype int32.
func searchSorted(_ sortedArray: MLXArray, _ values: MLXArray) -> MLXArray {
    let n = sortedArray.dim(0)
    let m = values.dim(0)
    var lo = MLXArray.zeros([m], dtype: .int32)
    var hi = MLXArray.full([m], values: MLXArray(Int32(n))).asType(.int32)

    let iters = Int(ceil(log2(Double(max(n, 2))))) + 1
    for _ in 0..<iters {
        let mid = floorDivide(lo + hi, MLXArray(Int32(2)))
        let midClamped = minimum(mid, n - 1)
        let goRight = sortedArray[midClamped] .< values
        lo = MLX.where(goRight, mid + 1, lo)
        hi = MLX.where(goRight, hi, mid)
    }
    return lo
}

/// A row-index column vector `[start, start+1, ..., end-1]` of shape `(end-start, 1)`.
@inline(__always)
func arangeColumn(_ start: Int, _ end: Int) -> MLXArray {
    MLXArray(Int32(start) ..< Int32(end)).expandedDimensions(axis: 1)
}

/// A row vector `[0, 1, ..., n-1]` of shape `(1, n)`.
@inline(__always)
func arangeRow(_ n: Int) -> MLXArray {
    MLXArray(Int32(0) ..< Int32(n)).expandedDimensions(axis: 0)
}
