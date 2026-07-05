// Symmetric sparse affinity graphs from a KNN graph.
//
// Two deterministic building blocks shared across the methods (no scatter-add, so
// exactly reproducible — see AffinityGraphTests):
//   - `reverseEdgeValues`: the searchSorted reverse-edge lookup that every
//     symmetrization uses (UMAP fuzzy union, t-SNE / DREAMS P-matrix).
//   - `buildPMatrix`: the perplexity-calibrated, symmetrized t-SNE P-matrix shared
//     verbatim by t-SNE and DREAMS.

import Foundation
import MLX

/// For a directed sparse edge set (`rows[i] -> cols[i]` with weight `vals[i]`),
/// return, for each edge, the weight on the reverse edge (`cols[i] -> rows[i]`),
/// or 0 when that reverse edge is absent.
///
/// This is the GPU symmetrization primitive: build int64 keys for forward and
/// reverse edges, sort the forward keys, and binary-search the reverse keys against
/// them. Used to form `A + Aᵀ`-style symmetric affinities without materializing a
/// dense matrix.
public func reverseEdgeValues(rows: MLXArray, cols: MLXArray, vals: MLXArray, n: Int) -> MLXArray {
    let nL = Int64(n)
    let fwdKeys = rows.asType(.int64) * nL + cols.asType(.int64)
    let revKeys = cols.asType(.int64) * nL + rows.asType(.int64)
    let sortIdx = argSort(fwdKeys)
    let sortedKeys = fwdKeys[sortIdx]
    let sortedVals = vals[sortIdx]
    var pos = searchSorted(sortedKeys, revKeys)
    pos = minimum(pos, sortedKeys.dim(0) - 1)
    let matched = sortedKeys[pos] .== revKeys
    return MLX.where(matched, sortedVals[pos], 0.0)
}

/// Build the symmetrized t-SNE P-matrix as a deduplicated sparse edge list.
///
/// Calibrates a per-point Gaussian bandwidth to the target `perplexity` via a
/// vectorized binary search, forms the conditional `P(j|i)`, symmetrizes to
/// `p_ij = (P(j|i) + P(i|j)) / 2n`, then makes both directions explicit and dedups.
///
/// - Parameters:
///   - knnIndices: `(n, k)` neighbor indices.
///   - knnDists0: `(n, k)` neighbor distances (the reference passes squared distances;
///     they are squared again as the bandwidth input — mirrored here for parity).
///   - perplexity: target perplexity.
///   - n: number of points.
/// - Returns: `(rows, cols, vals)` for the symmetric P-matrix.
public func buildPMatrix(_ knnIndices: MLXArray, _ knnDists0: MLXArray, perplexity: Float, n: Int)
    -> (MLXArray, MLXArray, MLXArray)
{
    let k = knnIndices.dim(1)

    let knnDists = knnDists0.asType(.float32)
    let sqDists = knnDists * knnDists

    // Binary search for bandwidth beta (vectorized).
    var lo = MLXArray.full([n], values: MLXArray(Float(1e-20)))
    var hi = MLXArray.full([n], values: MLXArray(Float(1e4)))
    var beta = MLXArray.ones([n])
    let targetH = Float(log(Double(perplexity)))

    for _ in 0..<64 {
        let logits = -beta.expandedDimensions(axis: 1) * sqDists
        let logitsMax = logits.max(axis: 1, keepDims: true)
        let shifted = logits - logitsMax
        let expL = MLX.exp(shifted)
        let sumExp = expL.sum(axis: 1)
        let p = expL / sumExp.expandedDimensions(axis: 1)
        let h = MLX.log(sumExp) - (p * shifted).sum(axis: 1)

        let converged = abs(h - targetH) .< 1e-5
        let tooHigh = (h .> targetH) & (.!converged)
        let tooLow = (h .< targetH) & (.!converged)

        let newLo = MLX.where(tooHigh, beta, lo)
        let newHi = MLX.where(tooLow, beta, hi)
        beta = MLX.where(tooHigh, MLX.where(newHi .< 1e4, (beta + newHi) / 2.0, beta * 2.0), beta)
        beta = MLX.where(tooLow, MLX.where(newLo .> 1e-20, (newLo + beta) / 2.0, beta / 2.0), beta)
        lo = newLo
        hi = newHi
        eval(beta)
        if all(converged).item(Bool.self) { break }
    }

    // Final conditional P(j|i).
    let logits = -beta.expandedDimensions(axis: 1) * sqDists
    let logitsMax = logits.max(axis: 1, keepDims: true)
    let expL = MLX.exp(logits - logitsMax)
    let sumExp = expL.sum(axis: 1, keepDims: true)
    let weights = expL / sumExp

    // Sparse edges.
    let rowsMx = repeated(MLXArray(Int32(0) ..< Int32(n)), count: k, axis: 0)
    let colsMx = knnIndices.reshaped([-1]).asType(.int32)
    let valsMx = weights.reshaped([-1])

    // Symmetrize: p_ij = (p(j|i) + p(i|j)) / (2n).
    let wRev = reverseEdgeValues(rows: rowsMx, cols: colsMx, vals: valsMx, n: n)
    let wSym = (valsMx + wRev) / (2.0 * Float(n))

    // Make both directions explicit, dedup keeping first occurrence (CPU compaction).
    let nL = Int64(n)
    let allRows = concatenated([rowsMx, colsMx])
    let allCols = concatenated([colsMx, rowsMx])
    let allVals = concatenated([wSym, wSym])
    let keys = allRows.asType(.int64) * nL + allCols.asType(.int64)
    let sortIdx2 = argSort(keys)
    let sortedKeys2 = keys[sortIdx2]
    eval(sortedKeys2, sortIdx2)

    let sk = sortedKeys2.asArray(Int64.self)
    let si = sortIdx2.asArray(Int32.self)
    var finalIdx: [Int32] = []
    finalIdx.reserveCapacity(sk.count)
    for i in 0..<sk.count where i == 0 || sk[i] != sk[i - 1] {
        finalIdx.append(si[i])
    }
    let finalIdxMx = MLXArray(finalIdx)

    return (allRows[finalIdxMx], allCols[finalIdxMx], allVals[finalIdxMx])
}
