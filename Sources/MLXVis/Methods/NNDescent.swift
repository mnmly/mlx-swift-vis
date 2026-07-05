// NNDescent approximate k-nearest neighbors.
// Port of mlx_vis/_nndescent/nndescent.py
//
// Faithful port of the NN-Descent descent loop (Dong et al., WWW 2011):
//   1. seeded random initialization of the k-NN graph (self-avoiding),
//   2. neighbor-of-neighbor local joins + reverse candidates each iteration,
//   3. distance-based merge / dedup / top-k update,
//   4. convergence test on the fraction of changed edges.
//
// Deviations from the Python source (all performance-only, not algorithmic):
//   - fp16 candidate distances (`useFP16Dists`, default on) — the memory-bandwidth
//     lever for large n (~5× at 150k where fp32 spills into memory pressure). Data is
//     max-abs-normalised into a fp16-safe range first so recall is preserved (~0.997)
//     across input magnitudes; distances only rank candidates. See PORTING_NOTES.
//   - No low-dim random-projection early iters (Python uses this to cut FLOPs further).
//   - No active-point pruning (single-shot per iter). Bounding the k² join width was
//     tried and rejected — it wrecks convergence recall (see PORTING_NOTES).
//   Row/memory chunking of the distance gather IS done (see `gatherDists`).

import Foundation
import MLX

/// Approximate nearest-neighbor graph construction via NN-Descent.
public final class NNDescent {
    public let k: Int
    public let nIters: Int
    public let delta: Float
    public let randomState: Int
    public let verbose: Bool

    /// Use fp16 for the candidate distance gather (`gatherDists`). The wide
    /// `(n, c, d)` working set in the early iterations is memory-bandwidth bound;
    /// half precision halves its bytes for a ~1.4× large-`n` speedup. Distances are
    /// only used to *rank* candidates and the data is normalised into a fp16-safe
    /// range first, so recall is essentially unchanged (~0.997 → 0.997) across input
    /// magnitudes. Set `false` to force the exact fp32 gather. See PORTING_NOTES.
    public var useFP16Dists: Bool = true

    public init(k: Int, randomState: Int = 42, verbose: Bool = false) {
        self.k = k
        self.nIters = 20
        self.delta = 0.015
        self.randomState = randomState
        self.verbose = verbose
    }

    /// Build the approximate KNN graph.
    /// - Returns: `(indices, distances)` with Euclidean distances.
    public func build(_ x0: MLXArray) -> (indices: MLXArray, distances: MLXArray) {
        let x0f = x0.asType(.float32)
        let n = x0f.dim(0)
        let k = min(self.k, n - 1)
        let mc = k

        MLXRandom.seed(UInt64(randomState))

        // fp16 distances overflow / lose all precision for large-magnitude inputs
        // (recall collapses). Normalise into a fp16-safe range by the max abs value;
        // ranking / top-k is scale-invariant, so the whole descent runs in scaled units
        // and the final distances are multiplied back by `distScale`. No-op for fp32.
        let distScale: Float = useFP16Dists
            ? MLX.maximum(MLX.abs(x0f).max(), MLXArray(Float(1e-12))).item(Float.self) : 1.0
        let x = useFP16Dists ? x0f / distScale : x0f

        // Precompute squared norms (n,).
        let sqNorms = (x * x).sum(axis: 1)
        eval(sqNorms)

        // --- Random init: pick k neighbors in [0, n-1] avoiding self. ---
        // randInt over [0, n-1) then shift any pick >= row index up by one.
        var indices = MLXRandom.randInt(0 ..< Int32(n - 1), [n, k]).asType(.int32)
        let rowIdx = MLXArray(Int32(0) ..< Int32(n)).expandedDimensions(axis: 1)  // (n,1)
        indices = MLX.where(indices .>= rowIdx, indices + 1, indices).asType(.int32)

        // Initial distances + sort ascending.
        var dists = NNDescent.gatherDists(x, sqNorms, indices, fp16: useFP16Dists)
        let si = argSort(dists, axis: 1)
        indices = takeAlong(indices, si, axis: 1)
        dists = takeAlong(dists, si, axis: 1)
        eval(indices, dists)

        var updateFrac: Float = 1.0
        let rowVec = MLXArray(Int32(0) ..< Int32(n)).expandedDimensions(axis: 1)  // (n,1)

        for it in 0 ..< nIters {
            // Source breadth scales with how much the graph is still changing.
            let scale = Swift.min(1.0, Foundation.sqrt(Double(updateFrac)) * 3.0)
            let mcNew = max(3, Int(Double(min(k, mc)) * scale))
            let jNN = max(k / 2, min(k, Int(Double(k) * scale)))

            // --- neighbor-of-neighbor candidates ---
            // For each row i take its first mcNew neighbors; for each of those take
            // their first jNN neighbors. Shape (n, mcNew * jNN).
            let mcNewClamped = min(mcNew, k)
            let nnSub = indices[0..., ..<mcNewClamped]                       // (n, mcNew)
            let gathered = indices[nnSub.reshaped([-1])]                     // (n*mcNew, k)
                .reshaped([n, mcNewClamped, k])[0..., 0..., ..<jNN]          // (n, mcNew, jNN)
            let nnOfNN = gathered.reshaped([n, mcNewClamped * jNN])

            // --- reverse candidates (for edge i->j, i is a candidate for j) ---
            let useRev = updateFrac >= 0.10
            var newCands: MLXArray
            if useRev {
                let revCands = NNDescent.computeReverseCandidates(indices, n: n, k: k)
                newCands = concatenated([nnOfNN, revCands], axis: 1)
            } else {
                newCands = nnOfNN
            }

            // Distances for the new candidates; reuse stored dists for current.
            let newD = NNDescent.gatherDists(x, sqNorms, newCands, fp16: useFP16Dists)

            // Merge current + new, mask self-edges, dedup, take top-k.
            let allCands = concatenated([indices, newCands], axis: 1)
            var allDists = concatenated([dists, newD], axis: 1)
            let selfMask = allCands .== rowVec
            allDists = MLX.where(selfMask, MLXArray(Float(1e30)), allDists)

            let (newIndices, newDists) = NNDescent.dedupTopK(allCands, allDists, nRows: n, k: k)
            eval(newIndices, newDists)

            // Convergence bookkeeping: fraction of edges that changed.
            let flags = newIndices .!= indices
            let changed = flags.asType(.int32).sum().item(Int.self)
            updateFrac = Float(changed) / Float(n * k)

            indices = newIndices
            dists = newDists

            if verbose {
                print("Iter \(it + 1)/\(nIters): \(changed) updates (\(updateFrac))")
            }
            if updateFrac < delta {
                if verbose { print("Converged at iteration \(it + 1)") }
                break
            }
        }

        // Undo the fp16 normalisation on the returned distances (ranking was invariant).
        let finalDists = MLX.sqrt(maximum(dists, 0.0)) * distScale
        eval(indices, finalDists)
        return (indices.asType(.int32), finalDists)
    }

    /// Squared distances from each point i to col_ids[i, :].
    /// X: (n,d), sqNorms: (n,), colIds: (n,c) -> (n,c).
    ///
    /// The gathered targets form an (n, c, d) tensor; for large n with a wide
    /// candidate set (c grows ~k² during the descent) that single allocation can
    /// reach tens of GB and overflow the Metal buffer limit. Tile over row blocks
    /// so peak memory is bounded to roughly one block's (rowChunk, c, d) — the
    /// result is bit-identical, only the materialisation is staged.
    static func gatherDists(_ x: MLXArray, _ sqNorms: MLXArray, _ colIds: MLXArray, fp16: Bool = false) -> MLXArray {
        let n = colIds.dim(0)
        let c = colIds.dim(1)
        let d = x.dim(1)
        // fp16 candidate data halves the (rowChunk, c, d) bytes; norms stay fp32 so the
        // squared-distance combine keeps its precision (only the dot product is fp16).
        let xg = fp16 ? x.asType(.float16) : x
        // Target ~1 GB of float32 for the (rowChunk, c, d) working set.
        let elemBudget = 256_000_000
        let rowChunk = max(1, min(n, elemBudget / max(1, c * d)))
        if rowChunk >= n {
            return gatherDistsBlock(xg, sqNorms, colIds, lo: 0, hi: n)
        }
        var parts: [MLXArray] = []
        var start = 0
        while start < n {
            let end = min(start + rowChunk, n)
            let block = gatherDistsBlock(xg, sqNorms, colIds, lo: start, hi: end)
            // Materialise each block now so the wide (block, c, d) intermediate is
            // freed before the next block — bounded peak, no accumulation.
            asyncEval(block)
            parts.append(block)
            start = end
        }
        return concatenated(parts, axis: 0)
    }

    /// `gatherDists` restricted to rows `lo..<hi` (the tiling primitive).
    private static func gatherDistsBlock(
        _ x: MLXArray, _ sqNorms: MLXArray, _ colIds: MLXArray, lo: Int, hi: Int
    ) -> MLXArray {
        let c = colIds.dim(1)
        let d = x.dim(1)
        let rc = hi - lo
        let xRows = x[lo ..< hi]                          // (rc, d)
        let flat = colIds[lo ..< hi].reshaped([-1])       // (rc*c,)
        let xTgt = x[flat].reshaped([rc, c, d])           // (rc, c, d)
        // dots[i,c] = sum_d xRows[i,d] * xTgt[i,c,d]
        let dots = einsum("id,icd->ic", xRows, xTgt)
        let nbrNorms = sqNorms[flat].reshaped([rc, c])
        return maximum(sqNorms[lo ..< hi].expandedDimensions(axis: 1) + nbrNorms - 2.0 * dots, 0.0)
    }

    /// For each edge (i -> j), record i as a reverse candidate of j.
    /// Returns (n, k) array of reverse candidate source indices (0-padded).
    static func computeReverseCandidates(_ indices: MLXArray, n: Int, k: Int) -> MLXArray {
        let srcAll = broadcast(
            MLXArray(Int32(0) ..< Int32(n)).expandedDimensions(axis: 1), to: [n, k]
        ).reshaped([-1])
        let dstAll = indices.reshaped([-1])

        let revOrder = argSort(dstAll, axis: 0)
        let revSrc = srcAll[revOrder]
        let revDst = dstAll[revOrder]

        // Position within each destination group via a cummax of group-start markers.
        let isNewGroup = concatenated([
            MLXArray([Int32(1)]),
            (revDst[1...] .!= revDst[..<(n * k - 1)]).asType(.int32),
        ])
        let globalPos = MLXArray(Int32(0) ..< Int32(n * k))
        let groupStartMarkers = MLX.where(isNewGroup .!= 0, globalPos, MLXArray(Int32(0)))
        let groupStarts = cummax(groupStartMarkers, axis: 0)
        let withinPos = globalPos - groupStarts

        let keep = withinPos .< Int32(k)
        var flatIdx = revDst * Int32(k) + withinPos
        flatIdx = MLX.where(keep, flatIdx, MLXArray(Int32(0)))
        let revSrcKept = MLX.where(keep, revSrc, MLXArray(Int32(0)))

        let zeros = MLXArray.zeros([n * k], dtype: .int32)
        let revCands = zeros.at[flatIdx].add(revSrcKept).reshaped([n, k])
        eval(revCands)
        return revCands
    }

    /// Deduplicate candidates per row and select the top-k by distance.
    static func dedupTopK(_ allCands: MLXArray, _ allDists: MLXArray, nRows: Int, k: Int)
        -> (MLXArray, MLXArray)
    {
        // Sort by candidate id so duplicates become adjacent.
        let candSort = argSort(allCands, axis: 1)
        let sortedC = takeAlong(allCands, candSort, axis: 1)
        var sortedD = takeAlong(allDists, candSort, axis: 1)

        let cols = allCands.dim(1)
        let isDup = concatenated([
            MLXArray.zeros([nRows, 1], dtype: .bool),
            sortedC[0..., 1...] .== sortedC[0..., ..<(cols - 1)],
        ], axis: 1)
        sortedD = MLX.where(isDup, MLXArray(Float(1e30)), sortedD)

        // Top-k smallest, then sort that block ascending.
        let part = argPartition(sortedD, kth: k - 1, axis: 1)[0..., ..<k]
        let topDists = takeAlong(sortedD, part, axis: 1)
        let subSort = argSort(topDists, axis: 1)
        let topIdx = takeAlong(part, subSort, axis: 1)

        let newIndices = takeAlong(sortedC, topIdx, axis: 1)
        let newDists = takeAlong(sortedD, topIdx, axis: 1)
        return (newIndices, newDists)
    }
}
