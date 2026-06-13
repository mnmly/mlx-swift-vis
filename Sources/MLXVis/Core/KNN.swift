// Shared KNN computation: brute-force (GPU) or NNDescent (approximate).
// Port of mlx_vis/_knn.py

import MLX

/// Strategy for k-nearest-neighbor computation.
public enum KNNMethod: Sendable {
    /// brute for n <= 20000, nndescent for larger.
    case auto
    /// Exact brute-force on GPU.
    case brute
    /// Approximate via NNDescent.
    case nndescent
}

/// Compute k-nearest neighbors.
///
/// - Parameters:
///   - x: Input data `(n, d)`.
///   - k: Number of neighbors.
///   - method: Selection strategy.
///   - returnEuclidean: If true return Euclidean distances (sqrt), else squared.
///   - randomState: Seed forwarded to NNDescent.
///   - verbose: Print progress.
/// - Returns: `(indices, distances)`, both `(n, k)`; indices int32, distances float32.
public func computeKNN(
    _ x: MLXArray,
    k: Int,
    method: KNNMethod = .auto,
    returnEuclidean: Bool = true,
    randomState: Int? = nil,
    verbose: Bool = false
) -> (indices: MLXArray, distances: MLXArray) {
    let n = x.dim(0)
    var resolved = method
    if resolved == .auto {
        resolved = n <= 20000 ? .brute : .nndescent
    }

    if resolved == .nndescent {
        let nn = NNDescent(k: k, randomState: randomState ?? 42, verbose: verbose)
        let (indices, dists) = nn.build(x)
        return (indices, returnEuclidean ? dists : dists * dists)
    }

    return bruteForceKNN(x, k: k, returnEuclidean: returnEuclidean)
}

/// Exact k-nearest neighbors via chunked pairwise distances on the GPU.
public func bruteForceKNN(_ x0: MLXArray, k: Int, returnEuclidean: Bool) -> (indices: MLXArray, distances: MLXArray) {
    let x = x0.asType(.float32)
    let n = x.dim(0)
    let sqNorms = (x * x).sum(axis: 1)
    eval(sqNorms)

    let chunkSize = min(n, max(1000, 500_000_000 / (n * 4)))
    var idxParts: [MLXArray] = []
    var distParts: [MLXArray] = []

    var start = 0
    while start < n {
        let end = min(start + chunkSize, n)
        let xChunk = x[start ..< end]

        var dChunk = maximum(
            sqNorms[start ..< end].expandedDimensions(axis: 1)
                + sqNorms.expandedDimensions(axis: 0)
                - 2.0 * xChunk.matmul(x.transposed()),
            0.0
        )

        // Set self-distance to +inf so a point never selects itself.
        let selfMask = (arangeColumn(start, end) .== arangeRow(n)).asType(.float32)
        dChunk = dChunk + selfMask * 1e30

        let idx = argSort(dChunk, axis: 1)[0..., ..<k].asType(.int32)
        let dists = takeAlong(dChunk, idx, axis: 1)
        // Schedule this chunk without blocking so the next chunk's matmul overlaps
        // with this chunk's reduction/readback. This also materializes the large
        // (chunk x n) intermediate now so it is freed before the next iteration —
        // bounded peak memory, no accumulation across chunks.
        asyncEval(idx, dists)
        idxParts.append(idx)
        distParts.append(dists)

        start = end
    }

    let indices = concatenated(idxParts, axis: 0)
    var dists = concatenated(distParts, axis: 0)
    if returnEuclidean {
        dists = MLX.sqrt(maximum(dists, 0.0))
    }
    eval(indices, dists)
    return (indices, dists)
}
