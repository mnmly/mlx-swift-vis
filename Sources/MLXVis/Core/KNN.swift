// Shared KNN computation: brute-force (GPU) or NNDescent (approximate).
// Port of mlx_vis/_knn.py

import Foundation
import MLX
import MLXRandom

/// Strategy for k-nearest-neighbor computation.
public enum KNNMethod: Sendable, Hashable, Codable {
    /// brute for n <= 20000, nndescent for larger.
    case auto
    /// Exact brute-force on GPU.
    case brute
    /// Approximate via NNDescent.
    case nndescent
}

/// Progress during an approximate (NNDescent) k-NN build:
/// `(iteration, totalIterations, updatedFraction)`.
///
/// `iteration` is 1-based, with a `0` call right after the random initialization;
/// `updatedFraction` is the share of the `n * k` neighbour slots that changed in that
/// iteration — NNDescent's own convergence measure, so reporting it costs nothing.
public typealias KNNProgressHandler = (Int, Int, Float) -> Void

/// The point count above which ``KNNMethod/auto`` switches to ``NNDescent``.
let knnAutoBruteLimit = 20000

/// Resolve ``KNNMethod/auto`` for a given point count.
func resolveKNNMethod(_ method: KNNMethod, n: Int) -> KNNMethod {
    method == .auto ? (n <= knnAutoBruteLimit ? .brute : .nndescent) : method
}

/// A short human-readable line for a k-NN build iteration, e.g.
/// `"KNN: iter 4/20 · 2.1% updating"`.
func knnProgressLine(iteration: Int, total: Int, updatedFraction: Float) -> String {
    String(
        format: "KNN: iter %d/%d · %.1f%% updating", iteration, total,
        Double(updatedFraction) * 100.0)
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
///   - onIteration: Optional per-iteration hook for the NNDescent path; see
///     ``KNNProgressHandler``. Ignored by the brute-force path (a single GPU pass).
/// - Returns: `(indices, distances)`, both `(n, k)`; indices int32, distances float32.
public func computeKNN(
    _ x: MLXArray,
    k: Int,
    method: KNNMethod = .auto,
    returnEuclidean: Bool = true,
    randomState: Int? = nil,
    verbose: Bool = false,
    onIteration: KNNProgressHandler? = nil
) -> (indices: MLXArray, distances: MLXArray) {
    let n = x.dim(0)
    let resolved = resolveKNNMethod(method, n: n)

    if resolved == .nndescent {
        let nn = NNDescent(k: k, randomState: randomState ?? 42, verbose: verbose)
        nn.onIteration = onIteration
        let (indices, dists) = nn.build(x)
        return (indices, returnEuclidean ? dists : dists * dists)
    }

    return bruteForceKNN(x, k: k, returnEuclidean: returnEuclidean)
}

/// Compute a k-nearest-neighbor graph as a cacheable ``KNNGraph`` value.
///
/// The same search as ``computeKNN(_:k:method:returnEuclidean:randomState:verbose:onIteration:)``,
/// with the result captured as plain host arrays that can be serialized and later
/// injected into a fit (see ``KNNGraph``) to skip the build entirely.
///
/// - Parameters:
///   - x: Input data `(n, d)` — the matrix the *method* will search, i.e. after any
///     PCA preprocessing that method applies.
///   - k: Number of neighbors.
///   - method: Selection strategy.
///   - returnEuclidean: If true store Euclidean distances, else squared.
///   - randomState: Seed forwarded to NNDescent.
///   - verbose: Print progress.
///   - onIteration: Optional per-iteration hook for the NNDescent path.
/// - Returns: The graph, tagged with the resolved method, distance convention and seed.
public func computeKNNGraph(
    _ x: MLXArray,
    k: Int,
    method: KNNMethod = .auto,
    returnEuclidean: Bool = true,
    randomState: Int? = nil,
    verbose: Bool = false,
    onIteration: KNNProgressHandler? = nil
) -> KNNGraph {
    let spec = KNNGraphSpec(
        n: x.dim(0), k: k, method: method,
        distanceKind: returnEuclidean ? .euclidean : .squared, randomState: randomState)
    let (indices, distances) = computeKNN(
        x, k: spec.k, method: spec.method, returnEuclidean: returnEuclidean,
        randomState: spec.randomState, verbose: verbose, onIteration: onIteration)
    return KNNGraph(
        indices: indices, distances: distances, distanceKind: spec.distanceKind,
        method: spec.method, randomState: spec.randomState)
}

/// The k-NN arrays a method will use, plus the graph to hand back to the caller.
struct KNNResolution {
    let indices: MLXArray
    let distances: MLXArray
    /// Non-nil only when export was requested.
    let exported: KNNGraph?
}

/// Single entry point every method uses for its k-NN step: build, or reuse a
/// precomputed graph.
///
/// Driving both branches off one ``KNNGraphSpec`` — the same value the method hands
/// callers from `knnGraphSpec(nSamples:)` — keeps the validated identity and the actual
/// build in lockstep; there is no second place where `k` or the method could drift.
func resolveKNN(
    _ x: MLXArray,
    spec: KNNGraphSpec,
    graph: KNNGraph?,
    export: Bool,
    verbose: Bool,
    onIteration: KNNProgressHandler?
) -> KNNResolution {
    if let graph {
        do {
            try graph.validate(against: spec)
        } catch {
            // Reachable only by assigning the `knnGraph` property directly; the
            // `fitTransform(_:knnGraph:)` overloads validate and throw first.
            preconditionFailure("MLXVis: precomputed k-NN graph rejected — \(error)")
        }
        // A build has exactly one side effect beyond its return value: NNDescent seeds
        // the global PRNG and draws its random initialization, consuming a single key
        // split. Replay that split (cheaply, without the draw) so every downstream
        // random op — negative sampling, spectral/random init — sees the same stream as
        // the internally-built path and the embeddings stay bit-identical.
        if spec.method == .nndescent {
            MLXRandom.seed(UInt64(spec.randomState ?? 42))
            _ = MLXRandom.globalState.next()
        }
        return KNNResolution(
            indices: graph.indicesArray, distances: graph.distancesArray,
            exported: export ? graph : nil)
    }

    let (indices, distances) = computeKNN(
        x, k: spec.k, method: spec.method,
        returnEuclidean: spec.distanceKind == .euclidean,
        randomState: spec.randomState, verbose: verbose, onIteration: onIteration)
    let exported =
        export
        ? KNNGraph(
            indices: indices, distances: distances, distanceKind: spec.distanceKind,
            method: spec.method, randomState: spec.randomState)
        : nil
    return KNNResolution(indices: indices, distances: distances, exported: exported)
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
