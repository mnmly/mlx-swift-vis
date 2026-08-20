// TriMap implementation in pure MLX for Apple Silicon.
// Port of mlx_vis/_trimap/trimap.py
//
// Reference: Amid & Warmuth, "TriMap: Large-scale Dimensionality Reduction
// Using Triplets" (arXiv:1910.00204).
//
// Triplet generation (KNN inlier triplets + random triplets) is done on CPU in
// plain Swift (mirroring the reference's numpy work); the embedding optimization
// runs on the GPU via delta-bar-delta gradient descent with momentum.

import Foundation
import MLX
import MLXRandom

/// TriMap dimensionality reduction via triplet constraints using MLX.
public final class TriMap {
    public var nComponents: Int
    public var nNeighbors: Int
    public var nInliers: Int
    public var nOutliers: Int
    public var nRandom: Int
    public var nIters: Int
    public var lr: Float
    public var weightTemp: Float
    public var pcaDim: Int?
    public var randomState: Int?
    public var verbose: Bool
    public var knnMethod: KNNMethod
    public var normalize: Normalization

    public private(set) var embedding: MLXArray?

    /// Optional progress hook, called periodically during `fitTransform` with
    /// `(iteration, totalIterations, embedding)` so callers can animate the
    /// optimization. Fires once at iteration 0 with the initial embedding, then on the `progressEvery` schedule. The embedding is `(nSamples, nComponents)`. No-op when unset.
    public var onEpoch: ((Int, Int, MLXArray) -> Void)?

    /// How often `onEpoch` fires, in optimizer steps (default 10). Set to 1 for a
    /// frame every step (smoothest animation, more overhead); ignored when `onEpoch`
    /// is unset. Independent of the internal graph-eval cadence (memory safeguard).
    public var progressEvery: Int = 10

    /// Optional phase hook, called at each setup milestone during `fitTransform`
    /// (PCA preprocess, KNN build, triplet generation, optimization start) with a
    /// short human-readable label. Lets callers surface the otherwise-silent
    /// pre-optimization work to a UI. Fires regardless of `verbose`; no-op when unset.
    public var onPhase: ((String) -> Void)?

    /// A precomputed k-NN graph to use instead of building one.
    ///
    /// Skips the neighbour search — the step that dominates the wall clock for large
    /// inputs — and feeds the rest of the pipeline the stored arrays verbatim: the
    /// affinity graph, the initialization and MLX's PRNG stream are bit-identical to the
    /// internally-built path, and the optimizer's own (pre-existing, scatter-add) run-to-
    /// run variation is unchanged either way. The graph must match
    /// ``TriMap/knnGraphSpec(nSamples:)``; prefer ``TriMap/fitTransform(_:knnGraph:)``,
    /// which validates and throws, over setting this directly (a mismatch here traps).
    public var knnGraph: KNNGraph?

    /// Capture the k-NN graph `fitTransform` used into ``TriMap/lastKNNGraph``.
    ///
    /// Off by default: capturing copies the `(n, k)` index and distance planes to the
    /// host. Turn it on for the run that fills a cache.
    public var exportKNNGraph: Bool = false

    /// The k-NN graph the most recent `fitTransform` used, when ``TriMap/exportKNNGraph``
    /// was set. Serialize it with ``KNNGraph/serialized()`` to cache it.
    public private(set) var lastKNNGraph: KNNGraph?

    public init(
        nComponents: Int = 2,
        nNeighbors: Int = 12,
        nInliers: Int = 12,
        nOutliers: Int = 4,
        nRandom: Int = 3,
        nIters: Int = 400,
        lr: Float = 1000.0,
        weightTemp: Float = 0.5,
        pcaDim: Int? = 100,
        randomState: Int? = nil,
        verbose: Bool = false,
        knnMethod: KNNMethod = .auto,
        normalize: Normalization = .none
    ) {
        self.nComponents = nComponents
        self.nNeighbors = nNeighbors
        self.nInliers = nInliers
        self.nOutliers = nOutliers
        self.nRandom = nRandom
        self.nIters = nIters
        self.lr = lr
        self.weightTemp = weightTemp
        self.pcaDim = pcaDim
        self.randomState = randomState
        self.verbose = verbose
        self.knnMethod = knnMethod
        self.normalize = normalize
    }

    private func log(_ msg: @autoclosure () -> String) {
        guard verbose || onPhase != nil else { return }
        let s = msg()
        if verbose { print(s) }
        onPhase?(s)
    }

    /// The k-NN build this configuration will run for `nSamples` points.
    ///
    /// Use it to key a k-NN cache, and to preflight a ``KNNGraph`` before injecting it.
    ///
    /// - Parameter n: Number of input rows.
    /// - Returns: The resolved search method, neighbour count, distance convention and
    ///   seed this instance will use.
    public func knnGraphSpec(nSamples n: Int) -> KNNGraphSpec {
        KNNGraphSpec(
            n: n, k: nNeighbors, method: knnMethod,
            distanceKind: .euclidean, randomState: randomState ?? 42)
    }

    /// Fit using a precomputed k-NN graph, skipping the neighbour search.
    ///
    /// - Parameters:
    ///   - x: Input data `(nSamples, nFeatures)`.
    ///   - graph: A graph matching ``TriMap/knnGraphSpec(nSamples:)``, e.g. one exported
    ///     from an earlier fit via ``TriMap/exportKNNGraph``.
    /// - Returns: The embedding `(nSamples, nComponents)`.
    /// - Throws: A ``KNNGraphError`` if `graph` does not match this configuration; the
    ///   graph is never truncated or converted to make it fit.
    public func fitTransform(_ x: MLXArray, knnGraph graph: KNNGraph) throws -> MLXArray {
        try graph.validate(against: knnGraphSpec(nSamples: x.dim(0)))
        let saved = knnGraph
        self.knnGraph = graph
        defer { self.knnGraph = saved }
        return fitTransform(x)
    }

    /// Relays NNDescent's per-iteration convergence measure to ``TriMap/onPhase``.
    /// `nil` — and therefore free — when nothing is listening.
    private var knnProgressHook: KNNProgressHandler? {
        guard verbose || onPhase != nil else { return nil }
        return { [self] iteration, total, frac in
            log(knnProgressLine(iteration: iteration, total: total, updatedFraction: frac))
        }
    }

    /// Fit TriMap and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        var x = normalizeInput(x0.asType(.float32), method: normalize)
        let n = x.dim(0)
        var dim = x.dim(1)
        if let rs = randomState { MLXRandom.seed(UInt64(rs)) }

        log("TriMap: \(n) points, \(dim) dims")

        // PCA preprocessing if dim exceeds pca_dim.
        if let pd = pcaDim, dim > pd {
            x = pcaReduce(x, dim: pd)
            eval(x)
            dim = pd
            log("PCA: reduced to \(pd) dims")
        }

        // k-NN (Euclidean distances, sorted ascending; no self).
        log("Computing k-NN...")
        let knn = resolveKNN(
            x, spec: knnGraphSpec(nSamples: n), graph: knnGraph, export: exportKNNGraph,
            verbose: verbose, onIteration: knnProgressHook)
        lastKNNGraph = knn.exported
        let (knnIndices, knnDistances) = (knn.indices, knn.distances)

        // Sigma: mean Euclidean distance to 4th-6th NN (0-indexed cols 3..5), clamped.
        let k = knnDistances.dim(1)
        let hiCol = min(6, k)
        let loCol = min(3, hiCol)
        let sigmaMx: MLXArray
        if hiCol > loCol {
            sigmaMx = maximum(knnDistances[0..., loCol ..< hiCol].mean(axis: 1), 1e-10)
        } else {
            sigmaMx = MLXArray.full([n], values: MLXArray(Float(1e-10)))
        }

        // Generate triplets and weights (CPU).
        log("Generating triplets...")
        let (triplets, weights) = generateTriplets(
            x: x, n: n, dim: dim, knnIndices: knnIndices,
            knnDistances: knnDistances, sigma: sigmaMx)
        log("Generated \(triplets.dim(0)) triplets")

        // Initialize embedding via PCA scaled by 0.01.
        var y = pcaReduce(x, dim: nComponents) * 0.01
        eval(y)

        // Delta-bar-delta optimization with momentum.
        var vel = MLXArray.zeros(like: y)
        var gain = MLXArray.ones(like: y)
        eval(vel, gain)

        log("Starting optimization...")
        if onEpoch != nil { eval(y); onEpoch?(0, max(1, nIters), y) }  // initial frame
        for itr in 1...max(1, nIters) {
            // Cooperative cancellation: return the best-so-far embedding. No-op outside a Task.
            if Task.isCancelled { break }
            let grad = computeGradient(y: y, triplets: triplets, weights: weights)

            // Gain update: shrink if grad and velocity agree in sign, grow otherwise.
            let sameSign = (vel * grad) .> 0.0
            gain = MLX.where(sameSign, gain * 0.8, gain + 0.2)
            gain = maximum(gain, 0.01)

            let momentum: Float = itr <= 250 ? 0.5 : 0.8
            vel = momentum * vel - lr * gain * grad
            y = y + vel

            let trimapTotal = max(1, nIters)
            let progressHit = onEpoch != nil && (itr % max(1, progressEvery) == 0 || itr == trimapTotal)
            if itr % 10 == 0 || progressHit {
                eval(y, vel, gain)
            }
            if progressHit { onEpoch?(itr, trimapTotal, y) }
        }

        eval(y)
        self.embedding = y
        return y
    }

    // MARK: - Triplet generation (CPU)

    /// Generate (anchor, inlier, outlier) triplets plus random triplets, and the
    /// tempered-log-transformed weights. Returns `(triplets: (T,3) int32, weights: (T,) float32)`.
    private func generateTriplets(
        x: MLXArray, n: Int, dim: Int,
        knnIndices: MLXArray, knnDistances: MLXArray, sigma: MLXArray
    ) -> (MLXArray, MLXArray) {
        let k = knnIndices.dim(1)
        let nInliersLocal = min(nInliers, k)

        // Pull needed arrays to CPU.
        let knnIdx = knnIndices.asArray(Int32.self)       // n*k
        let knnDist = knnDistances.asArray(Float.self)    // n*k
        let sig = sigma.asArray(Float.self)               // n
        let xArr = x.asArray(Float.self)                  // n*dim

        @inline(__always)
        func sqDist(_ a: Int, _ b: Int) -> Float {
            var s: Float = 0
            let ao = a * dim
            let bo = b * dim
            for d in 0..<dim {
                let diff = xArr[ao + d] - xArr[bo + d]
                s += diff * diff
            }
            return s
        }

        let nKnnTriplets = n * nInliersLocal * nOutliers
        let nRandTriplets = n * nRandom
        let total = nKnnTriplets + nRandTriplets

        var tripArr = [Int32](repeating: 0, count: total * 3)
        var wArr = [Float](repeating: 0, count: total)

        // Random outlier indices (KNN triplets) and random j/k (random triplets).
        let outliers = nKnnTriplets > 0
            ? MLXRandom.randInt(0 ..< Int32(n), [nKnnTriplets]).asArray(Int32.self)
            : []
        let randJ = nRandTriplets > 0
            ? MLXRandom.randInt(0 ..< Int32(n), [nRandTriplets]).asArray(Int32.self)
            : []
        let randK = nRandTriplets > 0
            ? MLXRandom.randInt(0 ..< Int32(n), [nRandTriplets]).asArray(Int32.self)
            : []

        // --- KNN triplets ---
        // anchors[t] = i; for each i, nInliers positions each repeated nOutliers times.
        var t = 0
        for i in 0..<n {
            let sigI = sig[i]
            for pos in 0..<nInliersLocal {
                let inlier = Int(knnIdx[i * k + pos])
                let d = knnDist[i * k + pos]
                let sigJ = sig[inlier]
                // P[i,j] = -d(i,j)^2 / (sigma_i * sigma_j)
                let pIj = -(d * d) / (sigI * sigJ)
                for _ in 0..<nOutliers {
                    let outlier = Int(outliers[t])
                    let dIkSq = sqDist(i, outlier)
                    let pOutlier = -dIkSq / (sigI * sig[outlier])
                    tripArr[t * 3 + 0] = Int32(i)
                    tripArr[t * 3 + 1] = Int32(inlier)
                    tripArr[t * 3 + 2] = Int32(outlier)
                    wArr[t] = pIj - pOutlier
                    t += 1
                }
            }
        }

        // --- Random triplets ---
        var r = 0
        for i in 0..<n {
            let sigI = sig[i]
            for _ in 0..<nRandom {
                let j = Int(randJ[r])
                let kk = Int(randK[r])
                let pIj = -sqDist(i, j) / (sigI * sig[j])
                let pIk = -sqDist(i, kk) / (sigI * sig[kk])
                tripArr[t * 3 + 0] = Int32(i)
                tripArr[t * 3 + 1] = Int32(j)
                tripArr[t * 3 + 2] = Int32(kk)
                wArr[t] = (pIj - pIk) * 0.1
                t += 1
                r += 1
            }
        }

        // Shift and transform: tempered_log(1 + w, t).
        var wMin = Float.greatestFiniteMagnitude
        for w in wArr where w < wMin { wMin = w }
        let temp = weightTemp
        let exponent = 1.0 - temp
        let invDenom = 1.0 / (1.0 - temp)
        for idx in 0..<total {
            let shifted = wArr[idx] - wMin
            wArr[idx] = (powf(1.0 + shifted, exponent) - 1.0) * invDenom
        }

        let triplets = MLXArray(tripArr, [total, 3])
        let weights = MLXArray(wArr)
        eval(triplets, weights)
        return (triplets, weights)
    }

    // MARK: - Gradient

    /// Vectorized triplet gradient over all triplets via scatter-add.
    private func computeGradient(y: MLXArray, triplets: MLXArray, weights: MLXArray) -> MLXArray {
        let ti = triplets[0..., 0]
        let tj = triplets[0..., 1]
        let tk = triplets[0..., 2]

        let yi = y[ti]
        let yj = y[tj]
        let yk = y[tk]

        let diffIj = yi - yj
        let diffIk = yi - yk

        let dIj = 1.0 + (diffIj * diffIj).sum(axis: 1)  // (T,)
        let dIk = 1.0 + (diffIk * diffIk).sum(axis: 1)  // (T,)

        let denom = (dIj + dIk) * (dIj + dIk)
        let w = (weights / denom).expandedDimensions(axis: 1)  // (T, 1)

        let gradSim = diffIj * (dIk.expandedDimensions(axis: 1) * w)  // (T, d)
        let gradOut = diffIk * (dIj.expandedDimensions(axis: 1) * w)  // (T, d)

        var grad = MLXArray.zeros(like: y)
        grad = grad.at[ti].add(gradSim - gradOut)
        grad = grad.at[tj].add(-gradSim)
        grad = grad.at[tk].add(gradOut)
        return grad
    }
}
