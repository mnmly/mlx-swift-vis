// DREAMS dimensionality reduction in pure MLX for Apple Silicon.
// Port of mlx_vis/_dreams/dreams.py
//
// DREAMS (Dimensionality Reduction Enhanced Across Multiple Scales) combines
// t-SNE local structure preservation with PCA global structure preservation via a
// regularization term: L = (1-lam)*L_tsne + (lam/n)*||Y - alpha*Y_tilde||^2.
//
// DREAMS reuses the shared t-SNE-family core (Core/TSNEOptimizer.swift) and supplies
// its PCA-regularization as a `transformGradient` hook: a 2D PCA reference embedding
// Y_tilde plus the gradient term (1-lam)*grad + (2*lam/n)*(Y - alpha*Y_tilde) each step.
// Repulsion (FFT / full / chunked) is shared via ``TSNERepulsion``.

import Foundation
import MLX
import MLXRandom

/// DREAMS dimensionality reduction using MLX on the Metal GPU.
public final class DREAMS {
    public var nComponents: Int
    public var perplexity: Float
    public var learningRate: Float
    public var nIter: Int
    public var earlyExaggeration: Float
    public var earlyExaggerationIter: Int
    public var lam: Float
    public var pcaDim: Int?
    public var randomState: Int?
    public var verbose: Int
    public var knnMethod: KNNMethod
    public var normalize: Normalization

    public private(set) var embedding: MLXArray?

    /// Optional progress hook, called periodically during `fitTransform` with
    /// `(iteration, totalIterations, embedding)` so callers can animate the
    /// optimization. Fires once at iteration 0 with the initial embedding, then on the `progressEvery` schedule. No-op when unset.
    public var onEpoch: ((Int, Int, MLXArray) -> Void)?

    /// How often `onEpoch` fires, in optimizer steps (default 10). Set to 1 for a
    /// frame every step (smoothest animation, more overhead); ignored when `onEpoch`
    /// is unset. Independent of the internal graph-eval cadence (memory safeguard).
    public var progressEvery: Int = 10

    /// Optional phase hook, called at each setup milestone during `fitTransform`
    /// (KNN build, affinity-graph construction, optimization start) with a short
    /// human-readable label. Lets callers surface the otherwise-silent
    /// pre-optimization work to a UI. Fires regardless of `verbose`; no-op when unset.
    public var onPhase: ((String) -> Void)?

    /// A precomputed k-NN graph to use instead of building one.
    ///
    /// Skips the neighbour search — the step that dominates the wall clock for large
    /// inputs — and feeds the rest of the pipeline the stored arrays verbatim: the
    /// affinity graph, the initialization and MLX's PRNG stream are bit-identical to the
    /// internally-built path, and the optimizer's own (pre-existing, scatter-add) run-to-
    /// run variation is unchanged either way. The graph must match
    /// ``DREAMS/knnGraphSpec(nSamples:)``; prefer ``DREAMS/fitTransform(_:knnGraph:)``,
    /// which validates and throws, over setting this directly (a mismatch here traps).
    public var knnGraph: KNNGraph?

    /// Capture the k-NN graph `fitTransform` used into ``DREAMS/lastKNNGraph``.
    ///
    /// Off by default: capturing copies the `(n, k)` index and distance planes to the
    /// host. Turn it on for the run that fills a cache.
    public var exportKNNGraph: Bool = false

    /// The k-NN graph the most recent `fitTransform` used, when ``DREAMS/exportKNNGraph``
    /// was set. Serialize it with ``KNNGraph/serialized()`` to cache it.
    public private(set) var lastKNNGraph: KNNGraph?

    public init(
        nComponents: Int = 2,
        perplexity: Float = 30.0,
        learningRate: Float = 200.0,
        nIter: Int = 500,
        earlyExaggeration: Float = 12.0,
        earlyExaggerationIter: Int = 250,
        lam: Float = 0.15,
        pcaDim: Int? = 50,
        randomState: Int? = nil,
        verbose: Int = 0,
        knnMethod: KNNMethod = .auto,
        normalize: Normalization = .none
    ) {
        self.nComponents = nComponents
        self.perplexity = perplexity
        self.learningRate = learningRate
        self.nIter = nIter
        self.earlyExaggeration = earlyExaggeration
        self.earlyExaggerationIter = earlyExaggerationIter
        self.lam = lam
        self.pcaDim = pcaDim
        self.randomState = randomState
        self.verbose = verbose
        self.knnMethod = knnMethod
        self.normalize = normalize
    }

    private func log(_ msg: @autoclosure () -> String) {
        guard verbose > 0 || onPhase != nil else { return }
        let s = msg()
        if verbose > 0 { print(s) }
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
            n: n, k: min(Int(3 * perplexity), n - 1), method: knnMethod,
            distanceKind: .squared, randomState: randomState)
    }

    /// Fit using a precomputed k-NN graph, skipping the neighbour search.
    ///
    /// - Parameters:
    ///   - x: Input data `(nSamples, nFeatures)`.
    ///   - graph: A graph matching ``DREAMS/knnGraphSpec(nSamples:)``, e.g. one exported
    ///     from an earlier fit via ``DREAMS/exportKNNGraph``.
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

    /// Relays NNDescent's per-iteration convergence measure to ``DREAMS/onPhase``.
    /// `nil` — and therefore free — when nothing is listening.
    private var knnProgressHook: KNNProgressHandler? {
        guard verbose > 0 || onPhase != nil else { return nil }
        return { [self] iteration, total, frac in
            log(knnProgressLine(iteration: iteration, total: total, updatedFraction: frac))
        }
    }

    /// Fit DREAMS and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        let xMx = normalizeInput(x0.asType(.float32), method: normalize)
        let n = xMx.dim(0)
        let dim = xMx.dim(1)

        // PCA preprocessing (high-dim -> pcaDim) before KNN.
        let xForKNN: MLXArray
        if let pcaDim, dim > pcaDim {
            log("Applying PCA: \(dim) -> \(pcaDim) dims...")
            xForKNN = pcaReduce(xMx, dim: pcaDim)
        } else {
            xForKNN = xMx
        }

        log("Computing k-NN...")
        let knn = resolveKNN(
            xForKNN, spec: knnGraphSpec(nSamples: n), graph: knnGraph, export: exportKNNGraph,
            verbose: verbose > 0, onIteration: knnProgressHook)
        lastKNNGraph = knn.exported
        let (knnIndices, knnDists) = (knn.indices, knn.distances)

        log("Building affinity graph...")
        let (edgeFrom, edgeTo, edgeWeights) = buildPMatrix(knnIndices, knnDists, perplexity: perplexity, n: n)
        eval(edgeFrom, edgeTo, edgeWeights)

        // 2D PCA reference embedding Y_tilde (top nComponents of xForKNN).
        let yTilde = pcaReduce(xForKNN, dim: nComponents)
        eval(yTilde)

        if let randomState { MLXRandom.seed(UInt64(randomState)) }

        // PCA initialization: scale so first dim has std = 1e-4.
        let col0 = yTilde[0..., 0]
        let mean0 = col0.mean()
        let std0 = MLX.sqrt(((col0 - mean0) * (col0 - mean0)).mean()).item(Float.self)
        var y = std0 > 0 ? yTilde * (1e-4 / std0) : yTilde * 1e-4
        eval(y)

        // DREAMS = t-SNE-family optimizer + a PCA-regularization gradient hook.
        let yTildeNorm = MLX.sqrt((yTilde * yTilde).sum())
        eval(yTildeNorm)
        let lam = self.lam
        let nF = Float(n)
        log("Starting optimization...")
        y = optimizeTSNEFamily(
            edgeFrom: edgeFrom, edgeTo: edgeTo, edgeWeights: edgeWeights, y: y, n: n,
            learningRate: learningRate, earlyExaggeration: earlyExaggeration,
            earlyExaggerationIter: earlyExaggerationIter, nEpochs: nIter,
            onEpoch: onEpoch, progressEvery: progressEvery, verbose: verbose,
            transformGradient: { grad, y, _ in
                // (1-λ)·grad + (2λ/n)(Y - α·Ỹ),  α = ‖Y‖_F / ‖Ỹ‖_F.
                guard lam > 0 else { return grad }
                let yNorm = MLX.sqrt((y * y).sum())
                let alpha = yNorm / maximum(yTildeNorm, MLXArray(Float(1e-12)))
                let gradReg = (2.0 * lam / nF) * (y - alpha * yTilde)
                return (1.0 - lam) * grad + gradReg
            })
        eval(y)
        self.embedding = y
        return y
    }

}
