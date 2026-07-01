// t-SNE implementation in pure MLX for Apple Silicon.
// Port of mlx_vis/_tsne/tsne.py
//
// Uses a sparse P matrix (KNN) and momentum-based gradient descent with adaptive
// gains. The repulsive force, P-matrix, and optimizer loop are shared with DREAMS
// via Core/{AffinityGraph,Repulsion,TSNEOptimizer}.swift; ``TSNERepulsion`` picks
// the FFT (n >= 16000, 2D), full, or chunked path automatically.

import Foundation
import MLX
import MLXRandom

/// t-SNE dimensionality reduction using MLX on the Metal GPU.
public final class TSNE {
    public var nComponents: Int
    public var perplexity: Float
    public var learningRate: Float
    public var nIter: Int
    public var earlyExaggeration: Float
    public var earlyExaggerationIter: Int
    public var randomState: Int?
    public var verbose: Int
    public var pcaDim: Int?
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
    /// (KNN build, affinity-graph construction, optimization start) with a short
    /// human-readable label. Lets callers surface the otherwise-silent
    /// pre-optimization work to a UI. Fires regardless of `verbose`; no-op when unset.
    public var onPhase: ((String) -> Void)?

    public init(
        nComponents: Int = 2,
        perplexity: Float = 30.0,
        learningRate: Float = 200.0,
        nIter: Int = 1000,
        earlyExaggeration: Float = 12.0,
        earlyExaggerationIter: Int = 250,
        randomState: Int? = nil,
        verbose: Int = 0,
        pcaDim: Int? = 50,
        knnMethod: KNNMethod = .auto,
        normalize: Normalization = .none
    ) {
        self.nComponents = nComponents
        self.perplexity = perplexity
        self.learningRate = learningRate
        self.nIter = nIter
        self.earlyExaggeration = earlyExaggeration
        self.earlyExaggerationIter = earlyExaggerationIter
        self.randomState = randomState
        self.verbose = verbose
        self.pcaDim = pcaDim
        self.knnMethod = knnMethod
        self.normalize = normalize
    }

    private func log(_ msg: @autoclosure () -> String) {
        guard verbose > 0 || onPhase != nil else { return }
        let s = msg()
        if verbose > 0 { print(s) }
        onPhase?(s)
    }

    /// Fit t-SNE and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        let xMx = normalizeInput(x0.asType(.float32), method: normalize)
        let n = xMx.dim(0)
        let dim = xMx.dim(1)

        let xForKNN: MLXArray
        if let pcaDim, dim > pcaDim {
            log("Applying PCA: \(dim) -> \(pcaDim) dims...")
            xForKNN = pcaReduce(xMx, dim: pcaDim)
        } else {
            xForKNN = xMx
        }

        let k = min(Int(3 * perplexity), n - 1)
        log("Computing k-NN...")
        let (knnIndices, knnDists) = computeKNN(
            xForKNN, k: k, method: knnMethod,
            returnEuclidean: false, randomState: randomState, verbose: verbose > 0)

        log("Building affinity graph...")
        let (edgeFrom, edgeTo, edgeWeights) = buildPMatrix(knnIndices, knnDists, perplexity: perplexity, n: n)
        eval(edgeFrom, edgeTo, edgeWeights)

        if let randomState { MLXRandom.seed(UInt64(randomState)) }
        // PCA initialization (top nComponents), scaled small.
        var y = pcaReduce(xMx, dim: nComponents) * 1e-4
        eval(y)

        log("Starting optimization...")
        y = optimizeTSNEFamily(
            edgeFrom: edgeFrom, edgeTo: edgeTo, edgeWeights: edgeWeights, y: y, n: n,
            learningRate: learningRate, earlyExaggeration: earlyExaggeration,
            earlyExaggerationIter: earlyExaggerationIter, nEpochs: nIter,
            onEpoch: onEpoch, progressEvery: progressEvery, verbose: verbose)
        eval(y)
        self.embedding = y
        return y
    }

}
