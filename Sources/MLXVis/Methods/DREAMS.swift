// DREAMS dimensionality reduction in pure MLX for Apple Silicon.
// Port of mlx_vis/_dreams/dreams.py
//
// DREAMS (Dimensionality Reduction Enhanced Across Multiple Scales) combines
// t-SNE local structure preservation with PCA global structure preservation via a
// regularization term: L = (1-lam)*L_tsne + (lam/n)*||Y - alpha*Y_tilde||^2.
//
// The optimization core mirrors t-SNE (sparse attractive forces over KNN edges,
// exact all-pairs repulsive forces, momentum GD with adaptive gains) and adds a
// 2D PCA reference embedding Y_tilde plus the regularization gradient each step.
//
// NOTE: the Python reference adds an FFT-accelerated repulsive path for n >= 16000
// purely as a speed optimization producing the same gradient as the exact path.
// This port uses the exact (full / chunked) path for all n, matching the t-SNE port.

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

    /// Fit DREAMS and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        let xMx = normalizeInput(x0.asType(.float32), method: normalize)
        let n = xMx.dim(0)
        let dim = xMx.dim(1)

        // PCA preprocessing (high-dim -> pcaDim) before KNN.
        let xForKNN: MLXArray
        if let pcaDim, dim > pcaDim {
            xForKNN = pcaReduce(xMx, dim: pcaDim)
        } else {
            xForKNN = xMx
        }

        let k = min(Int(3 * perplexity), n - 1)
        let (knnIndices, knnDists) = computeKNN(
            xForKNN, k: k, method: knnMethod,
            returnEuclidean: false, randomState: randomState, verbose: verbose > 0)

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
