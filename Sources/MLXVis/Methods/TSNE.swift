// t-SNE implementation in pure MLX for Apple Silicon.
// Port of mlx_vis/_tsne/tsne.py
//
// Uses a sparse P matrix (KNN), exact repulsive forces (full or chunked all-pairs
// on the GPU), and momentum-based gradient descent with adaptive gains.
//
// NOTE: the Python reference adds an FFT-accelerated repulsive path for n >= 16000
// purely as a speed optimization; it produces the same gradient as the exact path.
// This port uses the exact chunked path for all n (correct, slower at very large n).
// FFT acceleration is a future optimization (see PORTING_NOTES.md).

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

    /// Fit t-SNE and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        let xMx = normalizeInput(x0.asType(.float32), method: normalize)
        let n = xMx.dim(0)
        let dim = xMx.dim(1)

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

        if let randomState { MLXRandom.seed(UInt64(randomState)) }
        // PCA initialization (top nComponents), scaled small.
        var y = pcaReduce(xMx, dim: nComponents) * 1e-4
        eval(y)

        y = optimize(edgeFrom: edgeFrom, edgeTo: edgeTo, edgeWeights: edgeWeights, y: y, n: n)
        eval(y)
        self.embedding = y
        return y
    }

    // MARK: - Optimization

    private func optimize(edgeFrom: MLXArray, edgeTo: MLXArray, edgeWeights: MLXArray, y y0: MLXArray, n: Int) -> MLXArray {
        var y = y0
        let nEpochs = nIter
        let lr = learningRate
        var velocity = MLXArray.zeros(like: y)
        var gains = MLXArray.ones(like: y)
        let minGain: Float = 0.01

        let weightsExag = edgeWeights * earlyExaggeration
        eval(weightsExag)

        // Repulsive force: one module picks FFT / full / chunked once and reuses it.
        let repulsion = TSNERepulsion(n: n, dims: y.dim(1))
        if verbose > 0 && repulsion.usesFFT { print("Using FFT-accelerated repulsive (n=\(n))") }

        if onEpoch != nil { eval(y); onEpoch?(0, nEpochs, y) }  // initial frame
        for epoch in 0..<nEpochs {
            let momentum: Float = epoch < 250 ? 0.5 : 0.8
            let w = epoch < earlyExaggerationIter ? weightsExag : edgeWeights

            // Attractive (sparse).
            let diffA = y[edgeFrom] - y[edgeTo]
            let dsqA = (diffA * diffA).sum(axis: 1, keepDims: true)
            let fAttr = 4.0 * w.expandedDimensions(axis: 1) * diffA / (1.0 + dsqA)
            var grad = MLXArray.zeros(like: y)
            grad = grad.at[edgeFrom].add(fAttr)

            // Repulsive.
            let (z, repGrad) = repulsion(y)
            grad = grad - (4.0 / z) * repGrad

            // Phase transition reset.
            if epoch == earlyExaggerationIter {
                gains = MLXArray.ones(like: y)
                velocity = MLXArray.zeros(like: y)
            }

            // Adaptive gains.
            let inc = (velocity * grad) .< 0.0
            gains = MLX.where(inc, gains + 0.2, gains * 0.8)
            gains = maximum(gains, minGain)

            velocity = momentum * velocity - lr * (gains * grad)
            y = y + velocity
            y = y - y.mean(axis: 0)

            eval(y, velocity, gains)
            if onEpoch != nil && ((epoch + 1) % max(1, progressEvery) == 0 || epoch == nEpochs - 1) {
                onEpoch?(epoch + 1, nEpochs, y)
            }
            if verbose > 0 && (epoch + 1) % verbose == 0 {
                print("Epoch \(epoch + 1)/\(nEpochs)")
            }
        }
        return y
    }
}
