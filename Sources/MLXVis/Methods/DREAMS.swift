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
    /// optimization. No-op when unset.
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

        let (edgeFrom, edgeTo, edgeWeights) = buildP(knnIndices, knnDists, n: n)
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

        y = optimize(edgeFrom: edgeFrom, edgeTo: edgeTo, edgeWeights: edgeWeights,
                     y: y, n: n, yTilde: yTilde)
        eval(y)
        self.embedding = y
        return y
    }

    // MARK: - P matrix (identical to t-SNE)

    private func buildP(_ knnIndices: MLXArray, _ knnDists0: MLXArray, n: Int) -> (MLXArray, MLXArray, MLXArray) {
        let k = knnIndices.dim(1)

        let knnDists = knnDists0.asType(.float32)
        let sqDists = knnDists * knnDists

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

        let logits = -beta.expandedDimensions(axis: 1) * sqDists
        let logitsMax = logits.max(axis: 1, keepDims: true)
        let expL = MLX.exp(logits - logitsMax)
        let sumExp = expL.sum(axis: 1, keepDims: true)
        let weights = expL / sumExp

        let rowsMx = repeated(MLXArray(Int32(0) ..< Int32(n)), count: k, axis: 0)
        let colsMx = knnIndices.reshaped([-1]).asType(.int32)
        let valsMx = weights.reshaped([-1])

        let nL = Int64(n)
        let fwdKeys = rowsMx.asType(.int64) * nL + colsMx.asType(.int64)
        let revKeys = colsMx.asType(.int64) * nL + rowsMx.asType(.int64)
        let sortIdx = argSort(fwdKeys)
        let sortedKeys = fwdKeys[sortIdx]
        let sortedVals = valsMx[sortIdx]
        var pos = searchSorted(sortedKeys, revKeys)
        pos = minimum(pos, sortedKeys.dim(0) - 1)
        let matched = sortedKeys[pos] .== revKeys
        let wRev = MLX.where(matched, sortedVals[pos], 0.0)
        let wSym = (valsMx + wRev) / (2.0 * Float(n))

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

    // MARK: - Repulsive gradient

    private func repulsiveFull(_ y: MLXArray, eyeMask: MLXArray) -> (MLXArray, MLXArray) {
        let sqNorms = (y * y).sum(axis: 1)
        let dsq = maximum(
            sqNorms.expandedDimensions(axis: 1) + sqNorms.expandedDimensions(axis: 0)
                - 2.0 * y.matmul(y.transposed()), 0.0)
        let kernel = eyeMask / (1.0 + dsq)
        let z = kernel.sum()
        let ksq = kernel * kernel
        let repGrad = ksq.sum(axis: 1, keepDims: true) * y - ksq.matmul(y)
        return (z, repGrad)
    }

    private func repulsiveChunked(_ y: MLXArray, n: Int, chunkSize: Int, selfMasks: [MLXArray]) -> (MLXArray, MLXArray) {
        var z = MLXArray(Float(0))
        var repGrad = MLXArray.zeros(like: y)
        let sqNorms = (y * y).sum(axis: 1)

        var i = 0
        var start = 0
        while start < n {
            let end = min(start + chunkSize, n)
            let yChunk = y[start ..< end]
            let dsq = maximum(
                sqNorms[start ..< end].expandedDimensions(axis: 1) + sqNorms.expandedDimensions(axis: 0)
                    - 2.0 * yChunk.matmul(y.transposed()), 0.0)
            let kernel = selfMasks[i] / (1.0 + dsq)
            let zChunk = kernel.sum()
            let ksq = kernel * kernel
            let rg = ksq.sum(axis: 1, keepDims: true) * yChunk - ksq.matmul(y)
            z = z + zChunk
            repGrad = repGrad.at[start ..< end].add(rg)
            eval(z, repGrad)
            i += 1
            start = end
        }
        return (z, repGrad)
    }

    // MARK: - Optimization

    private func optimize(edgeFrom: MLXArray, edgeTo: MLXArray, edgeWeights: MLXArray,
                          y y0: MLXArray, n: Int, yTilde: MLXArray) -> MLXArray {
        var y = y0
        let nEpochs = nIter
        let lr = learningRate
        let lam = self.lam
        var velocity = MLXArray.zeros(like: y)
        var gains = MLXArray.ones(like: y)
        let minGain: Float = 0.01

        let weightsExag = edgeWeights * earlyExaggeration
        eval(weightsExag)

        // Frobenius norm of Y_tilde (constant across epochs).
        let yTildeNorm = MLX.sqrt((yTilde * yTilde).sum())
        eval(yTildeNorm)

        // FFT-accelerated repulsive for large 2D embeddings; exact full/chunked otherwise.
        let useFFT = n >= 16000 && y.dim(1) == 2
        let fullLimit = 1_000_000_000
        let useFull = !useFFT && (n * n * 4) < fullLimit
        let chunkSize = min(n, max(512, 2_000_000_000 / (n * 4)))

        var eyeMask = MLXArray.zeros([1])
        var selfMasks: [MLXArray] = []
        if useFFT {
            if verbose > 0 { print("Using FFT-accelerated repulsive (n=\(n))") }
        } else if useFull {
            eyeMask = 1.0 - MLXArray.eye(n)
            eval(eyeMask)
        } else {
            var start = 0
            while start < n {
                let end = min(start + chunkSize, n)
                let mask = 1.0 - (arangeColumn(start, end) .== arangeRow(n)).asType(.float32)
                eval(mask)
                selfMasks.append(mask)
                start = end
            }
        }

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
            let z: MLXArray
            let repGrad: MLXArray
            if useFFT {
                (z, repGrad) = fftRepulsiveGrad(y, n: n)
            } else if useFull {
                (z, repGrad) = repulsiveFull(y, eyeMask: eyeMask)
            } else {
                (z, repGrad) = repulsiveChunked(y, n: n, chunkSize: chunkSize, selfMasks: selfMasks)
            }
            grad = grad - (4.0 / z) * repGrad

            // DREAMS regularization: (2*lam/n) * (Y - alpha*Y_tilde), alpha = ||Y||_F / ||Y_tilde||_F.
            if lam > 0 {
                let yNorm = MLX.sqrt((y * y).sum())
                let alpha = yNorm / maximum(yTildeNorm, MLXArray(Float(1e-12)))
                let gradReg = (2.0 * lam / Float(n)) * (y - alpha * yTilde)
                grad = (1.0 - lam) * grad + gradReg
            }

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
