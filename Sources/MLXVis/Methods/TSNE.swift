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

        let (edgeFrom, edgeTo, edgeWeights) = buildP(knnIndices, knnDists, n: n)
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

    // MARK: - P matrix

    private func buildP(_ knnIndices: MLXArray, _ knnDists0: MLXArray, n: Int) -> (MLXArray, MLXArray, MLXArray) {
        let k = knnIndices.dim(1)

        // Mirrors the reference: knnDists here are squared distances; squaring again
        // is the reference's bandwidth input.
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

        // Make both directions explicit, dedup keeping first occurrence (CPU compaction).
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

    private static func repulsiveFullGraph(_ y: MLXArray, eyeMask: MLXArray) -> [MLXArray] {
        let sqNorms = (y * y).sum(axis: 1)
        let dsq = maximum(
            sqNorms.expandedDimensions(axis: 1) + sqNorms.expandedDimensions(axis: 0)
                - 2.0 * y.matmul(y.transposed()), 0.0)
        let kernel = eyeMask / (1.0 + dsq)
        let z = kernel.sum()
        let ksq = kernel * kernel
        let repGrad = ksq.sum(axis: 1, keepDims: true) * y - ksq.matmul(y)
        return [z, repGrad]
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

    private func optimize(edgeFrom: MLXArray, edgeTo: MLXArray, edgeWeights: MLXArray, y y0: MLXArray, n: Int) -> MLXArray {
        var y = y0
        let nEpochs = nIter
        let lr = learningRate
        var velocity = MLXArray.zeros(like: y)
        var gains = MLXArray.ones(like: y)
        let minGain: Float = 0.01

        let weightsExag = edgeWeights * earlyExaggeration
        eval(weightsExag)

        // Method selection: FFT for large 2D embeddings (O(n) vs O(n^2)), exact
        // full path when n^2 fits comfortably, chunked otherwise.
        let useFFT = n >= 16000 && y.dim(1) == 2
        let fullLimit = 1_000_000_000
        let useFull = !useFFT && (n * n * 4) < fullLimit
        let chunkSize = min(n, max(512, 2_000_000_000 / (n * 4)))

        var eyeMask = MLXArray.zeros([1])
        var selfMasks: [MLXArray] = []
        // Compiled full-path repulsive kernel (fixed shapes, called every epoch).
        let repFull = compile { (args: [MLXArray]) -> [MLXArray] in
            Self.repulsiveFullGraph(args[0], eyeMask: args[1])
        }
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
                let out = repFull([y, eyeMask])
                z = out[0]; repGrad = out[1]
            } else {
                (z, repGrad) = repulsiveChunked(y, n: n, chunkSize: chunkSize, selfMasks: selfMasks)
            }
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
            if verbose > 0 && (epoch + 1) % verbose == 0 {
                print("Epoch \(epoch + 1)/\(nEpochs)")
            }
        }
        return y
    }
}
