// UMAP implementation in pure MLX for Apple Silicon.
// Port of mlx_vis/_umap/umap.py

import Foundation
import MLX
import MLXRandom

/// UMAP dimensionality reduction using MLX on the Metal GPU.
public final class UMAP {
    public var nComponents: Int
    public var nNeighbors: Int
    public var minDist: Float
    public var spread: Float
    public var nEpochs: Int?
    public var learningRate: Float
    public var negativeSampleRate: Int
    public var randomState: Int?
    public var verbose: Bool
    public var pcaDim: Int?
    public var knnMethod: KNNMethod
    public var normalize: Normalization

    public private(set) var embedding: MLXArray?

    /// Optional progress hook, called periodically during `fitTransform` with
    /// `(epoch, totalEpochs, embedding)` so callers can animate the optimization
    /// as it converges. The embedding is `(nSamples, nComponents)` and already
    /// evaluated. No overhead when unset.
    public var onEpoch: ((Int, Int, MLXArray) -> Void)?

    /// How often `onEpoch` fires, in optimizer steps (default 10). Set to 1 for a
    /// frame every step (smoothest animation, more overhead); ignored when `onEpoch`
    /// is unset. Independent of the internal graph-eval cadence (memory safeguard).
    public var progressEvery: Int = 10

    public init(
        nComponents: Int = 2,
        nNeighbors: Int = 15,
        minDist: Float = 0.1,
        spread: Float = 1.0,
        nEpochs: Int? = nil,
        learningRate: Float = 1.0,
        negativeSampleRate: Int = 5,
        randomState: Int? = nil,
        verbose: Bool = false,
        pcaDim: Int? = nil,
        knnMethod: KNNMethod = .auto,
        normalize: Normalization = .none
    ) {
        self.nComponents = nComponents
        self.nNeighbors = nNeighbors
        self.minDist = minDist
        self.spread = spread
        self.nEpochs = nEpochs
        self.learningRate = learningRate
        self.negativeSampleRate = negativeSampleRate
        self.randomState = randomState
        self.verbose = verbose
        self.pcaDim = pcaDim
        self.knnMethod = knnMethod
        self.normalize = normalize
    }

    /// Fit UMAP and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        var x = normalizeInput(x0.asType(.float32), method: normalize)
        let n = x.dim(0)
        let dim = x.dim(1)

        // Optional PCA preprocessing for high-dimensional data.
        let xForKNN: MLXArray
        if let pcaDim, dim > pcaDim {
            if verbose { print("Applying PCA: \(dim) -> \(pcaDim) dims...") }
            xForKNN = pcaReduce(x, dim: pcaDim)
        } else {
            xForKNN = x
        }
        x = xForKNN

        if verbose { print("Computing nearest neighbors...") }
        let (knnIndices, knnDists) = computeKNN(
            x, k: nNeighbors, method: knnMethod,
            returnEuclidean: true, randomState: randomState, verbose: verbose)

        let epochs = nEpochs ?? (n <= 10000 ? 500 : 200)
        self.nEpochs = epochs

        if verbose { print("Building fuzzy simplicial set...") }
        let (rows, cols, vals) = fuzzySimplicialSet(knnIndices, knnDists, n: n, nEpochs: epochs)

        let (a, b) = Self.findABParams(spread: spread, minDist: minDist)

        if verbose { print("Initializing embedding...") }
        if let randomState { MLXRandom.seed(UInt64(randomState)) }
        var y = spectralInit(rows: rows, cols: cols, vals: vals, n: n)
        eval(y)

        y = optimize(edgeFrom: rows, edgeTo: cols, edgeWeights: vals,
                     y: y, a: a, b: b, n: n, nEpochs: epochs)
        eval(y)

        self.embedding = y
        return y
    }

    // MARK: - Fuzzy simplicial set

    private func fuzzySimplicialSet(
        _ knnIndices: MLXArray, _ knnDists0: MLXArray, n: Int, nEpochs: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let k = nNeighbors
        let target = Float(log2(Double(k)))
        let knnDists = knnDists0.asType(.float32)

        // Rho: distance to nearest non-zero neighbor.
        let mask = knnDists .> 0
        let rhosRaw = MLX.where(mask, knnDists, MLXArray(Float.infinity))
        let rhos = maximum(rhosRaw.min(axis: 1), MLXArray(Float(1e-8)))  // (n,)

        let distsShifted = maximum(knnDists - rhos.expandedDimensions(axis: 1), 0.0)
        let distsShiftedTail = distsShifted[0..., 1...]  // skip j=0 (rho contributor)

        // Binary search for sigma (vectorized over all N points).
        var lo = MLXArray.full([n], values: MLXArray(Float(1e-20)))
        var hi = MLXArray.full([n], values: MLXArray(Float(1e3)))
        var sigma = MLXArray.ones([n])

        for _ in 0..<64 {
            let valsExp = MLX.exp(-distsShiftedTail / sigma.expandedDimensions(axis: 1))
            let valsSum = valsExp.sum(axis: 1)

            let converged = abs(valsSum - target) .< 1e-5
            let tooHigh = (valsSum .> target) & (.!converged)
            let tooLow = (valsSum .< target) & (.!converged)

            hi = MLX.where(tooHigh, sigma, hi)
            lo = MLX.where(tooLow, sigma, lo)
            sigma = MLX.where(tooHigh, (lo + sigma) / 2.0, sigma)
            sigma = MLX.where(tooLow, MLX.where(hi .>= 1e3, sigma * 2.0, (sigma + hi) / 2.0), sigma)

            eval(sigma)
            if all(converged).item(Bool.self) { break }
        }

        // Edge weights.
        let weights = MLX.exp(-distsShifted / maximum(sigma.expandedDimensions(axis: 1), 1e-10))

        // Sparse edges (rows from arange*k, cols from knn indices).
        let rowsMx = repeated(MLXArray(Int32(0) ..< Int32(n)), count: k, axis: 0)
        let colsMx = knnIndices.reshaped([-1]).asType(.int32)
        let valsMx = weights.reshaped([-1])

        // Symmetrize on GPU: P = A + A^T - A * A^T via searchsorted key matching.
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

        let wSym = valsMx + wRev - valsMx * wRev

        // Prune weak edges on CPU.
        let threshold = (wSym.max() / Float(nEpochs)).item(Float.self)
        eval(wSym)
        let wSymArr = wSym.asArray(Float.self)
        let rowsArr = rowsMx.asArray(Int32.self)
        let colsArr = colsMx.asArray(Int32.self)

        var fRows: [Int32] = []
        var fCols: [Int32] = []
        var fVals: [Float] = []
        fRows.reserveCapacity(wSymArr.count)
        fCols.reserveCapacity(wSymArr.count)
        fVals.reserveCapacity(wSymArr.count)
        for i in 0..<wSymArr.count where wSymArr[i] >= threshold {
            fRows.append(rowsArr[i])
            fCols.append(colsArr[i])
            fVals.append(wSymArr[i])
        }

        return (MLXArray(fRows), MLXArray(fCols), MLXArray(fVals))
    }

    // MARK: - a, b parameters

    /// Find `a`, `b` via Gauss-Newton optimization (no scipy).
    public static func findABParams(spread: Float, minDist: Float) -> (Float, Float) {
        let count = 300
        let s = Double(spread)
        let md = Double(minDist)
        var xv = [Double](repeating: 0, count: count)
        var yv = [Double](repeating: 0, count: count)
        for i in 0..<count {
            let xi = Double(i) / Double(count - 1) * (s * 3.0)
            xv[i] = xi
            yv[i] = xi < md ? 1.0 : exp(-(xi - md) / s)
        }

        var a = 1.0, b = 1.0
        for _ in 0..<100 {
            // Build 2x2 normal equations J^T J step = J^T(-residual).
            var jtj00 = 0.0, jtj01 = 0.0, jtj11 = 0.0
            var jtr0 = 0.0, jtr1 = 0.0
            for i in 0..<count {
                let x2b = pow(xv[i], 2.0 * b)
                let denom = 1.0 + a * x2b
                let pred = 1.0 / denom
                let residual = pred - yv[i]
                let da = -x2b / (denom * denom)
                let db = -a * 2.0 * log(max(xv[i], 1e-20)) * x2b / (denom * denom)
                jtj00 += da * da
                jtj01 += da * db
                jtj11 += db * db
                jtr0 += da * (-residual)
                jtr1 += db * (-residual)
            }
            let det = jtj00 * jtj11 - jtj01 * jtj01
            if abs(det) < 1e-30 { break }
            let stepA = (jtr0 * jtj11 - jtr1 * jtj01) / det
            let stepB = (jtj00 * jtr1 - jtj01 * jtr0) / det
            a += stepA
            b += stepB
            if stepA * stepA + stepB * stepB < 1e-12 { break }
        }
        return (Float(a), Float(b))
    }

    // MARK: - Spectral initialization

    private func spectralInit(rows: MLXArray, cols: MLXArray, vals: MLXArray, n: Int) -> MLXArray {
        let k = nComponents + 1

        // Degrees via scatter-add, then symmetric normalization.
        var degrees = MLXArray.zeros([n])
        degrees = degrees.at[rows].add(vals)
        degrees = maximum(degrees, 1e-10)
        let dInvSqrt = 1.0 / MLX.sqrt(degrees)
        let wNorm = vals * dInvSqrt[rows] * dInvSqrt[cols]
        eval(wNorm)

        func sparseMatvec(_ xv: MLXArray) -> MLXArray {
            let gathered = wNorm.expandedDimensions(axis: 1) * xv[cols]
            var result = MLXArray.zeros(like: xv)
            result = result.at[rows].add(gathered)
            return result
        }

        var v = MLXRandom.normal([n, k])
        eval(v)

        for _ in 0..<100 {
            v = sparseMatvec(v)
            // Modified Gram-Schmidt.
            for j in 0..<k {
                for i in 0..<j {
                    let proj = (v[0..., j] * v[0..., i]).sum()
                    v = v.at[0..., j].add(-proj * v[0..., i])
                }
                let norm = MLX.sqrt((v[0..., j] * v[0..., j]).sum() + 1e-10)
                v = v.at[0..., j].multiply(1.0 / norm)
            }
            eval(v)
        }

        var embedding = v[0..., 1..<k]

        // Scale to [0, 10] + small noise.
        let expansion = 10.0 / embedding.abs().max()
        embedding = embedding * expansion
        embedding = embedding + MLXRandom.normal(embedding.shape) * 0.0001
        let lo = embedding.min(axis: 0)
        let hi = embedding.max(axis: 0)
        embedding = 10.0 * (embedding - lo) / (hi - lo + 1e-10)
        eval(embedding)
        return embedding
    }

    // MARK: - Optimization

    private func optimize(
        edgeFrom: MLXArray, edgeTo: MLXArray, edgeWeights: MLXArray,
        y y0: MLXArray, a: Float, b: Float, n: Int, nEpochs: Int
    ) -> MLXArray {
        var y = y0
        eval(edgeWeights)
        let weightsArr = edgeWeights.asArray(Float.self)
        let maxWeight = weightsArr.max() ?? 1.0

        // epochs_per_sample scheduling (CPU).
        let e = weightsArr.count
        var epochsPerSample = [Double](repeating: -1.0, count: e)
        for i in 0..<e {
            let nSamples = Double(nEpochs) * Double(weightsArr[i]) / Double(maxWeight)
            epochsPerSample[i] = nSamples > 0 ? Double(nEpochs) / nSamples : -1.0
        }
        var epochsPerNext = epochsPerSample

        // Pre-compute per-epoch active edge index sets on the HOST (cheap Int32
        // arrays). Keeping these host-side — rather than materializing nEpochs
        // MLXArrays up front and holding them on-GPU for the whole optimization —
        // keeps peak GPU memory flat; each epoch's index array is wrapped lazily
        // inside the loop and freed right after the step.
        var activeSets = [[Int32]?](repeating: nil, count: nEpochs)
        for epoch in 0..<nEpochs {
            var active: [Int32] = []
            for i in 0..<e where epochsPerNext[i] <= Double(epoch) {
                epochsPerNext[i] += epochsPerSample[i]
                active.append(Int32(i))
            }
            activeSets[epoch] = active.isEmpty ? nil : active
        }

        let alpha = learningRate

        // NOTE: the per-edge SGD step is intentionally NOT compiled. Its scatter-add
        // updates have a variable-length index (the active-edge set changes every
        // epoch), which `compile(shapeless:)` cannot shape-infer, and non-shapeless
        // would recompile every epoch. It is also cheap relative to the O(n^2)
        // attractive/repulsive work, so compilation buys little here.
        if onEpoch != nil { eval(y); onEpoch?(0, nEpochs, y) }  // initial frame
        for epoch in 0..<nEpochs {
            guard let active = activeSets[epoch] else { continue }
            let activeMx = MLXArray(active)
            let ef = edgeFrom[activeMx]
            let et = edgeTo[activeMx]
            let nActive = activeMx.dim(0)
            let alphaEpoch = alpha * (1.0 - Float(epoch) / Float(nEpochs))

            // Negative sampling: fixed rate per active edge.
            let nNeg = negativeSampleRate * nActive
            let pat = MLXArray(Int32(0) ..< Int32(nNeg))
            let modIdx = pat - floorDivide(pat, MLXArray(Int32(nActive))) * nActive
            let negFrom = ef[modIdx]
            let negTo = MLXRandom.randInt(0 ..< n, [nNeg])

            y = Self.sgdStepGraph(y, ef: ef, et: et, negFrom: negFrom, negTo: negTo,
                                  alphaEpoch: MLXArray(alphaEpoch), a: a, b: b)[0]

            let progressHit = onEpoch != nil && ((epoch + 1) % max(1, progressEvery) == 0 || epoch == nEpochs - 1)
            if (epoch + 1) % 10 == 0 || epoch == nEpochs - 1 || progressHit {
                eval(y)
            }
            if progressHit { onEpoch?(epoch + 1, nEpochs, y) }
            if verbose && (epoch + 1) % 50 == 0 {
                print("Epoch \(epoch + 1)/\(nEpochs)")
            }
        }
        return y
    }

    /// Single SGD step as a pure graph function (compilable). `alphaEpoch` is a
    /// scalar `MLXArray` input; `a`/`b` are baked-in constants.
    private static func sgdStepGraph(
        _ y0: MLXArray, ef: MLXArray, et: MLXArray, negFrom: MLXArray, negTo: MLXArray,
        alphaEpoch: MLXArray, a: Float, b: Float
    ) -> [MLXArray] {
        var y = y0
        // Positive forces (attract).
        let diff = y[ef] - y[et]
        let distSq = maximum((diff * diff).sum(axis: 1, keepDims: true), 1e-6)
        let powVal = MLX.pow(distSq, b)
        let gradCoeff = -2.0 * a * b * MLX.pow(distSq, b - 1.0) / (1.0 + a * powVal)
        let posGrad = clip(gradCoeff * diff, min: -4.0, max: 4.0) * alphaEpoch

        // Negative forces (repel).
        let negDiff = y[negFrom] - y[negTo]
        let negDistSq = maximum((negDiff * negDiff).sum(axis: 1, keepDims: true), 1e-6)
        let negPow = MLX.pow(negDistSq, b)
        let negGradCoeff = 2.0 * b / ((0.001 + negDistSq) * (1.0 + a * negPow))
        let negGrad = clip(negGradCoeff * negDiff, min: -4.0, max: 4.0) * alphaEpoch

        y = y.at[ef].add(posGrad)
        y = y.at[et].add(-posGrad)
        y = y.at[negFrom].add(negGrad)
        return [y]
    }
}
