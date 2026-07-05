// PaCMAP and LocalMAP implementation in pure MLX for Apple Silicon.
// Port of mlx_vis/_pacmap/pacmap.py
//
// LocalMAP subclasses PaCMAP and swaps the final optimization phase for a
// locally-adjusted attractive step plus periodic resampling of further pairs
// using embedding-space locality.

import Foundation
import MLX
import MLXRandom

/// PaCMAP dimensionality reduction using MLX on the Metal GPU.
public class PaCMAP {
    public var nComponents: Int
    public var nNeighbors: Int?
    public var mnRatio: Float
    public var fpRatio: Float
    public var learningRate: Float
    /// (phase1, phase2, phase3) iteration counts.
    public var numIters: (Int, Int, Int)
    public var randomState: Int?
    public var verbose: Bool
    public var applyPCA: Bool
    public var knnMethod: KNNMethod
    public var normalize: Normalization

    public private(set) var embedding: MLXArray?

    /// Optional progress hook, called periodically during `fitTransform` with
    /// `(iteration, totalIterations, embedding)` so callers can animate the
    /// optimization. Fires once at iteration 0 with the initial embedding, then on the `progressEvery` schedule. Inherited by `LocalMAP`. No-op when unset.
    public var onEpoch: ((Int, Int, MLXArray) -> Void)?

    /// How often `onEpoch` fires, in optimizer steps (default 10). Set to 1 for a
    /// frame every step (smoothest animation, more overhead); ignored when `onEpoch`
    /// is unset. Independent of the internal graph-eval cadence (memory safeguard).
    public var progressEvery: Int = 10

    /// Optional phase hook, called at each setup milestone during `fitTransform`
    /// (PCA preprocess, KNN build, pair sampling, optimization start) with a short
    /// human-readable label. Lets callers surface the otherwise-silent
    /// pre-optimization work — which dominates the wall clock for large inputs — to
    /// a UI. Fires regardless of `verbose`; no-op when unset. Inherited by `LocalMAP`.
    public var onPhase: ((String) -> Void)?

    // Set during preprocessing: true if PCA reduction to 100 dims was applied.
    fileprivate var pcaSolution = false

    public init(
        nComponents: Int = 2,
        nNeighbors: Int? = 10,
        mnRatio: Float = 0.5,
        fpRatio: Float = 2.0,
        learningRate: Float = 1.0,
        numIters: (Int, Int, Int) = (100, 100, 250),
        randomState: Int? = nil,
        verbose: Bool = false,
        applyPCA: Bool = true,
        knnMethod: KNNMethod = .auto,
        normalize: Normalization = .none
    ) {
        self.nComponents = nComponents
        self.nNeighbors = nNeighbors
        self.mnRatio = mnRatio
        self.fpRatio = fpRatio
        self.learningRate = learningRate
        self.numIters = numIters
        self.randomState = randomState
        self.verbose = verbose
        self.applyPCA = applyPCA
        self.knnMethod = knnMethod
        self.normalize = normalize
    }

    private func log(_ msg: @autoclosure () -> String) {
        guard verbose || onPhase != nil else { return }
        let s = msg()
        if verbose { print(s) }
        onPhase?(s)
    }

    /// Fit PaCMAP and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        return fitTransformImpl(x0, lowDistThres: nil)
    }

    // MARK: - Preprocess

    /// If dim > 100 and PCA enabled, reduce to 100 dims (descending eigenvector order).
    /// Otherwise apply the reference's scalar min/max + per-feature centering.
    private func preprocess(_ x: MLXArray) -> MLXArray {
        let dim = x.dim(1)
        if dim > 100 && applyPCA {
            pcaSolution = true
            let reduced = pcaReduce(x, dim: 100)
            eval(reduced)
            log("Applied PCA, dimensionality reduced to 100")
            return reduced
        }

        pcaSolution = false
        // Scalar min/max (global, matching the reference's np.min/np.max with no axis).
        var xp = x - x.min()
        xp = xp / xp.max()
        xp = xp - xp.mean(axis: 0, keepDims: true)
        eval(xp)
        log("X normalised")
        return xp
    }

    // MARK: - Pair counts

    private func decideNumPairs(_ n: Int) -> (Int, Int, Int) {
        var nNeighborsLocal: Int
        if let nn = nNeighbors {
            nNeighborsLocal = nn
        } else {
            if n <= 10000 {
                nNeighborsLocal = 10
            } else {
                nNeighborsLocal = Int((10.0 + 15.0 * (log10(Double(n)) - 4.0)).rounded())
            }
        }

        var nMN = Int((Double(nNeighborsLocal) * Double(mnRatio)).rounded())
        var nFP = Int((Double(nNeighborsLocal) * Double(fpRatio)).rounded())

        nNeighborsLocal = min(nNeighborsLocal, n - 1)
        nMN = min(nMN, n - 1)
        nFP = min(nFP, max(0, n - 1 - nNeighborsLocal))

        if nNeighborsLocal + nMN + nFP >= n {
            log("Sample size is smaller than total assigned points; reorganising n_neighbors, n_MN, n_FP")
            let denom = 1.0 + Double(mnRatio) + Double(fpRatio)
            nNeighborsLocal = max(1, Int(Double(n) / denom))
            nMN = Int(Double(nNeighborsLocal) * Double(mnRatio))
            nFP = Int(Double(nNeighborsLocal) * Double(fpRatio))
            nNeighborsLocal = min(nNeighborsLocal, n - 1)
            nMN = min(nMN, n - 1)
            nFP = min(nFP, max(0, n - 1 - nNeighborsLocal))
        }

        precondition(nNeighborsLocal >= 1, "The number of nearest neighbours can't be less than 1")
        return (nNeighborsLocal, nMN, nFP)
    }

    // MARK: - Pair sampling

    /// Sample neighbour pairs using scaled distance (sigma = mean of 4th-6th neighbours).
    /// Returns flattened src/dst int32 arrays of length n*nNeighbors.
    private func sampleNeighbors(
        knnDistances: MLXArray, knnIndices: MLXArray, n: Int, nNeighbors: Int
    ) -> (MLXArray, MLXArray) {
        // sigma: mean of distances to 4th-6th neighbours (columns 3..5), clamped.
        let kExtra = knnDistances.dim(1)
        let hiCol = min(6, kExtra)
        let loCol = min(3, hiCol)
        let sig: MLXArray
        if hiCol > loCol {
            sig = maximum(knnDistances[0..., loCol ..< hiCol].mean(axis: 1), 1e-10)  // (n,)
        } else {
            sig = MLXArray.full([n], values: MLXArray(Float(1e-10)))
        }

        let sigNb = sig[knnIndices]  // (n, kExtra)
        let scaled = (knnDistances * knnDistances) / (sig.expandedDimensions(axis: 1) * sigNb)

        let sortedIdx = argSort(scaled, axis: 1)[0..., ..<nNeighbors]  // (n, nNeighbors)
        let picked = takeAlong(knnIndices, sortedIdx, axis: 1)         // (n, nNeighbors)

        let src = repeated(MLXArray(Int32(0) ..< Int32(n)), count: nNeighbors, axis: 0)
        let dst = picked.reshaped([-1]).asType(.int32)
        eval(src, dst)
        return (src, dst)
    }

    /// Sample mid-near pairs: for each point, draw 6 random candidates and pick the
    /// 2nd-closest. Repeated nMN times. Returns flattened src/dst (length n*nMN).
    private func sampleMNPairs(_ x: MLXArray, nMN: Int, n: Int) -> (MLXArray, MLXArray) {
        if nMN == 0 {
            return (MLXArray([] as [Int32]), MLXArray([] as [Int32]))
        }

        var pickedParts: [MLXArray] = []
        let idx = MLXArray(Int32(0) ..< Int32(n)).expandedDimensions(axis: 1)  // (n, 1)

        for _ in 0..<nMN {
            // Random candidates avoiding self via (idx + offset) % n, offset in [1, n-1].
            let offsets = MLXRandom.randInt(1 ..< Int32(n), [n, 6]).asType(.int32)
            let candidates = (idx + offsets) % Int32(n)  // (n, 6)

            let candFlat = candidates.reshaped([-1])
            let xCand = x[candFlat].reshaped([n, 6, -1])         // (n, 6, dim)
            let diffs = xCand - x.expandedDimensions(axis: 1)    // (n, 6, dim)
            let dists = (diffs * diffs).sum(axis: 2)             // (n, 6)

            let sortedIdx = argSort(dists, axis: 1)              // (n, 6)
            let picked = takeAlong(candidates, sortedIdx[0..., 1..<2], axis: 1).squeezed(axis: 1)  // (n,)
            pickedParts.append(picked.asType(.int32))
        }
        eval(pickedParts)

        // src is arange(n) repeated per round (round-major to match Python's j*n layout).
        let srcOne = MLXArray(Int32(0) ..< Int32(n))
        let src = concatenated(Array(repeating: srcOne, count: nMN), axis: 0)
        let dst = concatenated(pickedParts, axis: 0)
        eval(src, dst)
        return (src, dst)
    }

    /// Sample further pairs (repulsive), excluding self and the sampled nearest neighbours.
    ///
    /// Fully vectorized on GPU (mirrors ``resampleLocalFPPairs``): draw an oversampled
    /// `(n, S)` candidate block in one shot, mask self + the point's `nNeighbors`
    /// neighbours, then keep the first `nFP` valid per row via a valid-first argsort.
    /// Replaces the previous per-point CPU rejection loop, which issued a GPU roundtrip
    /// per point (the dominant cost of a large-n PaCMAP/LocalMAP fit). `neighborDst` is
    /// the flattened nn dst array (length `n*nNeighbors`), reshaped to `(n, nNeighbors)`.
    ///
    /// Only ~`nNeighbors + 1` of `n` values are excluded, so `S = 2·nFP + nNeighbors + 8`
    /// yields ≥ `nFP` valid candidates per row with overwhelming probability; as with
    /// the reference's random sampling, an occasional duplicate far-pair is harmless.
    private func sampleFPPairs(
        n: Int, neighborDst: MLXArray?, nNeighbors: Int, nFP: Int
    ) -> (MLXArray, MLXArray) {
        if nFP == 0 {
            return (MLXArray([] as [Int32]), MLXArray([] as [Int32]))
        }

        let sampleSize = max(2 * nFP + nNeighbors + 8, 32)
        let candidates = MLXRandom.randInt(0 ..< Int32(n), [n, sampleSize]).asType(.int32)  // (n, S)
        let srcIds = MLXArray(Int32(0) ..< Int32(n)).reshaped([n, 1])
        var mask = candidates .!= srcIds
        if let nbr = neighborDst, nNeighbors > 0 {
            let nbCols = nbr.reshaped([n, nNeighbors])  // (n, K)
            for k in 0..<nNeighbors {
                mask = mask & (candidates .!= nbCols[0..., k ..< (k + 1)])
            }
        }

        // Valid candidates sort to the front (invalid → key = sampleSize); take nFP.
        let colIdx = broadcast(
            MLXArray(Int32(0) ..< Int32(sampleSize)).reshaped([1, sampleSize]), to: [n, sampleSize])
        let sortKey = MLX.where(mask, colIdx, MLXArray(Int32(sampleSize)))
        let order = argSort(sortKey, axis: 1)
        let dst = takeAlong(candidates, order, axis: 1)[0..., ..<nFP]  // (n, nFP)
        let src = broadcast(srcIds, to: [n, nFP])
        eval(src, dst)
        return (src.reshaped([-1]), dst.reshaped([-1]))
    }

    /// Resample further pairs using embedding-space locality (LocalMAP, pure GPU).
    /// Returns updated flattened src/dst (src unchanged).
    private func resampleLocalFPPairs(
        neighborDst: MLXArray?, fpSrc: MLXArray, fpDst: MLXArray,
        y: MLXArray, lowDistThres: Float, n: Int, nFP: Int, nNeighbors: Int
    ) -> (MLXArray, MLXArray) {
        if nFP == 0 || lowDistThres <= 0 { return (fpSrc, fpDst) }

        let thresholdSq = lowDistThres * lowDistThres
        let sampleSize = max(nFP * 8, 64)

        let candidates = MLXRandom.randInt(0 ..< Int32(n), [n, sampleSize]).asType(.int32)  // (n, S)
        let srcIds = MLXArray(Int32(0) ..< Int32(n)).reshaped([n, 1])
        var mask = candidates .!= srcIds

        if let nbrDst = neighborDst, nNeighbors > 0 {
            let nbCols = nbrDst.reshaped([n, nNeighbors])  // (n, K)
            for k in 0..<nNeighbors {
                mask = mask & (candidates .!= nbCols[0..., k ..< (k + 1)])
            }
        }

        let diff = y[candidates] - y[srcIds]      // (n, S, dim)
        let distSq = (diff * diff).sum(axis: 2)   // (n, S)
        mask = mask & (distSq .<= thresholdSq)

        // Sort valid candidates first per row (invalid → key = sampleSize).
        let colIdx = broadcast(
            MLXArray(Int32(0) ..< Int32(sampleSize)).reshaped([1, sampleSize]),
            to: [n, sampleSize])
        let sortKey = MLX.where(mask, colIdx, MLXArray(Int32(sampleSize)))
        let order = argSort(sortKey, axis: 1)

        let sortedCands = takeAlong(candidates, order, axis: 1)[0..., ..<nFP]  // (n, nFP)
        let sortedValid = takeAlong(mask, order, axis: 1)[0..., ..<nFP]        // (n, nFP)

        let existingDst = fpDst.reshaped([n, nFP])
        let resultDst = MLX.where(sortedValid, sortedCands, existingDst).reshaped([-1]).asType(.int32)
        eval(resultDst)
        return (fpSrc, resultDst)
    }

    // MARK: - Optimization steps

    /// One Adam step with NN (attract), MN (attract), FP (repel) pairs.
    private func stepWithMN(
        y y0: MLXArray, m m0: MLXArray, v v0: MLXArray,
        srcNN: MLXArray, dstNN: MLXArray, srcMN: MLXArray, dstMN: MLXArray,
        srcFP: MLXArray, dstFP: MLXArray,
        wNb: Float, wMN: Float, wFP: Float, lrT: Float
    ) -> (MLXArray, MLXArray, MLXArray) {
        var y = y0
        var grad = MLXArray.zeros(like: y)

        // NN attractive.
        var diff = y[srcNN] - y[dstNN]
        var d = (diff * diff).sum(axis: 1, keepDims: true) + 1.0
        var g = (wNb * 20.0 / ((10.0 + d) * (10.0 + d))) * diff
        grad = grad.at[srcNN].add(g)
        grad = grad.at[dstNN].add(-g)

        // MN attractive.
        if srcMN.dim(0) > 0 {
            diff = y[srcMN] - y[dstMN]
            d = (diff * diff).sum(axis: 1, keepDims: true) + 1.0
            g = (wMN * 20000.0 / ((10000.0 + d) * (10000.0 + d))) * diff
            grad = grad.at[srcMN].add(g)
            grad = grad.at[dstMN].add(-g)
        }

        // FP repulsive.
        if srcFP.dim(0) > 0 {
            diff = y[srcFP] - y[dstFP]
            d = (diff * diff).sum(axis: 1, keepDims: true) + 1.0
            g = (wFP * 2.0 / ((1.0 + d) * (1.0 + d))) * diff
            grad = grad.at[srcFP].add(-g)
            grad = grad.at[dstFP].add(g)
        }

        let m = 0.9 * m0 + 0.1 * grad
        let v = 0.999 * v0 + 0.001 * (grad * grad)
        y = y - lrT * m / (MLX.sqrt(v) + 1e-7)
        return (y, m, v)
    }

    /// One Adam step with NN (attract) and FP (repel) only.
    private func stepNoMN(
        y y0: MLXArray, m m0: MLXArray, v v0: MLXArray,
        srcNN: MLXArray, dstNN: MLXArray, srcFP: MLXArray, dstFP: MLXArray,
        wNb: Float, wFP: Float, lrT: Float
    ) -> (MLXArray, MLXArray, MLXArray) {
        var y = y0
        var grad = MLXArray.zeros(like: y)

        var diff = y[srcNN] - y[dstNN]
        var d = (diff * diff).sum(axis: 1, keepDims: true) + 1.0
        var g = (wNb * 20.0 / ((10.0 + d) * (10.0 + d))) * diff
        grad = grad.at[srcNN].add(g)
        grad = grad.at[dstNN].add(-g)

        if srcFP.dim(0) > 0 {
            diff = y[srcFP] - y[dstFP]
            d = (diff * diff).sum(axis: 1, keepDims: true) + 1.0
            g = (wFP * 2.0 / ((1.0 + d) * (1.0 + d))) * diff
            grad = grad.at[srcFP].add(-g)
            grad = grad.at[dstFP].add(g)
        }

        let m = 0.9 * m0 + 0.1 * grad
        let v = 0.999 * v0 + 0.001 * (grad * grad)
        y = y - lrT * m / (MLX.sqrt(v) + 1e-7)
        return (y, m, v)
    }

    /// One LocalMAP Adam step: NN attractive with local distance weighting, FP repulsive.
    private func stepNoMNLocal(
        y y0: MLXArray, m m0: MLXArray, v v0: MLXArray,
        srcNN: MLXArray, dstNN: MLXArray, srcFP: MLXArray, dstFP: MLXArray,
        wNb: Float, wFP: Float, lrT: Float, nnScale: Float
    ) -> (MLXArray, MLXArray, MLXArray) {
        var y = y0
        var grad = MLXArray.zeros(like: y)

        var diff = y[srcNN] - y[dstNN]
        var d = (diff * diff).sum(axis: 1, keepDims: true) + 1.0
        var g = (wNb * 20.0 * nnScale / (((10.0 + d) * (10.0 + d)) * MLX.sqrt(d))) * diff
        grad = grad.at[srcNN].add(g)
        grad = grad.at[dstNN].add(-g)

        if srcFP.dim(0) > 0 {
            diff = y[srcFP] - y[dstFP]
            d = (diff * diff).sum(axis: 1, keepDims: true) + 1.0
            g = (wFP * 2.0 / ((1.0 + d) * (1.0 + d))) * diff
            grad = grad.at[srcFP].add(-g)
            grad = grad.at[dstFP].add(g)
        }

        let m = 0.9 * m0 + 0.1 * grad
        let v = 0.999 * v0 + 0.001 * (grad * grad)
        y = y - lrT * m / (MLX.sqrt(v) + 1e-7)
        return (y, m, v)
    }

    // MARK: - Shared fit/transform

    /// Shared implementation for PaCMAP and LocalMAP. `lowDistThres != nil` enables LocalMAP.
    fileprivate func fitTransformImpl(_ x0: MLXArray, lowDistThres: Float?) -> MLXArray {
        let x = normalizeInput(x0.asType(.float32), method: normalize)
        let n = x.dim(0)
        precondition(n > 0, "The sample size must be larger than 0")
        if let rs = randomState { MLXRandom.seed(UInt64(rs)) }
        let localmap = lowDistThres != nil

        // Preprocess (PCA reduction or scalar normalization).
        let xProc = preprocess(x)

        let (nNeighborsLocal, nMN, nFP) = decideNumPairs(n)
        log("n=\(n), dim=\(xProc.dim(1)), n_neighbors=\(nNeighborsLocal), n_MN=\(nMN), n_FP=\(nFP)")

        // KNN (extra neighbours for scaled-distance reselection).
        log("Computing KNN...")
        let knnK = min(n - 1, nNeighborsLocal + 50)
        let (knnIndices, knnDistances) = computeKNN(
            xProc, k: knnK, method: knnMethod,
            returnEuclidean: true, randomState: randomState, verbose: verbose)

        // Sample pairs.
        log("Sampling neighbour pairs...")
        let (srcNN, dstNN) = sampleNeighbors(
            knnDistances: knnDistances, knnIndices: knnIndices, n: n, nNeighbors: nNeighborsLocal)

        log("Sampling MN pairs...")
        let (srcMN, dstMN) = sampleMNPairs(xProc, nMN: nMN, n: n)

        log("Sampling FP pairs...")
        let neighborDstForFP: MLXArray? = nNeighborsLocal > 0 ? dstNN : nil
        var (srcFP, dstFP) = sampleFPPairs(
            n: n, neighborDst: neighborDstForFP, nNeighbors: nNeighborsLocal, nFP: nFP)
        eval(srcFP, dstFP)

        // Initialise embedding (PCA init scaled by 0.01).
        var y: MLXArray
        if pcaSolution {
            y = xProc[0..., ..<nComponents] * 0.01
        } else {
            y = pcaInit(xProc, nComponents: nComponents)
        }
        eval(y)

        // Adam state.
        var m = MLXArray.zeros(like: y)
        var v = MLXArray.zeros(like: y)
        let beta1 = 0.9
        let beta2 = 0.999
        let lr = learningRate

        let (phase1, phase2, phase3) = numIters
        let numItersTotal = phase1 + phase2 + phase3
        let wMNInit: Float = 1000.0
        let localNNScale: Float = localmap ? (lowDistThres! / 2.0) : 0.0
        let neighborDstMx: MLXArray? = nNeighborsLocal > 0 ? dstNN : nil

        log("Starting optimisation...")
        if onEpoch != nil { eval(y); onEpoch?(0, numItersTotal, y) }  // initial frame
        for itr in 0..<numItersTotal {
            let lrT = Float(
                Double(lr) * sqrt(1.0 - pow(beta2, Double(itr + 1)))
                    / (1.0 - pow(beta1, Double(itr + 1))))

            if itr < phase1 {
                let wMN = (1.0 - Float(itr) / Float(phase1)) * wMNInit
                    + (Float(itr) / Float(phase1)) * 3.0
                (y, m, v) = stepWithMN(
                    y: y, m: m, v: v,
                    srcNN: srcNN, dstNN: dstNN, srcMN: srcMN, dstMN: dstMN,
                    srcFP: srcFP, dstFP: dstFP,
                    wNb: 2.0, wMN: wMN, wFP: 1.0, lrT: lrT)
            } else if itr < phase1 + phase2 {
                (y, m, v) = stepWithMN(
                    y: y, m: m, v: v,
                    srcNN: srcNN, dstNN: dstNN, srcMN: srcMN, dstMN: dstMN,
                    srcFP: srcFP, dstFP: dstFP,
                    wNb: 3.0, wMN: 3.0, wFP: 1.0, lrT: lrT)
            } else if localmap {
                (y, m, v) = stepNoMNLocal(
                    y: y, m: m, v: v,
                    srcNN: srcNN, dstNN: dstNN, srcFP: srcFP, dstFP: dstFP,
                    wNb: 1.0, wFP: 1.0, lrT: lrT, nnScale: localNNScale)
            } else {
                (y, m, v) = stepNoMN(
                    y: y, m: m, v: v,
                    srcNN: srcNN, dstNN: dstNN, srcFP: srcFP, dstFP: dstFP,
                    wNb: 1.0, wFP: 1.0, lrT: lrT)
            }

            if localmap, let ldt = lowDistThres, itr > phase1 + phase2, itr % 10 == 0, nFP > 0 {
                eval(y, m, v)
                (srcFP, dstFP) = resampleLocalFPPairs(
                    neighborDst: neighborDstMx, fpSrc: srcFP, fpDst: dstFP,
                    y: y, lowDistThres: ldt, n: n, nFP: nFP, nNeighbors: nNeighborsLocal)
                eval(srcFP, dstFP)
            }

            let progressHit = onEpoch != nil && ((itr + 1) % max(1, progressEvery) == 0 || itr == numItersTotal - 1)
            if (itr + 1) % 10 == 0 || progressHit {
                eval(y, m, v)
            }
            if progressHit { onEpoch?(itr + 1, numItersTotal, y) }
        }

        eval(y)
        self.embedding = y
        return y
    }

    // MARK: - PCA init

    /// PCA initialisation: top-`nComponents` projection scaled by 0.01.
    private func pcaInit(_ x: MLXArray, nComponents: Int) -> MLXArray {
        let y = pcaReduce(x, dim: nComponents)
        return y * 0.01
    }
}

/// LocalMAP extends PaCMAP with local graph adjustment in the final phase.
public final class LocalMAP: PaCMAP {
    public var lowDistThres: Float

    public init(
        nComponents: Int = 2,
        nNeighbors: Int? = 10,
        mnRatio: Float = 0.5,
        fpRatio: Float = 2.0,
        learningRate: Float = 1.0,
        numIters: (Int, Int, Int) = (100, 100, 250),
        randomState: Int? = nil,
        verbose: Bool = false,
        applyPCA: Bool = true,
        knnMethod: KNNMethod = .auto,
        lowDistThres: Float = 10.0,
        normalize: Normalization = .none
    ) {
        self.lowDistThres = lowDistThres
        super.init(
            nComponents: nComponents, nNeighbors: nNeighbors, mnRatio: mnRatio,
            fpRatio: fpRatio, learningRate: learningRate, numIters: numIters,
            randomState: randomState, verbose: verbose, applyPCA: applyPCA,
            knnMethod: knnMethod, normalize: normalize)
    }

    /// Fit LocalMAP and return the embedding `(nSamples, nComponents)`.
    public override func fitTransform(_ x0: MLXArray) -> MLXArray {
        return fitTransformImpl(x0, lowDistThres: lowDistThres)
    }
}
