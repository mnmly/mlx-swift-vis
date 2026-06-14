// CNE (Contrastive Neighbor Embedding) in pure MLX for Apple Silicon.
// Port of mlx_vis/_cne/cne.py
//
// Reference: Damrich et al., "From t-SNE to UMAP with contrastive learning",
// ICLR 2023. Unifies t-SNE and UMAP through contrastive losses on the neighbor
// graph. InfoNCE loss gives t-SNE-like results, NEG loss gives UMAP-like results.

import Foundation
import MLX
import MLXRandom

/// Contrastive loss variant for CNE.
public enum CNELoss: Sendable {
    /// InfoNCE (t-SNE-like, default).
    case infonce
    /// Noise-contrastive estimation.
    case nce
    /// Negative sampling (UMAP-like).
    case neg
}

/// Contrastive Neighbor Embedding using MLX on the Metal GPU.
public final class CNE {
    public var nComponents: Int
    public var nNeighbors: Int
    public var nNegatives: Int
    public var loss: CNELoss
    public var nIter: Int
    public var learningRate: Float
    public var batchSize: Int?
    public var pcaDim: Int?
    public var randomState: Int?
    public var verbose: Bool
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
        nNeighbors: Int = 15,
        nNegatives: Int = 5,
        loss: CNELoss = .infonce,
        nIter: Int = 500,
        learningRate: Float = 1.0,
        batchSize: Int? = nil,
        pcaDim: Int? = 50,
        randomState: Int? = nil,
        verbose: Bool = false,
        knnMethod: KNNMethod = .auto,
        normalize: Normalization = .none
    ) {
        self.nComponents = nComponents
        self.nNeighbors = nNeighbors
        self.nNegatives = nNegatives
        self.loss = loss
        self.nIter = nIter
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.pcaDim = pcaDim
        self.randomState = randomState
        self.verbose = verbose
        self.knnMethod = knnMethod
        self.normalize = normalize
    }

    /// Fit CNE and return the embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        var x = normalizeInput(x0.asType(.float32), method: normalize)
        let n = x.dim(0)
        let dim = x.dim(1)

        if verbose {
            print("CNE: \(n) points, \(dim) dims, loss=\(loss), k=\(nNeighbors), m=\(nNegatives)")
        }

        // PCA preprocessing for high-dimensional input.
        if let pcaDim, dim > pcaDim {
            if verbose { print("PCA: \(dim) -> \(pcaDim) dims...") }
            x = pcaReduce(x, dim: pcaDim)
        }

        if verbose { print("Computing k-NN...") }
        let (knnIndices, _) = computeKNN(
            x, k: nNeighbors, method: knnMethod,
            returnEuclidean: true, randomState: randomState, verbose: verbose)

        // Build symmetrized, deduplicated, self-loop-free edge list (CPU).
        let edges = Self.buildEdges(knnIndices, n: n)
        let nEdges = edges.dim(0)
        if verbose { print("Graph: \(nEdges) edges (symmetrized)") }

        // PCA initialization scaled by 0.01.
        var y = pcaReduce(x, dim: nComponents) * 0.01
        eval(y)

        // Determine batch size (None or >= nEdges means full batch).
        var bs = batchSize ?? nEdges
        if bs >= nEdges { bs = nEdges }

        if let randomState { MLXRandom.seed(UInt64(randomState)) }

        y = optimize(y: y, edges: edges, n: n, nEdges: nEdges, batchSize: bs)
        eval(y)

        self.embedding = y
        return y
    }

    // MARK: - Edge construction

    /// Build a symmetrized edge list from k-NN indices.
    ///
    /// For each directed edge i->j, adds both (i,j) and (j,i), deduplicates via a
    /// canonical key, and removes self-loops. Returns `(E, 2)` int32 array.
    static func buildEdges(_ knnIndices: MLXArray, n: Int) -> MLXArray {
        let k = knnIndices.dim(1)
        eval(knnIndices)
        let dstArr = knnIndices.reshaped([-1]).asType(.int32).asArray(Int32.self)

        let total = n * k
        var allSrc = [Int32]()
        var allDst = [Int32]()
        allSrc.reserveCapacity(total * 2)
        allDst.reserveCapacity(total * 2)

        // Forward edges i->j (src = repeat arange(n), k each).
        for i in 0..<n {
            for j in 0..<k {
                allSrc.append(Int32(i))
                allDst.append(dstArr[i * k + j])
            }
        }
        // Reverse edges j->i.
        for idx in 0..<total {
            allSrc.append(allDst[idx])
            allDst.append(allSrc[idx])
        }

        // Deduplicate via canonical key src*n + dst, dropping self-loops.
        let nL = Int64(n)
        var seen = Set<Int64>()
        seen.reserveCapacity(allSrc.count)
        var eSrc = [Int32]()
        var eDst = [Int32]()
        eSrc.reserveCapacity(allSrc.count)
        eDst.reserveCapacity(allSrc.count)
        for idx in 0..<allSrc.count {
            let s = allSrc[idx]
            let d = allDst[idx]
            if s == d { continue }
            let key = Int64(s) * nL + Int64(d)
            if seen.insert(key).inserted {
                eSrc.append(s)
                eDst.append(d)
            }
        }

        let e = eSrc.count
        // Interleave into (E, 2) row-major: [s0, d0, s1, d1, ...].
        var flat = [Int32]()
        flat.reserveCapacity(e * 2)
        for idx in 0..<e {
            flat.append(eSrc[idx])
            flat.append(eDst[idx])
        }
        return MLXArray(flat, [e, 2])
    }

    // MARK: - Contrastive gradients

    /// InfoNCE gradient. `ei`, `ej`: `(B,)`; `negIndices`: `(B, m)`. Returns grad `(n, d)`.
    private func infonceGrad(_ y: MLXArray, ei: MLXArray, ej: MLXArray, negIndices: MLXArray) -> MLXArray {
        let yi = y[ei]
        let yj = y[ej]
        let yk = y[negIndices]

        let diffPos = yi - yj
        let dPos = 1.0 + (diffPos * diffPos).sum(axis: -1)
        let sPos = 1.0 / dPos

        let diffNeg = yi.expandedDimensions(axis: 1) - yk
        let dNeg = 1.0 + (diffNeg * diffNeg).sum(axis: -1)
        let sNeg = 1.0 / dNeg

        let sNegSum = sNeg.sum(axis: 1)
        let z = sPos + sNegSum
        let wPos = 1.0 - sPos / z
        let wNeg = sNeg / z.expandedDimensions(axis: 1)

        let gAttr = wPos.expandedDimensions(axis: 1) * 2.0 * sPos.expandedDimensions(axis: 1) * diffPos
        let gRep = -(wNeg.expandedDimensions(axis: 2) * 2.0 * sNeg.expandedDimensions(axis: 2) * diffNeg).sum(axis: 1)

        let gradPerEdge = gAttr + gRep
        let negGrad = wNeg.expandedDimensions(axis: 2) * 2.0 * sNeg.expandedDimensions(axis: 2) * diffNeg

        return scatterGrad(y, ei: ei, ej: ej, negIndices: negIndices,
                           eiGrad: gradPerEdge, ejGrad: -gAttr, negGrad: negGrad)
    }

    /// NCE gradient. `invM = 1 / m`.
    private func nceGrad(_ y: MLXArray, ei: MLXArray, ej: MLXArray, negIndices: MLXArray, invM: Float) -> MLXArray {
        let yi = y[ei]
        let yj = y[ej]
        let yk = y[negIndices]

        let diffPos = yi - yj
        let dPos = 1.0 + (diffPos * diffPos).sum(axis: -1)
        let sPos = 1.0 / dPos

        let diffNeg = yi.expandedDimensions(axis: 1) - yk
        let dNeg = 1.0 + (diffNeg * diffNeg).sum(axis: -1)
        let sNeg = 1.0 / dNeg

        let gAttr = (2.0 * invM * sPos / (sPos + invM)).expandedDimensions(axis: 1) * diffPos
        let coefNeg = sNeg * sNeg / (sNeg + invM)
        let gRep = (-2.0 * coefNeg.expandedDimensions(axis: 2) * diffNeg).sum(axis: 1)
        let negGrad = 2.0 * coefNeg.expandedDimensions(axis: 2) * diffNeg

        return scatterGrad(y, ei: ei, ej: ej, negIndices: negIndices,
                           eiGrad: gAttr + gRep, ejGrad: -gAttr, negGrad: negGrad)
    }

    /// NEG (negative sampling) gradient.
    private func negGradFn(_ y: MLXArray, ei: MLXArray, ej: MLXArray, negIndices: MLXArray) -> MLXArray {
        let yi = y[ei]
        let yj = y[ej]
        let yk = y[negIndices]

        let diffPos = yi - yj
        let dPos = 1.0 + (diffPos * diffPos).sum(axis: -1)
        let sPos = 1.0 / dPos

        let diffNeg = yi.expandedDimensions(axis: 1) - yk
        let dNeg = 1.0 + (diffNeg * diffNeg).sum(axis: -1)
        let sNeg = 1.0 / dNeg

        let gAttr = (2.0 * sPos).expandedDimensions(axis: 1) * diffPos
        let oneMinusS = maximum(1.0 - sNeg, 1e-8)
        let wNeg = sNeg * sNeg / oneMinusS
        let gRep = (-2.0 * wNeg.expandedDimensions(axis: 2) * diffNeg).sum(axis: 1)
        let negGrad = 2.0 * wNeg.expandedDimensions(axis: 2) * diffNeg

        return scatterGrad(y, ei: ei, ej: ej, negIndices: negIndices,
                           eiGrad: gAttr + gRep, ejGrad: -gAttr, negGrad: negGrad)
    }

    /// Scatter per-edge gradients into a full `(n, d)` gradient buffer.
    private func scatterGrad(
        _ y: MLXArray, ei: MLXArray, ej: MLXArray, negIndices: MLXArray,
        eiGrad: MLXArray, ejGrad: MLXArray, negGrad: MLXArray
    ) -> MLXArray {
        let d = y.dim(1)
        var grad = MLXArray.zeros(like: y)
        grad = grad.at[ei].add(eiGrad)
        grad = grad.at[ej].add(ejGrad)
        grad = grad.at[negIndices.reshaped([-1])].add(negGrad.reshaped([-1, d]))
        return grad
    }

    // MARK: - Optimization

    private func optimize(y y0: MLXArray, edges: MLXArray, n: Int, nEdges: Int, batchSize: Int) -> MLXArray {
        var y = y0
        var m = MLXArray.zeros(like: y)
        var v = MLXArray.zeros(like: y)
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let eps: Float = 1e-8
        let lr = learningRate
        let nNeg = nNegatives
        let invM = 1.0 / Float(nNeg)
        let fullBatch = batchSize >= nEdges

        if verbose { print("Starting optimization...") }

        if onEpoch != nil { eval(y); onEpoch?(0, nIter, y) }  // initial frame
        for itr in 1...max(1, nIter) {
            let batchEdges: MLXArray
            if !fullBatch {
                let idx = MLXRandom.randInt(0 ..< nEdges, [batchSize])
                batchEdges = edges[idx]
            } else {
                batchEdges = edges
            }

            let ei = batchEdges[0..., 0]
            let ej = batchEdges[0..., 1]
            let b = batchEdges.dim(0)

            let negIndices = MLXRandom.randInt(0 ..< n, [b, nNeg])

            var grad: MLXArray
            switch loss {
            case .infonce: grad = infonceGrad(y, ei: ei, ej: ej, negIndices: negIndices)
            case .nce: grad = nceGrad(y, ei: ei, ej: ej, negIndices: negIndices, invM: invM)
            case .neg: grad = negGradFn(y, ei: ei, ej: ej, negIndices: negIndices)
            }

            if !fullBatch {
                grad = grad * (Float(nEdges) / Float(batchSize))
            }

            // Adam update.
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            let mHat = m / (1.0 - pow(beta1, Float(itr)))
            let vHat = v / (1.0 - pow(beta2, Float(itr)))
            y = y - lr * mHat / (MLX.sqrt(vHat) + eps)

            let progressHit = onEpoch != nil && (itr % max(1, progressEvery) == 0 || itr == nIter)
            if itr % 10 == 0 || itr == nIter || progressHit {
                eval(y, m, v)
            }
            if progressHit { onEpoch?(itr, nIter, y) }
            if verbose && itr % 50 == 0 {
                print("Iteration \(itr)/\(nIter)")
            }
        }
        return y
    }
}
