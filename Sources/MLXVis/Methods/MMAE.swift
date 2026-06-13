// MMAE (Manifold-Matching Autoencoder) in pure MLX for Apple Silicon.
// Port of mlx_vis/_mmae/mmae.py
//
// Reference: Cheret et al., "Manifold-Matching Autoencoders", arXiv:2603.16568, 2026.
//
// The key idea: align pairwise Euclidean distances in the latent space to those
// in a reference space (raw input or PCA-reduced input) via MSE on distance
// matrices computed per minibatch. This preserves global geometry and, by the
// stability theorem, topology. The autoencoder also reconstructs the (possibly
// PCA-reduced) input, and the encoder generalizes to out-of-sample points.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Pairwise Euclidean distance matrix for a batch of row vectors `(B, d)`.
private func pairwiseEuclidean(_ z: MLXArray) -> MLXArray {
    let sqNorms = (z * z).sum(axis: 1)
    let distsSq = sqNorms.expandedDimensions(axis: 1)
        + sqNorms.expandedDimensions(axis: 0)
        - 2.0 * z.matmul(z.transposed())
    return MLX.sqrt(maximum(distsSq, 0.0) + 1e-12)
}

/// MLP with ReLU activations. Hidden layers have no bias; output layer keeps bias.
/// Mirrors the Python `_MLP`.
private final class MLP: Module, UnaryLayer {
    @ModuleInfo var linears: [Linear]

    init(dims: [Int]) {
        let n = dims.count - 1
        var layers: [Linear] = []
        layers.reserveCapacity(n)
        for i in 0..<n {
            // Only the output layer (i == n - 1) keeps a bias.
            layers.append(Linear(dims[i], dims[i + 1], bias: i == n - 1))
        }
        self._linears = ModuleInfo(wrappedValue: layers)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for i in 0..<(linears.count - 1) {
            h = relu(linears[i](h))
        }
        h = linears[linears.count - 1](h)
        return h
    }
}

/// Autoencoder with a narrower decoder, mirroring the Python `_Autoencoder`.
private final class Autoencoder: Module {
    @ModuleInfo var encoder: MLP
    @ModuleInfo var decoder: MLP

    init(inputDim: Int, hiddenDims: [Int], latentDim: Int, outputDim: Int) {
        let encDims = [inputDim] + hiddenDims + [latentDim]
        // dec_hidden = [d // 2 for d in reversed(hidden_dims)]; dec_dims uses dec_hidden[1:]
        let decHidden = hiddenDims.reversed().map { $0 / 2 }
        let decDims = [latentDim] + Array(decHidden.dropFirst()) + [outputDim]
        self._encoder = ModuleInfo(wrappedValue: MLP(dims: encDims))
        self._decoder = ModuleInfo(wrappedValue: MLP(dims: decDims))
        super.init()
    }

    /// Returns `(z, recon)`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let z = encoder(x)
        let recon = decoder(z)
        return (z, recon)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        encoder(x)
    }
}

/// Manifold-Matching Autoencoder for dimensionality reduction using MLX.
public final class MMAE {
    public var nComponents: Int
    public var nEpochs: Int
    public var batchSize: Int
    public var lr: Float
    public var weightDecay: Float
    public var lambdaMM: Float
    public var hiddenDims: [Int]
    public var pcaDim: Int?
    public var randomState: Int?
    public var verbose: Bool
    /// External reference embedding `(nSamples, k)`. Overrides `pcaDim`/raw input.
    public var reference: MLXArray?
    public var knnMethod: KNNMethod
    public var normalize: Normalization

    public private(set) var embedding: MLXArray?

    public init(
        nComponents: Int = 2,
        nEpochs: Int = 39,
        batchSize: Int = 512,
        lr: Float = 2e-3,
        weightDecay: Float = 0,
        lambdaMM: Float = 0.40,
        hiddenDims: [Int] = [256, 128, 64],
        pcaDim: Int? = nil,
        randomState: Int? = nil,
        verbose: Bool = false,
        reference: MLXArray? = nil,
        knnMethod: KNNMethod = .auto,
        normalize: Normalization = .none
    ) {
        self.nComponents = nComponents
        self.nEpochs = nEpochs
        self.batchSize = batchSize
        self.lr = lr
        self.weightDecay = weightDecay
        self.lambdaMM = lambdaMM
        self.hiddenDims = hiddenDims
        self.pcaDim = pcaDim
        self.randomState = randomState
        self.verbose = verbose
        self.reference = reference
        self.knnMethod = knnMethod
        self.normalize = normalize
    }

    /// Build the reference embedding for MM-reg.
    private func buildReference(_ x: MLXArray, n: Int, d: Int, reference: MLXArray?) -> MLXArray {
        if let reference {
            let ref = reference.asType(.float32)
            precondition(ref.dim(0) == n, "reference has \(ref.dim(0)) samples, expected \(n)")
            if verbose { print("MMAE: using external reference (\(ref.dim(1))D)") }
            return ref
        }
        if let pcaDim, d > pcaDim {
            if verbose { print("MMAE: PCA reference \(d)D -> \(pcaDim)D") }
            return pcaReduce(x, dim: pcaDim)
        }
        if verbose { print("MMAE: using raw input (\(d)D) as reference space") }
        return x
    }

    /// Fit the autoencoder and return the latent embedding `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        fitTransform(x0, reference: nil)
    }

    /// Fit with an optional per-call reference embedding (overrides the init-time reference).
    public func fitTransform(_ x0: MLXArray, reference: MLXArray?) -> MLXArray {
        let xNorm = normalizeInput(x0.asType(.float32), method: normalize)
        let n = xNorm.dim(0)
        let d = xNorm.dim(1)

        if let randomState { MLXRandom.seed(UInt64(randomState)) }

        let ref = reference ?? self.reference
        let eMx = buildReference(xNorm, n: n, d: d, reference: ref)

        // GPU "normalization": global mean-center + global rsqrt scale (matches Python).
        let mean = xNorm.mean(axis: 0)
        let xc = xNorm - mean
        let xTrain = xc * MLX.rsqrt((xc * xc).mean() + 1e-8)
        eval(xTrain, eMx)

        // Use PCA-reduced input as reconstruction target when available.
        var reconDim = d
        var xRecon = xTrain
        var sameReconRef = false
        if let pcaDim, d > pcaDim {
            xRecon = eMx
            reconDim = pcaDim
            sameReconRef = true
        }

        let model = Autoencoder(
            inputDim: d, hiddenDims: hiddenDims, latentDim: nComponents, outputDim: reconDim)
        eval(model.parameters())

        let optimizer = AdamW(learningRate: lr, weightDecay: weightDecay)

        let lambdaMM = self.lambdaMM
        let batchSize = min(self.batchSize, n)

        // loss(model, [x_batch, e_batch, r_batch]) -> [loss]
        let lossFn: (Autoencoder, [MLXArray]) -> [MLXArray] = { model, args in
            let (z, recon) = model(args[0])
            let reconLoss = ((args[2] - recon) * (args[2] - recon)).mean()
            let dZ = pairwiseEuclidean(z)
            let dE = pairwiseEuclidean(args[1])
            let mmLoss = ((dZ - dE) * (dZ - dE)).mean()
            return [reconLoss + lambdaMM * mmLoss]
        }
        let lossAndGrad = valueAndGrad(model: model, lossFn)

        if verbose {
            print("MMAE: architecture \(d) -> \(hiddenDims) -> \(nComponents); n=\(n), bs=\(batchSize), epochs=\(nEpochs)")
        }

        // Train on subsample per epoch; encoder generalizes to full dataset.
        var trainN = min(n, 40000)
        trainN = (trainN / batchSize) * batchSize  // align to batch_size
        let nBatches = max(0, trainN / batchSize)

        model.train()
        for _ in 0..<nEpochs {
            // Sample trainN distinct indices (replace=False) on CPU, matching Python.
            let perm = MMAE.choice(n: n, count: trainN)
            let xShuf = take(xTrain, perm, axis: 0)
            let eShuf = take(eMx, perm, axis: 0)
            let rShuf = sameReconRef ? eShuf : take(xRecon, perm, axis: 0)

            for i in 0..<nBatches {
                let b = i * batchSize
                let xb = xShuf[b..<(b + batchSize)]
                let eb = eShuf[b..<(b + batchSize)]
                let rb = rShuf[b..<(b + batchSize)]
                let (_, grads) = lossAndGrad(model, [xb, eb, rb])
                optimizer.update(model: model, gradients: grads)
                eval(model.parameters())
            }
        }

        // Final embedding over the full dataset (inference mode; no dropout here).
        model.train(false)
        let y = model.encode(xTrain)
        eval(y)
        self.embedding = y
        return y
    }

    /// Sample `count` distinct indices from `0..<n` without replacement (CPU shuffle).
    private static func choice(n: Int, count: Int) -> MLXArray {
        if count >= n {
            // Full permutation.
            var idx = Array(0..<n).map { Int32($0) }
            shuffle(&idx)
            return MLXArray(idx)
        }
        var idx = Array(0..<n).map { Int32($0) }
        shuffle(&idx)
        return MLXArray(Array(idx[0..<count]))
    }

    /// Fisher-Yates shuffle using MLX's RNG (seeded for reproducibility).
    private static func shuffle(_ a: inout [Int32]) {
        let n = a.count
        guard n > 1 else { return }
        // Draw all random ints in one go to limit eval overhead.
        let r = MLXRandom.randInt(0 ..< Int32(n), [n]).asArray(Int32.self)
        for i in stride(from: n - 1, through: 1, by: -1) {
            let j = Int(r[i]) % (i + 1)
            a.swapAt(i, j)
        }
    }
}
