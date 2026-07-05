// The t-SNE-family optimizer: sparse attractive forces over KNN edges, exact/FFT
// repulsive forces, and momentum gradient descent with delta-bar-delta adaptive
// gains and an early-exaggeration phase.
//
// This is the shared core of t-SNE and DREAMS. The single seam is `transformGradient`:
// t-SNE passes nothing (identity); DREAMS passes its PCA-regularization combine
// `(1-λ)·grad + (2λ/n)(Y − α·Ỹ)`. Repulsion path selection and progress emission
// live here, so both methods reduce to "build P → init → optimize".

import MLX

/// Run the t-SNE-family optimization loop.
///
/// - Parameters:
///   - edgeFrom/edgeTo/edgeWeights: symmetric P-matrix edge list (see `buildPMatrix`).
///   - y0: initial embedding `(n, dims)`.
///   - transformGradient: optional hook applied to the full gradient each epoch,
///     `(grad, y, epoch) -> grad`. Identity when nil (plain t-SNE).
/// - Returns: the optimized embedding.
func optimizeTSNEFamily(
    edgeFrom: MLXArray, edgeTo: MLXArray, edgeWeights: MLXArray,
    y y0: MLXArray, n: Int,
    learningRate: Float, earlyExaggeration: Float, earlyExaggerationIter: Int, nEpochs: Int,
    onEpoch: ((Int, Int, MLXArray) -> Void)?, progressEvery: Int, verbose: Int,
    transformGradient: ((MLXArray, MLXArray, Int) -> MLXArray)? = nil
) -> MLXArray {
    var y = y0
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
        // Cooperative cancellation: if the caller's Task is cancelled, stop and return
        // the best-so-far embedding (already mean-centered). No-op outside a Task.
        if Task.isCancelled { break }
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

        // Method-specific gradient hook (DREAMS regularization; identity for t-SNE).
        if let transformGradient { grad = transformGradient(grad, y, epoch) }

        // Phase transition reset.
        if epoch == earlyExaggerationIter {
            gains = MLXArray.ones(like: y)
            velocity = MLXArray.zeros(like: y)
        }

        // Adaptive gains (delta-bar-delta).
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
