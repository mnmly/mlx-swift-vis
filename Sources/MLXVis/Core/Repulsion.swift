// Repulsive force for t-SNE-family embeddings (t-SNE, DREAMS).
//
// One module behind one interface — `callAsFunction(y) -> (Z, grad)` — that picks
// the path ONCE at init and reuses the precomputed masks / compiled kernel across
// epochs:
//   - FFT  (FIt-SNE) for large 2-D embeddings — O(n);          see fftRepulsiveGrad
//   - full all-pairs (compiled) when n^2 fits comfortably;
//   - chunked all-pairs otherwise.
//
// Extracted verbatim from the duplicated TSNE/DREAMS repulsive code so the numerics
// are preserved exactly (guarded by the FFT parity tests and the cluster-separation
// regression).

import MLX

/// t-SNE repulsive-force gradient. Select once with `init(n:dims:)`, then call per
/// epoch. The returned `Z` is the partition function; `grad` is the unnormalized
/// repulsive gradient (caller scales by `4 / Z`).
public struct TSNERepulsion {
    private enum Mode { case fft, full, chunked }
    private let mode: Mode
    private let n: Int
    private let chunkSize: Int
    private let eyeMask: MLXArray
    private let selfMasks: [MLXArray]
    private let fullKernel: (@Sendable ([MLXArray]) -> [MLXArray])?

    /// - Parameters:
    ///   - n: number of points.
    ///   - dims: embedding dimensionality (FFT path requires 2).
    public init(n: Int, dims: Int) {
        self.n = n
        let useFFT = n >= 16000 && dims == 2
        let fullLimit = 1_000_000_000
        let useFull = !useFFT && (n * n * 4) < fullLimit
        self.chunkSize = min(n, max(512, 2_000_000_000 / (n * 4)))

        if useFFT {
            mode = .fft
            eyeMask = MLXArray.zeros([1])
            selfMasks = []
            fullKernel = nil
        } else if useFull {
            mode = .full
            let m = 1.0 - MLXArray.eye(n)
            eval(m)
            eyeMask = m
            selfMasks = []
            fullKernel = compile { Self.fullGraph($0[0], eyeMask: $0[1]) }
        } else {
            mode = .chunked
            var masks: [MLXArray] = []
            var start = 0
            while start < n {
                let end = min(start + chunkSize, n)
                let mask = 1.0 - (arangeColumn(start, end) .== arangeRow(n)).asType(.float32)
                eval(mask)
                masks.append(mask)
                start = end
            }
            selfMasks = masks
            eyeMask = MLXArray.zeros([1])
            fullKernel = nil
        }
    }

    /// True when the FFT-accelerated path is in use (large 2-D embedding).
    public var usesFFT: Bool { mode == .fft }

    public func callAsFunction(_ y: MLXArray) -> (z: MLXArray, grad: MLXArray) {
        switch mode {
        case .fft:
            let r = fftRepulsiveGrad(y, n: n)
            return (r.z, r.repGrad)
        case .full:
            let out = fullKernel!([y, eyeMask])
            return (out[0], out[1])
        case .chunked:
            return chunked(y)
        }
    }

    private static func fullGraph(_ y: MLXArray, eyeMask: MLXArray) -> [MLXArray] {
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

    private func chunked(_ y: MLXArray) -> (MLXArray, MLXArray) {
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
}
