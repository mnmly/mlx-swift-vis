// Lightweight PCA via MLX SVD.
// Port of mlx_vis/pca.py

import MLX
import MLXLinalg

/// Principal Component Analysis using MLX.
public final class PCA {
    public let nComponents: Int
    public private(set) var components: MLXArray?
    public private(set) var mean: MLXArray?
    public private(set) var embedding: MLXArray?

    public init(nComponents: Int = 2) {
        self.nComponents = nComponents
    }

    /// Fit PCA and return the projection of shape `(nSamples, nComponents)`.
    public func fitTransform(_ x0: MLXArray) -> MLXArray {
        let x = x0.asType(.float32)
        let mean = x.mean(axis: 0)
        self.mean = mean
        let centered = x - mean

        let n = centered.dim(0)
        let d = centered.dim(1)

        let vt: MLXArray
        if n > d {
            // n >> d: SVD on the d x d covariance matrix (much faster).
            let cov = centered.transposed().matmul(centered) / Float(n - 1)
            eval(cov)
            vt = MLXLinalg.svd(cov, stream: .cpu).2
        } else {
            vt = MLXLinalg.svd(centered, stream: .cpu).2
        }

        let comps = vt[..<nComponents]
        self.components = comps
        let y = centered.matmul(comps.transposed())
        eval(y)
        self.embedding = y
        return y
    }
}

/// Project `x` onto its top `dim` principal components in descending-eigenvalue
/// order via a covariance SVD on the CPU.
///
/// The Python reference uses `mx.linalg.eigh` (ascending) then reverses; mlx-swift
/// has no `eigh`, but the covariance matrix is symmetric PSD so `svd(cov)` already
/// yields eigenvectors in descending singular-value (= eigenvalue) order. Used both
/// for in-pipeline reduction before KNN and for PCA-based embedding initialization.
func pcaReduce(_ x: MLXArray, dim: Int) -> MLXArray {
    let xf = x.asType(.float32)
    let n = xf.dim(0)
    let mean = xf.mean(axis: 0)
    let centered = xf - mean
    let cov = centered.transposed().matmul(centered) / Float(n - 1)
    eval(cov)
    let vt = MLXLinalg.svd(cov, stream: .cpu).2          // rows = eigenvectors, descending
    let comps = vt[..<dim]                                // (dim, d)
    let y = centered.matmul(comps.transposed())          // (n, dim)
    eval(y)
    return y
}
