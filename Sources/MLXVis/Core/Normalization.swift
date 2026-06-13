// Shared input normalization for all DR methods.
// Port of mlx_vis/_normalize.py

import MLX

/// Input normalization applied before dimensionality reduction.
public enum Normalization: Sendable {
    /// No normalization (passthrough).
    case none
    /// z-score standardization per feature.
    case standard
    /// min-max scaling to [0, 1] per feature.
    case minmax
}

/// Normalize input data before dimensionality reduction.
///
/// - Parameters:
///   - x: Input data, shape `(nSamples, nFeatures)`.
///   - method: Normalization strategy.
/// - Returns: Normalized data of the same shape.
public func normalizeInput(_ x: MLXArray, method: Normalization) -> MLXArray {
    switch method {
    case .none:
        return x
    case .standard:
        let mean = x.mean(axis: 0)
        let centered = x - mean
        // Population std (ddof=0), matching numpy's default.
        let std = MLX.sqrt((centered * centered).mean(axis: 0)) + 1e-8
        return centered / std
    case .minmax:
        let xmin = x.min(axis: 0)
        let xmax = x.max(axis: 0)
        return (x - xmin) / (xmax - xmin + 1e-8)
    }
}
