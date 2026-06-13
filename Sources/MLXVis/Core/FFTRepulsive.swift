// FFT-accelerated repulsive gradient for 2D t-SNE-style embeddings.
// Port of TSNE._fft_repulsive from mlx_vis/_tsne/tsne.py.
//
// This is an approximation of the exact O(n^2) repulsive gradient (interpolated
// onto a grid, convolved with the Cauchy kernel 1/(1+d^2) via FFT, then
// interpolated back) — the FIt-SNE method. It produces the same (Z, repGrad) the
// exact path does, to interpolation accuracy, in O(n + ng^2 log ng) instead of
// O(n^2). Used by t-SNE / DREAMS for large n (2D only).
//
// Grid resolution depends on the live embedding extent, so this is intentionally
// NOT compiled (shapes change as the embedding spreads); it matches the Python
// staticmethod, which is likewise uncompiled.

import Foundation
import MLX
import MLXFFT

/// FFT-accelerated repulsive gradient. `y` must be `(n, 2)`.
/// - Returns: `(Z, repGrad)` matching the exact repulsive computation.
public func fftRepulsiveGrad(_ y: MLXArray, n: Int) -> (z: MLXArray, repGrad: MLXArray) {
    let nInterp = 3

    // Lagrange nodes/denominators (constant; n_interp = 3).
    let h = 1.0 / 3.0
    let nodes: [Float] = [(0.5 + 0) * Float(h), (0.5 + 1) * Float(h), (0.5 + 2) * Float(h)]
    let denom: [Float] = [
        (nodes[0] - nodes[1]) * (nodes[0] - nodes[2]),
        (nodes[1] - nodes[0]) * (nodes[1] - nodes[2]),
        (nodes[2] - nodes[0]) * (nodes[2] - nodes[1]),
    ]

    // Grid setup — extent must be CPU-known to size the grid, so pull min/max.
    let yMin = y.min(axis: 0)
    let yMax = y.max(axis: 0)
    eval(yMin, yMax)
    let mn = yMin.asArray(Float.self)
    let mx2 = yMax.asArray(Float.self)
    var cr = [mx2[0] - mn[0], mx2[1] - mn[1]]
    let pad = [cr[0] * 0.01 + 1e-8, cr[1] * 0.01 + 1e-8]
    let bl = [mn[0] - pad[0], mn[1] - pad[1]]
    cr = [cr[0] + 2 * pad[0], cr[1] + 2 * pad[1]]

    let nBoxes = Swift.max(50, Int(ceil(Double(Swift.max(cr[0], cr[1])))))
    let ng = nBoxes * nInterp

    // Scale points into [0, nBoxes].
    let blMx = MLXArray(bl)
    let crMx = MLXArray(cr)
    let yScaled = (y - blMx) / crMx * Float(nBoxes)

    let boxIdx = clip(floor(yScaled).asType(.int32), min: 0, max: nBoxes - 1)
    let yRel = yScaled - boxIdx.asType(.float32)

    // Lagrange weights per dim: (n, 3) tensor-product factors.
    let nodesRow = MLXArray(nodes).expandedDimensions(axis: 0)  // (1, 3)
    let diffX = yRel[0..., 0..<1] - nodesRow                    // (n, 3)
    let diffY = yRel[0..., 1..<2] - nodesRow                    // (n, 3)

    let wx0 = diffX[0..., 1] * diffX[0..., 2] / denom[0]
    let wx1 = diffX[0..., 0] * diffX[0..., 2] / denom[1]
    let wx2 = diffX[0..., 0] * diffX[0..., 1] / denom[2]
    let wy0 = diffY[0..., 1] * diffY[0..., 2] / denom[0]
    let wy1 = diffY[0..., 0] * diffY[0..., 2] / denom[1]
    let wy2 = diffY[0..., 0] * diffY[0..., 1] / denom[2]

    let bx = boxIdx[0..., 0]
    let by = boxIdx[0..., 1]
    let gx0 = bx * Int32(nInterp); let gx1 = gx0 + 1; let gx2 = gx0 + 2
    let gy0 = by * Int32(nInterp); let gy1 = gy0 + 1; let gy2 = gy0 + 2
    let ngI = Int32(ng)

    // 9 flat grid indices per point: (n, 9).
    let flatIdx = stacked([
        gx0 * ngI + gy0, gx0 * ngI + gy1, gx0 * ngI + gy2,
        gx1 * ngI + gy0, gx1 * ngI + gy1, gx1 * ngI + gy2,
        gx2 * ngI + gy0, gx2 * ngI + gy1, gx2 * ngI + gy2,
    ], axis: 1)

    // 9 interpolation weights per point: (n, 9).
    let interpW = stacked([
        wx0 * wy0, wx0 * wy1, wx0 * wy2,
        wx1 * wy0, wx1 * wy1, wx1 * wy2,
        wx2 * wy0, wx2 * wy1, wx2 * wy2,
    ], axis: 1)

    // Charges: (n, 4) = [1, 1, y0, y1].
    let onesN = MLXArray.ones([n])
    let charges = stacked([onesN, onesN, y[0..., 0], y[0..., 1]], axis: 1)

    // Step 1: scatter charges to the grid.
    let weightedCharges = interpW.expandedDimensions(axis: 2) * charges.expandedDimensions(axis: 1)  // (n,9,4)
    let wcFlat = weightedCharges.reshaped([n * 9, 4])
    let idxFlat = flatIdx.reshaped([n * 9])
    var wGridFlat = MLXArray.zeros([ng * ng, 4])
    wGridFlat = wGridFlat.at[idxFlat].add(wcFlat)
    let wGrid = wGridFlat.reshaped([ng, ng, 4])

    // Step 2: FFT convolution with the Cauchy kernel.
    let m = ng
    let hx = cr[0] / Float(m)
    let hy = cr[1] / Float(m)
    var ax = MLXArray(Int32(0) ..< Int32(2 * m)).asType(.float32)
    ax = MLX.where(ax .>= Float(m), ax - Float(2 * m), ax)
    let dsqGrid = MLX.pow(ax.expandedDimensions(axis: 1) * hx, 2)
        + MLX.pow(ax.expandedDimensions(axis: 0) * hy, 2)
    let k1 = 1.0 / (1.0 + dsqGrid)
    let fk1 = MLXFFT.rfft2(k1)
    let fk2 = MLXFFT.rfft2(k1 * k1)

    var wPadded = MLXArray.zeros([4, 2 * m, 2 * m])
    let wgT = wGrid.transposed(2, 0, 1)  // (4, M, M)
    wPadded = wPadded.at[0..., ..<m, ..<m].add(wgT)
    let fw = MLXFFT.rfft2(wPadded)  // (4, 2M, M+1)

    let fkBatch = stacked([fk1, fk2, fk2, fk2], axis: 0)  // (4, 2M, M+1)
    let result = MLXFFT.irfft2(fw * fkBatch, s: [2 * m, 2 * m])  // (4, 2M, 2M)
    let pot = result[0..., ..<m, ..<m]  // (4, M, M)

    // Step 3: gather potentials back to points.
    let potFlat = pot.reshaped([4, m * m]).transposed()  // (M*M, 4)
    let gathered = potFlat[idxFlat].reshaped([n, 9, 4])
    let phi = (interpW.expandedDimensions(axis: 2) * gathered).sum(axis: 1)  // (n, 4)

    let z = phi[0..., 0].sum() - Float(n)
    let repGrad = stacked([
        y[0..., 0] * phi[0..., 1] - phi[0..., 2],
        y[0..., 1] * phi[0..., 1] - phi[0..., 3],
    ], axis: 1)

    return (z, repGrad)
}
