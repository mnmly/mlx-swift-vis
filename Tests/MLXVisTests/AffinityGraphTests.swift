// Deterministic golden lock for the shared affinity-graph primitives.
//
// buildPMatrix / reverseEdgeValues have NO scatter-add, so they are exactly
// reproducible (unlike the chaotic optimizers). This pins their output so the
// extraction from TSNE/DREAMS is provably behavior-preserving and stays that way.

import MLX
import XCTest

import MLXVis

final class AffinityGraphTests: XCTestCase {
    private func fixedKNN() -> (MLXArray, MLXArray) {
        MLXRandom.seed(7)
        let x = MLXRandom.normal([200, 16])
        return bruteForceKNN(x, k: 30, returnEuclidean: false)  // squared dists (t-SNE input)
    }

    func testBuildPMatrixGolden() {
        let (idx, dist) = fixedKNN()
        let (rows, cols, vals) = buildPMatrix(idx, dist, perplexity: 10, n: 200)
        eval(rows, cols, vals)

        let edges = rows.dim(0)
        let vsum = vals.sum().item(Float.self)
        let wdot = (vals * cols.asType(.float32)).sum().item(Float.self)

        // Symmetric, normalized P: sum(P) ≈ 1.
        XCTAssertEqual(Double(vsum), 1.0, accuracy: 1e-3, "P should sum to 1")
        // Deterministic structure lock (no scatter-add → exact run to run).
        XCTAssertEqual(edges, 8700, "edge count drift")
        XCTAssertEqual(Double(wdot), 96.31388092041016, accuracy: 1e-2, "weighted-col-sum drift")
    }
}
