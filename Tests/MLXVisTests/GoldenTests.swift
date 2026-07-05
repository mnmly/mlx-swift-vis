// Behavior-preservation guard for the t-SNE-family / affinity-graph refactor.
//
// IMPORTANT: UMAP / t-SNE / DREAMS are NOT bit-reproducible. GPU scatter-add
// (`at[].add`, used for the attractive gradient) is atomic/unordered, so each run
// injects ~1e-6 noise that the chaotic dynamics amplify into a different layout.
// Positional golden fixtures are therefore impossible.
//
// What IS stable run-to-run is the embedding's STRUCTURE: well-separated clusters
// in high-D stay separated in 2-D. We assert that separation ratio. A refactor that
// breaks the optimizer (wrong gradient assembly, dropped force) collapses or scrambles
// the clusters and fails this. The deterministic building blocks (affinity graph,
// repulsion on a fixed y) are golden-tested exactly in AffinityGraphTests / the FFT
// parity tests.

import Foundation
import MLX
import XCTest

import MLXVis

final class GoldenTests: XCTestCase {
    /// Three well-separated Gaussian blobs in `d` dims; returns (X, labels).
    private func clusters(perCluster: Int = 100, d: Int = 20) -> (MLXArray, [Int]) {
        MLXRandom.seed(123)
        let centers: [[Float]] = [
            (0..<d).map { _ in 0 },
            (0..<d).map { i in i < 1 ? 30 : 0 },
            (0..<d).map { i in i < 2 && i >= 1 ? 30 : 0 },
        ]
        var parts: [MLXArray] = []
        var labels: [Int] = []
        for (c, center) in centers.enumerated() {
            let blob = MLXRandom.normal([perCluster, d]) + MLXArray(center)
            parts.append(blob)
            labels.append(contentsOf: Array(repeating: c, count: perCluster))
        }
        return (concatenated(parts, axis: 0), labels)
    }

    /// min over cluster pairs of (centroid distance) / max within-cluster RMS radius.
    /// >> 1 means clusters are cleanly separated in the embedding.
    private func separationRatio(_ y: MLXArray, _ labels: [Int]) -> Float {
        let pts = y.asArray(Float.self)
        let n = y.dim(0)
        let k = (labels.max() ?? 0) + 1
        var sum = [[Float]](repeating: [0, 0], count: k)
        var cnt = [Int](repeating: 0, count: k)
        for i in 0..<n {
            sum[labels[i]][0] += pts[i * 2]; sum[labels[i]][1] += pts[i * 2 + 1]
            cnt[labels[i]] += 1
        }
        let cent = (0..<k).map { [sum[$0][0] / Float(cnt[$0]), sum[$0][1] / Float(cnt[$0])] }
        var radius = [Float](repeating: 0, count: k)
        for i in 0..<n {
            let dx = pts[i * 2] - cent[labels[i]][0], dy = pts[i * 2 + 1] - cent[labels[i]][1]
            radius[labels[i]] += dx * dx + dy * dy
        }
        let rms = (0..<k).map { (radius[$0] / Float(cnt[$0])).squareRoot() }
        var minRatio = Float.greatestFiniteMagnitude
        for a in 0..<k {
            for b in (a + 1)..<k {
                let dx = cent[a][0] - cent[b][0], dy = cent[a][1] - cent[b][1]
                let dist = (dx * dx + dy * dy).squareRoot()
                minRatio = Swift.min(minRatio, dist / Swift.max(rms[a], rms[b], 1e-6))
            }
        }
        return minRatio
    }

    func testUMAPSeparatesClusters() {
        let (x, labels) = clusters()
        let y = UMAP(nComponents: 2, nNeighbors: 15, nEpochs: 200, randomState: 0).fitTransform(x)
        let r = separationRatio(y, labels)
        print("UMAP separation ratio: \(r)")
        XCTAssertGreaterThan(r, 2.0, "UMAP failed to separate clusters")
    }

    func testPaCMAPSeparatesClusters() {
        let (x, labels) = clusters()
        let y = PaCMAP(nComponents: 2, nNeighbors: 10, randomState: 0).fitTransform(x)
        let r = separationRatio(y, labels)
        print("PaCMAP separation ratio: \(r)")
        XCTAssertGreaterThan(r, 2.0, "PaCMAP failed to separate clusters")
    }

    func testLocalMAPSeparatesClusters() {
        let (x, labels) = clusters()
        let y = LocalMAP(nComponents: 2, nNeighbors: 10, randomState: 0, lowDistThres: 10.0).fitTransform(x)
        let r = separationRatio(y, labels)
        print("LocalMAP separation ratio: \(r)")
        XCTAssertGreaterThan(r, 2.0, "LocalMAP failed to separate clusters")
    }

    func testTSNESeparatesClusters() {
        let (x, labels) = clusters()
        let y = TSNE(nComponents: 2, perplexity: 30, nIter: 500, randomState: 0, pcaDim: 15).fitTransform(x)
        let r = separationRatio(y, labels)
        print("TSNE separation ratio: \(r)")
        XCTAssertGreaterThan(r, 2.0, "TSNE failed to separate clusters")
    }

    func testDREAMSSeparatesClusters() {
        let (x, labels) = clusters()
        let y = DREAMS(nComponents: 2, perplexity: 30, nIter: 500, pcaDim: 15, randomState: 0).fitTransform(x)
        let r = separationRatio(y, labels)
        print("DREAMS separation ratio: \(r)")
        XCTAssertGreaterThan(r, 2.0, "DREAMS failed to separate clusters")
    }

    /// n = 4500 (> the FFT gate of 4000) so the repulsive force runs through the
    /// FFT-accelerated path. Guards that lowering the gate from 16000 keeps t-SNE /
    /// DREAMS embeddings well-separated (and NaN-free) in the newly-FFT range.
    func testDREAMSSeparatesClustersOnFFTPath() {
        let (x, labels) = clusters(perCluster: 1500)  // 3 × 1500 = 4500 > 4000
        XCTAssertGreaterThanOrEqual(x.dim(0), 4000, "must exceed FFT gate")
        let y = DREAMS(nComponents: 2, perplexity: 30, nIter: 500, pcaDim: 15, randomState: 0).fitTransform(x)
        eval(y)
        XCTAssertFalse(MLX.any(y .!= y).item(Bool.self), "FFT-path DREAMS produced NaN")
        let r = separationRatio(y, labels)
        print("DREAMS (FFT path, n=\(x.dim(0))) separation ratio: \(r)")
        XCTAssertGreaterThan(r, 2.0, "DREAMS on FFT path failed to separate clusters")
    }
}
