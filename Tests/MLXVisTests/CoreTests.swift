import MLX
import XCTest

import MLXVis

final class CoreTests: XCTestCase {
    func testPCAShape() {
        let x = MLXRandom.normal([200, 16])
        let y = PCA(nComponents: 2).fitTransform(x)
        XCTAssertEqual(y.shape, [200, 2])
    }

    func testBruteForceKNNShape() {
        let x = MLXRandom.normal([128, 8])
        let (idx, dist) = bruteForceKNN(x, k: 5, returnEuclidean: true)
        XCTAssertEqual(idx.shape, [128, 5])
        XCTAssertEqual(dist.shape, [128, 5])
    }

    func testFindABParams() {
        // Reference values from the Python UMAP defaults (spread=1, min_dist=0.1).
        let (a, b) = UMAP.findABParams(spread: 1.0, minDist: 0.1)
        XCTAssertEqual(a, 1.577, accuracy: 0.05)
        XCTAssertEqual(b, 0.895, accuracy: 0.05)
    }

    func testUMAPShape() {
        let x = MLXRandom.normal([400, 20])
        let y = UMAP(nComponents: 2, nNeighbors: 15, nEpochs: 50, randomState: 42)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [400, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    func testTSNEShape() {
        let x = MLXRandom.normal([300, 30])
        let y = TSNE(nComponents: 2, perplexity: 15, nIter: 60, randomState: 42, pcaDim: 20)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [300, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    func testPaCMAPShape() {
        let x = MLXRandom.normal([300, 30])
        let y = PaCMAP(nComponents: 2, nNeighbors: 10, numIters: (10, 10, 20), randomState: 42)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [300, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    func testTriMapShape() {
        let x = MLXRandom.normal([300, 30])
        let y = TriMap(
            nComponents: 2, nNeighbors: 10, nInliers: 8, nOutliers: 2,
            nRandom: 2, nIters: 30, randomState: 42)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [300, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    func testDREAMSShape() {
        let x = MLXRandom.normal([300, 30])
        let y = DREAMS(nComponents: 2, perplexity: 15, nIter: 60, pcaDim: 20, randomState: 42)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [300, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    func testCNEShape() {
        let x = MLXRandom.normal([300, 30])
        let y = CNE(nComponents: 2, nNeighbors: 10, nIter: 60, pcaDim: 20, randomState: 42)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [300, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    func testMMAEShape() {
        let x = MLXRandom.normal([300, 30])
        let y = MMAE(
            nComponents: 2, nEpochs: 5, batchSize: 64, pcaDim: 20, randomState: 42)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [300, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    func testNNDescentRecall() {
        MLXRandom.seed(7)
        let n = 2000
        let k = 10
        let x = MLXRandom.normal([n, 16])

        let (exactIdx, _) = bruteForceKNN(x, k: k, returnEuclidean: true)
        let (approxIdx, approxDist) = NNDescent(k: k, randomState: 42).build(x)

        XCTAssertEqual(approxIdx.shape, [n, k])
        XCTAssertEqual(approxDist.shape, [n, k])

        let exact = exactIdx.asType(.int32).asArray(Int32.self)
        let approx = approxIdx.asType(.int32).asArray(Int32.self)

        var totalRecall = 0.0
        for row in 0 ..< n {
            var exactSet = Set<Int32>()
            for j in 0 ..< k { exactSet.insert(exact[row * k + j]) }
            var found = 0
            for j in 0 ..< k where exactSet.contains(approx[row * k + j]) { found += 1 }
            totalRecall += Double(found) / Double(k)
        }
        let recall = totalRecall / Double(n)
        print("NNDescent recall: \(recall)")
        XCTAssertGreaterThan(recall, 0.80)
    }

    func testLocalMAPShape() {
        let x = MLXRandom.normal([300, 30])
        let y = LocalMAP(
            nComponents: 2, nNeighbors: 10, numIters: (10, 10, 20),
            randomState: 42, lowDistThres: 10.0)
            .fitTransform(x)
        XCTAssertEqual(y.shape, [300, 2])
        XCTAssertFalse(y.anyNaN().item(Bool.self))
    }

    /// The FFT repulsive gradient must approximate the exact O(n^2) one (the trusted
    /// path). Verified directly rather than via end-to-end embedding parity.
    func testFFTRepulsiveMatchesExact() {
        let n = 3000
        MLXRandom.seed(7)
        let y = MLXRandom.normal([n, 2]) * 10.0  // realistic embedding spread
        eval(y)

        // Exact repulsive (same formula as TSNE.repulsiveFullGraph).
        let sq = (y * y).sum(axis: 1)
        let dsq = MLX.maximum(
            sq.expandedDimensions(axis: 1) + sq.expandedDimensions(axis: 0)
                - 2.0 * y.matmul(y.transposed()), 0.0)
        let kernel = (1.0 - MLXArray.eye(n)) / (1.0 + dsq)
        let zExact = kernel.sum()
        let ksq = kernel * kernel
        let gExact = ksq.sum(axis: 1, keepDims: true) * y - ksq.matmul(y)
        eval(zExact, gExact)

        let (zFFT, gFFT) = fftRepulsiveGrad(y, n: n)
        eval(zFFT, gFFT)

        let zRel = abs(zFFT - zExact).item(Float.self) / abs(zExact).item(Float.self)
        let gNum = MLX.sqrt(((gFFT - gExact) * (gFFT - gExact)).sum()).item(Float.self)
        let gDen = MLX.sqrt((gExact * gExact).sum()).item(Float.self)
        let gRel = gNum / gDen
        // These bound FIt-SNE's *inherent* interpolation error (the Python reference
        // has the same), not port correctness — that's gated by the tight
        // Swift-vs-Python check in ParityTests.testFFTRepulsiveParity.
        print("FFT repulsive vs exact — Z rel err: \(zRel), grad rel err: \(gRel)")
        XCTAssertLessThan(zRel, 0.05, "FFT Z partition diverges from exact")
        XCTAssertLessThan(gRel, 0.15, "FFT repulsive gradient diverges from exact")
    }
}

extension MLXArray {
    fileprivate func anyNaN() -> MLXArray { MLX.any(self .!= self) }
}
