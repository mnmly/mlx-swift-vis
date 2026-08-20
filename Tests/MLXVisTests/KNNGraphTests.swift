// Precomputed k-NN graph: export/inject parity, validation, serialization, and the
// NNDescent progress hook.
//
// The parity bar, in three exact assertions plus one bounded one:
//   1. the k-NN build is bit-reproducible (`testKNNBuildIsDeterministic`),
//   2. an injected graph reaches the optimizer verbatim (`testInjectedGraphIsUsedVerbatim`),
//   3. injection leaves MLX's global PRNG exactly where a build would
//      (`testInjectionPreservesGlobalRandomState`) — NNDescent's random init consumes one
//      key split, and every later draw in a fit reads from that stream,
//   so the whole KNN → affinity graph → initialization pipeline is bit-identical
//   (`testInjectedGraphReproducesInitializationExactly`, UMAP with the optimizer off).
// Past that point the optimizers use GPU scatter-add, whose float accumulation order is
// nondeterministic: a fit does not repeat itself bit for bit even with no caching at all,
// and the difference amplifies over epochs. So the end-to-end check
// (`testInjectedGraphMatchesWithinRunToRunNoise`) asserts the build-vs-inject spread sits
// inside the method's own run-to-run spread.

import Foundation
import MLX
import XCTest

@testable import MLXVis

final class KNNGraphTests: XCTestCase {

    /// One fit, optionally fed a precomputed graph; always exports the graph it used.
    private typealias Run = (KNNGraph?) -> (embedding: MLXArray, graph: KNNGraph?)

    private func data(_ n: Int, _ d: Int, seed: UInt64 = 7) -> MLXArray {
        MLXRandom.seed(seed)
        let x = MLXRandom.normal([n, d])
        eval(x)
        return x
    }

    /// Every NNDescent-consuming method, forced onto the approximate path so the small
    /// test `n` still exercises the code the cache is for.
    private func cases(_ x: MLXArray) -> [(String, Run)] {
        [
            ("UMAP", { g in
                let r = UMAP(
                    nComponents: 2, nNeighbors: 10, nEpochs: 40, randomState: 42,
                    knnMethod: .nndescent)
                r.exportKNNGraph = true
                r.knnGraph = g
                let y = r.fitTransform(x)
                eval(y)
                return (y, r.lastKNNGraph)
            }),
            ("TSNE", { g in
                let r = TSNE(
                    nComponents: 2, perplexity: 10, nIter: 40, randomState: 42, pcaDim: nil,
                    knnMethod: .nndescent)
                r.exportKNNGraph = true
                r.knnGraph = g
                let y = r.fitTransform(x)
                eval(y)
                return (y, r.lastKNNGraph)
            }),
            ("PaCMAP", { g in
                let r = PaCMAP(
                    nComponents: 2, nNeighbors: 10, numIters: (10, 10, 20), randomState: 42,
                    knnMethod: .nndescent)
                r.exportKNNGraph = true
                r.knnGraph = g
                let y = r.fitTransform(x)
                eval(y)
                return (y, r.lastKNNGraph)
            }),
            ("LocalMAP", { g in
                let r = LocalMAP(
                    nComponents: 2, nNeighbors: 10, numIters: (10, 10, 20), randomState: 42,
                    knnMethod: .nndescent, lowDistThres: 10.0)
                r.exportKNNGraph = true
                r.knnGraph = g
                let y = r.fitTransform(x)
                eval(y)
                return (y, r.lastKNNGraph)
            }),
            ("TriMap", { g in
                let r = TriMap(
                    nComponents: 2, nNeighbors: 10, nInliers: 8, nOutliers: 2, nRandom: 2,
                    nIters: 30, pcaDim: nil, randomState: 42, knnMethod: .nndescent)
                r.exportKNNGraph = true
                r.knnGraph = g
                let y = r.fitTransform(x)
                eval(y)
                return (y, r.lastKNNGraph)
            }),
            ("DREAMS", { g in
                let r = DREAMS(
                    nComponents: 2, perplexity: 10, nIter: 40, pcaDim: nil, randomState: 42,
                    knnMethod: .nndescent)
                r.exportKNNGraph = true
                r.knnGraph = g
                let y = r.fitTransform(x)
                eval(y)
                return (y, r.lastKNNGraph)
            }),
            ("CNE", { g in
                let r = CNE(
                    nComponents: 2, nNeighbors: 10, nIter: 40, pcaDim: nil, randomState: 42,
                    knnMethod: .nndescent)
                r.exportKNNGraph = true
                r.knnGraph = g
                let y = r.fitTransform(x)
                eval(y)
                return (y, r.lastKNNGraph)
            }),
        ]
    }

    private func maxDelta(_ a: MLXArray, _ b: MLXArray) -> Float {
        let av = a.asArray(Float.self)
        let bv = b.asArray(Float.self)
        guard av.count == bv.count else { return .infinity }
        var m: Float = 0
        for i in 0 ..< av.count { m = Swift.max(m, Swift.abs(av[i] - bv[i])) }
        return m
    }

    /// The k-NN build is itself bit-reproducible — the precondition for caching it.
    func testKNNBuildIsDeterministic() {
        let x = data(1200, 32)
        for k in [10, 30, 60] {
            let a = computeKNNGraph(x, k: k, method: .nndescent, randomState: 42)
            let b = computeKNNGraph(x, k: k, method: .nndescent, randomState: 42)
            XCTAssertEqual(a.indices, b.indices, "k=\(k): NNDescent indices differ across builds")
            XCTAssertEqual(a.distances, b.distances, "k=\(k): NNDescent distances differ across builds")
        }
        let e = computeKNNGraph(x, k: 10, method: .brute)
        let f = computeKNNGraph(x, k: 10, method: .brute)
        XCTAssertEqual(e.indices, f.indices)
        XCTAssertEqual(e.distances, f.distances)
    }

    /// Skipping the build must leave MLX's global PRNG exactly where a build would
    /// have: NNDescent seeds it and consumes one key split, and every later random draw
    /// in a fit (negative sampling, pair/triplet sampling, random init) reads from it.
    func testInjectionPreservesGlobalRandomState() {
        let x = data(900, 24)
        let spec = KNNGraphSpec(
            n: 900, k: 12, method: .nndescent, distanceKind: .euclidean, randomState: 42)
        let graph = computeKNNGraph(x, k: 12, method: .nndescent, randomState: 42)

        func drawAfter(_ g: KNNGraph?) -> [Float] {
            MLXRandom.seed(999)  // arbitrary prior state; the KNN step must overwrite it
            _ = resolveKNN(x, spec: spec, graph: g, export: false, verbose: false, onIteration: nil)
            let sample = MLXRandom.uniform(0.0 ..< 1.0, [8])
            eval(sample)
            return sample.asArray(Float.self)
        }
        XCTAssertEqual(drawAfter(nil), drawAfter(graph), "injection left a different PRNG state")
    }

    /// End-to-end bit-identity, with the (chaotic, scatter-add-nondeterministic)
    /// optimizer loop disabled: UMAP's spectral initialization is a pure function of the
    /// k-NN graph, so this compares the whole KNN → fuzzy simplicial set → init pipeline.
    func testInjectedGraphReproducesInitializationExactly() throws {
        let x = data(1200, 32)
        func run(_ g: KNNGraph?) -> (MLXArray, KNNGraph?) {
            let r = UMAP(
                nComponents: 2, nNeighbors: 10, nEpochs: 0, randomState: 42,
                knnMethod: .nndescent)
            r.exportKNNGraph = true
            r.knnGraph = g
            let y = r.fitTransform(x)
            eval(y)
            return (y, r.lastKNNGraph)
        }
        let (yBuild, exportedOpt) = run(nil)
        let exported = try XCTUnwrap(exportedOpt, "exportKNNGraph produced nothing")

        let (yBuild2, _) = run(nil)
        XCTAssertEqual(maxDelta(yBuild, yBuild2), 0, "control: build path is not deterministic")

        let (yInject, _) = run(exported)
        XCTAssertEqual(maxDelta(yBuild, yInject), 0, "injected graph changed the initialization")

        let restored = try KNNGraph(serialized: exported.serialized())
        XCTAssertEqual(restored, exported, "serialization round-trip lost data")
        let (yRestored, _) = run(restored)
        XCTAssertEqual(maxDelta(yBuild, yRestored), 0, "deserialized graph changed the initialization")
    }

    /// Full fits, every NNDescent consumer. The optimizers use GPU scatter-add, whose
    /// float accumulation order is nondeterministic, so a fit does not repeat itself bit
    /// for bit even without any caching — and small differences amplify over epochs. The
    /// bar therefore is: injecting must be indistinguishable from rebuilding, i.e. the
    /// build-vs-inject spread must sit inside the method's own run-to-run spread, and the
    /// k-NN graph it was handed must be identical.
    func testInjectedGraphMatchesWithinRunToRunNoise() throws {
        let x = data(1200, 32)
        print("method      build|build   inject|inject   build|inject")
        for (name, run) in cases(x) {
            let (yA, exportedOpt) = run(nil)
            let exported = try XCTUnwrap(exportedOpt, "\(name): exportKNNGraph produced nothing")
            let (yB, exported2) = run(nil)
            XCTAssertEqual(
                exported.indices, exported2?.indices, "\(name): k-NN indices are not deterministic")
            XCTAssertEqual(
                exported.distances, exported2?.distances,
                "\(name): k-NN distances are not deterministic")

            let (yC, _) = run(exported)
            let (yD, _) = run(exported)

            let noise = Swift.max(maxDelta(yA, yB), maxDelta(yC, yD))
            let cross = maxDelta(yA, yC)
            let pad = String(repeating: " ", count: Swift.max(0, 10 - name.count))
            print(String(format: "%@%@  %-12.6g  %-14.6g  %-12.6g",
                         name, pad, Double(noise), Double(maxDelta(yC, yD)), Double(cross)))

            XCTAssertFalse(MLX.any(yA .!= yA).item(Bool.self), "\(name): NaN in the embedding")
            if noise == 0 {
                XCTAssertEqual(cross, 0, "\(name): deterministic method changed under injection")
            } else {
                XCTAssertLessThanOrEqual(
                    cross, noise * 5,
                    "\(name): injection moved the embedding beyond run-to-run noise")
            }
        }
    }

    /// The exact brute-force path is trivially included: same export/inject contract.
    func testInjectionOnBruteForcePath() throws {
        let x = data(600, 16)
        func run(_ g: KNNGraph?) -> (MLXArray, KNNGraph?) {
            let r = UMAP(
                nComponents: 2, nNeighbors: 10, nEpochs: 40, randomState: 42, knnMethod: .brute)
            r.exportKNNGraph = true
            r.knnGraph = g
            let y = r.fitTransform(x)
            eval(y)
            return (y, r.lastKNNGraph)
        }
        let (y0, g0) = run(nil)
        let graph = try XCTUnwrap(g0)
        XCTAssertEqual(graph.method, .brute)
        XCTAssertNil(graph.randomState, "brute force is seedless")
        let (y0b, _) = run(nil)
        let (y1, _) = run(graph)
        let noise = maxDelta(y0, y0b)
        XCTAssertLessThanOrEqual(maxDelta(y0, y1), Swift.max(noise * 5, 0))
    }

    /// `computeKNNGraph` is the standalone build entry point and agrees with `computeKNN`.
    func testComputeKNNGraphMatchesComputeKNN() {
        let x = data(800, 24)
        let graph = computeKNNGraph(x, k: 8, method: .nndescent, randomState: 3)
        let (idx, dist) = computeKNN(x, k: 8, method: .nndescent, randomState: 3)
        XCTAssertEqual(graph.n, 800)
        XCTAssertEqual(graph.k, 8)
        XCTAssertEqual(graph.method, .nndescent)
        XCTAssertEqual(graph.randomState, 3)
        XCTAssertEqual(graph.indices, idx.asType(.int32).asArray(Int32.self))
        XCTAssertEqual(graph.distances, dist.asType(.float32).asArray(Float.self))
    }

    /// A graph built for the wrong `k` (or `n`, or distance convention) is rejected, not
    /// silently truncated.
    func testValidationRejectsMismatches() throws {
        let x = data(500, 16)
        let umap = UMAP(nComponents: 2, nNeighbors: 10, nEpochs: 10, randomState: 42,
                        knnMethod: .nndescent)
        let spec = umap.knnGraphSpec(nSamples: 500)
        XCTAssertEqual(spec.k, 10)
        XCTAssertEqual(spec.method, .nndescent)
        XCTAssertEqual(spec.distanceKind, .euclidean)
        XCTAssertEqual(spec.randomState, 42)

        // Wrong k.
        let wrongK = computeKNNGraph(x, k: 12, method: .nndescent, randomState: 42)
        XCTAssertThrowsError(try umap.fitTransform(x, knnGraph: wrongK)) { err in
            XCTAssertEqual(
                err as? KNNGraphError, .neighborCountMismatch(expected: 10, found: 12))
        }
        // Wrong n.
        let wrongN = computeKNNGraph(data(400, 16), k: 10, method: .nndescent, randomState: 42)
        XCTAssertThrowsError(try umap.fitTransform(x, knnGraph: wrongN)) { err in
            XCTAssertEqual(err as? KNNGraphError, .pointCountMismatch(expected: 500, found: 400))
        }
        // Wrong distance convention (a t-SNE graph handed to UMAP).
        let squared = computeKNNGraph(
            x, k: 10, method: .nndescent, returnEuclidean: false, randomState: 42)
        XCTAssertThrowsError(try umap.fitTransform(x, knnGraph: squared)) { err in
            XCTAssertEqual(
                err as? KNNGraphError,
                .distanceKindMismatch(expected: .euclidean, found: .squared))
        }
        // Wrong seed.
        let otherSeed = computeKNNGraph(x, k: 10, method: .nndescent, randomState: 7)
        XCTAssertThrowsError(try umap.fitTransform(x, knnGraph: otherSeed))
        // Wrong search method.
        let brute = computeKNNGraph(x, k: 10, method: .brute)
        XCTAssertThrowsError(try umap.fitTransform(x, knnGraph: brute)) { err in
            XCTAssertEqual(err as? KNNGraphError, .methodMismatch(expected: .nndescent, found: .brute))
        }
        // The matching graph goes through.
        let good = computeKNNGraph(x, k: 10, method: .nndescent, randomState: 42)
        XCTAssertNoThrow(try umap.fitTransform(x, knnGraph: good))
    }

    /// Each method reports the `k` it will actually request.
    func testKNNGraphSpecs() {
        let n = 30000  // above the auto threshold, so `.auto` resolves to nndescent
        XCTAssertEqual(UMAP(nNeighbors: 15, randomState: 42).knnGraphSpec(nSamples: n).method, .nndescent)
        XCTAssertEqual(UMAP(nNeighbors: 15, randomState: 42).knnGraphSpec(nSamples: n).k, 15)
        // PaCMAP asks for 50 extra neighbours for its scaled-distance reselection.
        XCTAssertEqual(PaCMAP(nNeighbors: 10).knnGraphSpec(nSamples: n).k, 60)
        XCTAssertEqual(LocalMAP(nNeighbors: 10).knnGraphSpec(nSamples: n).k, 60)
        // t-SNE / DREAMS derive k from perplexity and want squared distances.
        let tsne = TSNE(perplexity: 30).knnGraphSpec(nSamples: n)
        XCTAssertEqual(tsne.k, 90)
        XCTAssertEqual(tsne.distanceKind, .squared)
        XCTAssertEqual(DREAMS(perplexity: 30).knnGraphSpec(nSamples: n).k, 90)
        XCTAssertEqual(TriMap(nNeighbors: 12).knnGraphSpec(nSamples: n).k, 12)
        XCTAssertEqual(CNE(nNeighbors: 15).knnGraphSpec(nSamples: n).k, 15)
        // Below the threshold `.auto` stays exact, and the seed is dropped (unused).
        let small = UMAP(nNeighbors: 15, randomState: 42).knnGraphSpec(nSamples: 500)
        XCTAssertEqual(small.method, .brute)
        XCTAssertNil(small.randomState)
        // Specs are Hashable, so they can key a cache directly.
        XCTAssertEqual(
            UMAP(nNeighbors: 15, randomState: 42).knnGraphSpec(nSamples: n),
            UMAP(nNeighbors: 15, randomState: 42).knnGraphSpec(nSamples: n))
    }

    /// Serialization is a self-describing binary blob of the exact declared size.
    func testSerializationFormat() throws {
        let x = data(300, 8)
        let g = computeKNNGraph(x, k: 6, method: .nndescent, randomState: 11)
        let blob = g.serialized()
        XCTAssertEqual(blob.count, g.serializedByteCount)
        XCTAssertEqual(blob.count, 35 + 300 * 6 * 8)
        let back = try KNNGraph(serialized: blob)
        XCTAssertEqual(back, g)

        // Truncated / corrupt payloads are rejected.
        XCTAssertThrowsError(try KNNGraph(serialized: blob.prefix(100)))
        var flipped = blob
        flipped[0] = 0x00
        XCTAssertThrowsError(try KNNGraph(serialized: flipped))
    }

    /// The NNDescent progress hook fires once per iteration with sane fractions.
    func testNNDescentProgressHook() {
        let x = data(4000, 16)
        var calls: [(iter: Int, total: Int, frac: Float)] = []
        let nn = NNDescent(k: 10, randomState: 0)
        nn.onIteration = { i, t, f in calls.append((i, t, f)) }
        eval(nn.build(x).indices)

        XCTAssertGreaterThan(calls.count, 2, "progress hook barely fired")
        XCTAssertEqual(calls[0].iter, 0, "should announce the graph before the first join")
        XCTAssertEqual(calls[0].frac, 1.0)
        for (i, c) in calls.enumerated() {
            XCTAssertEqual(c.iter, i, "iterations must be consecutive from 0")
            XCTAssertEqual(c.total, 20)
            XCTAssertGreaterThanOrEqual(c.frac, 0)
            XCTAssertLessThanOrEqual(c.frac, 1)
        }
        XCTAssertLessThan(calls.last!.frac, calls[1].frac, "descent should be converging")
        print("NNDescent progress: " + calls.map {
            knnProgressLine(iteration: $0.iter, total: $0.total, updatedFraction: $0.frac)
        }.joined(separator: " | "))
    }

    /// …and reaches a method's `onPhase` as a human-readable line.
    func testProgressReachesOnPhase() {
        let x = data(3000, 16)
        var lines: [String] = []
        let r = UMAP(
            nComponents: 2, nNeighbors: 10, nEpochs: 10, randomState: 42, knnMethod: .nndescent)
        r.onPhase = { lines.append($0) }
        eval(r.fitTransform(x))

        let knnLines = lines.filter { $0.hasPrefix("KNN: iter") }
        XCTAssertFalse(knnLines.isEmpty, "no KNN iteration lines reached onPhase")
        XCTAssertTrue(knnLines[0].contains("% updating"), "unexpected format: \(knnLines[0])")
        print("onPhase KNN lines: \(knnLines)")
    }

    /// Injecting a graph must not re-run the build, and must not leave the k-NN arrays
    /// re-derived: the optimizer sees the stored values verbatim.
    func testInjectedGraphIsUsedVerbatim() throws {
        let x = data(700, 16)
        let g = computeKNNGraph(x, k: 8, method: .nndescent, randomState: 42)
        let spec = KNNGraphSpec(
            n: 700, k: 8, method: .nndescent, distanceKind: .euclidean, randomState: 42)
        let r = resolveKNN(
            x, spec: spec, graph: g, export: false, verbose: false, onIteration: nil)
        XCTAssertEqual(r.indices.asType(.int32).asArray(Int32.self), g.indices)
        XCTAssertEqual(r.distances.asType(.float32).asArray(Float.self), g.distances)
    }
}
