// Benchmark + memory-leak harness (library-only equivalent of a `bench` subcommand).
//
// Run in RELEASE for meaningful numbers (Debug is ~5x slower):
//   xcodebuild -scheme MLXVis -destination 'platform=macOS' \
//     -derivedDataPath .xcdd -configuration Release test \
//     -only-testing:MLXVisTests/BenchmarkTests
//
// - `testCompileSpeedup` toggles MLX's global compile flag to isolate the speedup
//   attributable to compiling t-SNE's O(n^2) repulsive kernel.
// - `testMemoryStability` loops a fit and watches GPU.activeMemory: a flat active
//   footprint across iterations means no leak (the larger cache/peak is MLX's
//   reusable buffer pool, not a leak).

import Dispatch
import Foundation
import MLX
import XCTest

import MLXVis

final class BenchmarkTests: XCTestCase {

    private func seconds(_ body: () -> Void) -> Double {
        let t0 = DispatchTime.now().uptimeNanoseconds
        body()
        let t1 = DispatchTime.now().uptimeNanoseconds
        return Double(t1 - t0) / 1e9
    }

    private func data(_ n: Int, _ d: Int) -> MLXArray {
        MLXRandom.seed(0)
        return MLXRandom.normal([n, d])
    }

    /// Compile should not regress and is expected to speed up the t-SNE repulsive path.
    func testCompileSpeedup() {
        let x = data(3000, 30)   // n^2 * 4 < 1e9 -> full (compiled) repulsive path
        let iters = 80

        func runTSNE() {
            let y = TSNE(nComponents: 2, perplexity: 30, nIter: iters,
                         randomState: 0, pcaDim: 20).fitTransform(x)
            eval(y)
        }

        // Warm up (first call pays one-time compilation + lazy graph build).
        MLX.compile(enable: true); runTSNE()
        let compiled = seconds { runTSNE() }

        MLX.compile(enable: false)
        runTSNE()  // warm
        let eager = seconds { runTSNE() }
        MLX.compile(enable: true)  // restore default

        print(String(format: "t-SNE %d iters @%dx%d — compiled: %.3fs, eager: %.3fs (%.2fx)",
                     iters, x.dim(0), x.dim(1), compiled, eager, eager / compiled))
        // Allow noise; assert compile is not a meaningful regression.
        XCTAssertLessThan(compiled, eager * 1.15, "compiled path regressed vs eager")
    }

    /// End-to-end smoke test of the FFT-accelerated path (gate is n >= 4000, 2D).
    func testFFTEndToEnd() {
        let n = 16000
        let x = data(n, 20)
        var y = MLXArray.zeros([1])
        let t = seconds {
            y = TSNE(nComponents: 2, perplexity: 30, nIter: 30, randomState: 0,
                     pcaDim: nil, knnMethod: .brute).fitTransform(x)
            eval(y)
        }
        print(String(format: "t-SNE FFT path @%dx20, 30 iters: %.2fs", n, t))
        XCTAssertEqual(y.shape, [n, 2])
        let nan = MLX.any(y .!= y).item(Bool.self)
        XCTAssertFalse(nan, "FFT path produced NaN")
    }

    /// Quantify the k-NN cache: how much of a fit's wall clock is the neighbour search,
    /// and how much of it a precomputed ``KNNGraph`` removes.
    ///
    /// Set `MLXVIS_BIG_BENCH=1` to add a 100k x 1024 case (needs ~64 GB headroom).
    func testKNNGraphCacheSpeedup() {
        var sizes: [(Int, Int)] = [(30_000, 256)]
        if ProcessInfo.processInfo.environment["MLXVIS_BIG_BENCH"] == "1" {
            sizes.append((100_000, 1024))
        }
        for (n, d) in sizes {
            let x = data(n, d)
            eval(x)
            let k = 15

            func umap() -> UMAP {
                UMAP(nComponents: 2, nNeighbors: k, nEpochs: 200, randomState: 42,
                     knnMethod: .nndescent)
            }

            var graph: KNNGraph?
            let tKNN = seconds {
                graph = computeKNNGraph(x, k: k, method: .nndescent, randomState: 42)
            }
            guard let graph else { XCTFail("no graph"); return }

            let tBuild = seconds { eval(umap().fitTransform(x)) }
            let tInject = seconds {
                eval(try! umap().fitTransform(x, knnGraph: graph))
            }

            let bytes = graph.serializedByteCount
            print(String(
                format: "KNN cache @%dx%d (k=%d): KNN build %.2fs | fit w/ build %.2fs"
                    + " | fit w/ cached graph %.2fs | saved %.2fs (%.1f%%) | graph %.1f MB",
                n, d, k, tKNN, tBuild, tInject, tBuild - tInject,
                (tBuild - tInject) / tBuild * 100, Double(bytes) / 1e6))

            XCTAssertLessThan(tInject, tBuild, "cached graph did not save time at n=\(n)")
        }
    }

    /// Run `work` `iters` times, clearing the cache between runs, and return the
    /// post-clear `activeMemory` sample after each run.
    private func activeMemorySamples(iters: Int, _ work: () -> Void) -> [Int] {
        var samples: [Int] = []
        for _ in 0..<iters {
            // Drain transient MLXArrays per iteration so the post-clear measurement
            // reflects only persistently-held state, not undrained autorelease temporaries.
            autoreleasepool { work() }
            // Wait for the stream so deferred buffer frees complete before measuring,
            // otherwise activeMemory races MLX's asynchronous deallocation.
            Stream.defaultStream(Device.gpu).synchronize()
            Memory.clearCache()
            samples.append(Memory.activeMemory)
        }
        return samples
    }

    /// Leak guard for EVERY method + KNN: repeated fits must not grow the active GPU
    /// footprint. Baseline is the sample after the 2nd run (the 1st pays one-time
    /// allocations); the last run must stay within 30% of it (flat = no leak; the
    /// larger cache/peak is MLX's reusable buffer pool, not a leak).
    func testMemoryStabilityAllMethods() {
        let x = data(900, 20)
        let cases: [(String, () -> Void)] = [
            ("KNN", { eval(bruteForceKNN(x, k: 10, returnEuclidean: true).indices) }),
            ("NNDescent", { eval(NNDescent(k: 10, randomState: 0).build(x).indices) }),
            ("UMAP", { eval(UMAP(nComponents: 2, nNeighbors: 15, nEpochs: 60, randomState: 0).fitTransform(x)) }),
            ("TSNE", { eval(TSNE(nComponents: 2, perplexity: 15, nIter: 50, randomState: 0, pcaDim: 15).fitTransform(x)) }),
            ("PaCMAP", { eval(PaCMAP(nComponents: 2, nNeighbors: 10, numIters: (10, 10, 20), randomState: 0).fitTransform(x)) }),
            ("LocalMAP", { eval(LocalMAP(nComponents: 2, nNeighbors: 10, numIters: (10, 10, 20), randomState: 0, lowDistThres: 10.0).fitTransform(x)) }),
            ("TriMap", { eval(TriMap(nComponents: 2, nNeighbors: 10, nInliers: 8, nOutliers: 2, nRandom: 2, nIters: 30, randomState: 0).fitTransform(x)) }),
            ("DREAMS", { eval(DREAMS(nComponents: 2, perplexity: 15, nIter: 50, pcaDim: 15, randomState: 0).fitTransform(x)) }),
            ("CNE", { eval(CNE(nComponents: 2, nNeighbors: 10, nIter: 50, pcaDim: 15, randomState: 0).fitTransform(x)) }),
            ("MMAE", { eval(MMAE(nComponents: 2, nEpochs: 5, batchSize: 64, pcaDim: 15, randomState: 0).fitTransform(x)) }),
        ]
        for (name, work) in cases {
            let s = activeMemorySamples(iters: 5, work)
            let baseline = Double(s[1])
            let last = Double(s[s.count - 1])
            print("\(name) activeMemory across fits (bytes): \(s)")
            XCTAssertLessThan(last, baseline * 1.30, "\(name): active GPU memory grew across fits: \(s)")
        }
    }
}
