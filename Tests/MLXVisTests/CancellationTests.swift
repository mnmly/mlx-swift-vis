// Cooperative cancellation: a fit run inside a cancelled Task stops early and returns
// a valid best-so-far embedding (never throws, never hangs).
import MLX
import XCTest

@testable import MLXVis

final class CancellationTests: XCTestCase {
    // Written from the fit's Task, read only after `await task.value` (which
    // happens-after the task completes) — no concurrent access, so no data race.
    private final class EpochCounter: @unchecked Sendable { var last = -1 }

    private func data(_ n: Int, _ d: Int) -> MLXArray {
        MLXRandom.seed(0)
        return MLXRandom.normal([n, d])
    }

    /// Run `fit` inside a Task that is guaranteed cancelled before its optimization
    /// loop starts, then return the produced shape and the last epoch the loop reached.
    private func runCancelled(_ fit: @escaping @Sendable (EpochCounter) -> MLXArray) async -> (shape: [Int], lastEpoch: Int) {
        let counter = EpochCounter()
        let task = Task { () -> [Int] in
            // Spin until this Task is cancelled, so the synchronous fit below runs with
            // `Task.isCancelled == true` from the first epoch — deterministic early exit.
            while !Task.isCancelled { await Task.yield() }
            let y = fit(counter)
            eval(y)
            return y.shape
        }
        task.cancel()
        let shape = await task.value
        return (shape, counter.last)
    }

    func testTSNECancellation() async {  // shared TSNEOptimizer loop (also covers DREAMS)
        let x = data(2000, 16)
        let (shape, last) = await runCancelled { c in
            let m = TSNE(nComponents: 2, perplexity: 15, nIter: 1000, randomState: 0, pcaDim: 8)
            m.progressEvery = 1
            m.onEpoch = { i, _, _ in c.last = i }
            return m.fitTransform(x)
        }
        XCTAssertEqual(shape, [2000, 2])
        XCTAssertEqual(last, 0, "loop should break at epoch 0 when pre-cancelled")
    }

    func testUMAPCancellation() async {  // UMAP's own SGD loop
        let x = data(2000, 16)
        let (shape, last) = await runCancelled { c in
            let m = UMAP(nComponents: 2, nNeighbors: 15, nEpochs: 500, randomState: 0)
            m.progressEvery = 1
            m.onEpoch = { i, _, _ in c.last = i }
            return m.fitTransform(x)
        }
        XCTAssertEqual(shape, [2000, 2])
        XCTAssertEqual(last, 0, "loop should break at epoch 0 when pre-cancelled")
    }

    func testMMAECancellation() async {  // breaks training, still encodes -> valid shape
        let x = data(2000, 16)
        let (shape, _) = await runCancelled { _ in
            MMAE(nComponents: 2, nEpochs: 50, batchSize: 64, pcaDim: 8, randomState: 0).fitTransform(x)
        }
        XCTAssertEqual(shape, [2000, 2])
    }
}
