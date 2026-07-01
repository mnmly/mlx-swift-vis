# MLXVis

Swift port of [mlx-vis](https://github.com/hanxiao/mlx-vis) — pure-MLX dimensionality
reduction for Apple Silicon, built on [mlx-swift](https://github.com/ml-explore/mlx-swift).

Ports UMAP, t-SNE, PaCMAP, LocalMAP, TriMap, DREAMS, CNE, MMAE, and NNDescent. All
computation runs on the Metal GPU through MLX. No scipy / sklearn equivalents — every
algorithm is reimplemented in Swift MLX.

> This is the **algorithms-only** port (the numerical core that produces embeddings).
> The Python project's GPU video renderer / plotting layer is not included.

<img width="600" height="335" alt="2D reduction of DINOv3 [CLS] features over Rauschenberg Foundation archival materials, computed with MLXVis" src="https://github.com/user-attachments/assets/8ed54022-31fe-45e1-a41b-3ee060ab07b6" />

*2D reduction of DINOv3 `[CLS]` features over archived materials from the [Robert Rauschenberg Foundation](https://rauschenbergfoundation.org). Non-commercial research visualization; not affiliated with or endorsed by the Foundation. Artworks © Robert Rauschenberg Foundation, used under the Foundation's [Fair Use Policy](https://rauschenbergfoundation.org/foundation/fair-use-policy).*

## Requirements

- macOS 14+ / iOS 17+ / visionOS 1+
- mlx-swift ≥ 0.31

## Install (SwiftPM)

```swift
.package(url: "https://github.com/mnmly/mlx-swift-vis.git", from: "0.1.0")
// target dependency: .product(name: "MLXVis", package: "mlx-swift-vis")
```

## Usage

Every method is a class with a `fitTransform(_:)` that takes and returns an `MLXArray`
of shape `(nSamples, nFeatures) -> (nSamples, nComponents)`.

```swift
import MLX
import MLXVis

let x = MLXRandom.normal([10_000, 128])   // (n, d)

// UMAP
let y1 = UMAP(nComponents: 2, nNeighbors: 15).fitTransform(x)

// t-SNE
let y2 = TSNE(nComponents: 2, perplexity: 30).fitTransform(x)

// PaCMAP / LocalMAP
let y3 = PaCMAP(nComponents: 2, nNeighbors: 10).fitTransform(x)
let y4 = LocalMAP(nComponents: 2, nNeighbors: 10, lowDistThres: 10.0).fitTransform(x)

// TriMap, DREAMS, CNE, MMAE
let y5 = TriMap(nComponents: 2).fitTransform(x)
let y6 = DREAMS(nComponents: 2).fitTransform(x)
let y7 = CNE(nComponents: 2).fitTransform(x)
let y8 = MMAE(nComponents: 2).fitTransform(x)

// Approximate KNN graph (auto-used by the methods above for n > 20000)
let (indices, distances) = NNDescent(k: 15).build(x)
```

Each initializer mirrors the corresponding Python `__init__` parameters (snake_case →
camelCase, e.g. `n_neighbors` → `nNeighbors`, `random_state` → `randomState`).

## Differences from the Python reference

These are the deliberate, documented deviations made during the port (see
[`PORTING_NOTES.md`](PORTING_NOTES.md) for the full operation-mapping table):

- **API boundary is `MLXArray`**, not numpy — no host round-trip at the edges.
- **PCA uses covariance SVD**, not `eigh` (mlx-swift 0.31 has no `eigh`). The covariance
  matrix is symmetric PSD, so `svd(cov)` yields eigenvectors in descending order.
- **t-SNE / DREAMS use the FFT-accelerated (FIt-SNE) repulsive path for `n ≥ 16000`
  (2D)**, like the Python reference, with an exact full/chunked fallback otherwise.
  The FFT port matches Python's `_fft_repulsive` within 1e-3 on identical input
  (same `libmlx` FFT). t-SNE on 16,000×20 runs ~0.3s for 30 iterations (Release).
- **NNDescent** ports the full descent loop (random init → neighbor-of-neighbor joins +
  reverse candidates → dedup/top-k → convergence). The fp16 / random-projection FLOP
  reductions in the Python version are omitted (performance-only; the graph is
  unchanged). Measured recall vs exact KNN: ~0.86 on a 2000×16 random benchmark.
- **Random sampling** is seeded from `randomState` but does not bit-match numpy's RNG, so
  stochastic methods will not be element-identical to Python — algorithmic behavior is
  preserved. (These methods are also not bit-reproducible run-to-run: GPU scatter-add is
  atomic/unordered. Tests assert stable *structure*, not exact positions.)
- **Performance:** the t-SNE/DREAMS repulsive kernel is `compile()`-fused (~1.5× in
  Release) and brute-force KNN is async-pipelined. The package is leak-checked
  (`activeMemory` stays flat across repeated fits — see `BenchmarkTests`).

## Build & test

```bash
xcodebuild -scheme MLXVis -destination 'platform=macOS' -derivedDataPath .xcdd build
xcodebuild -scheme MLXVis -destination 'platform=macOS' -derivedDataPath .xcdd test
```

The scheme name is `MLXVis` (the package name). `swift build` / `swift test` do **not**
work — they cannot load MLX's Metal library; use `xcodebuild`.

## License & attribution

`MLXVis` is licensed under the **Apache License, Version 2.0** — see [`LICENSE`](LICENSE).

It is a **Swift port of [mlx-vis](https://github.com/hanxiao/mlx-vis)** by
**Han Xiao**, which is also Apache-2.0. This is a derivative work; the upstream
attribution and a statement of the significant changes are in [`NOTICE`](NOTICE),
and the technical divergences (and what this port adds) are detailed above and in
[`PORTING_NOTES.md`](PORTING_NOTES.md). In short:

- **Ported:** all nine dimensionality-reduction algorithms (UMAP, t-SNE, PaCMAP,
  LocalMAP, TriMap, DREAMS, CNE, MMAE, NNDescent).
- **Changed:** MLXArray API surface; covariance-SVD PCA; seeded-but-not-numpy RNG.
- **Added:** `onEpoch`/`progressEvery` animation hooks, an `onPhase` setup-progress
  hook (KNN build, pair/graph sampling — the otherwise-silent work that dominates
  the wall clock on large inputs), FFT + `compile()` repulsive path, async KNN
  pipelining, a leak-checked benchmark harness, and shared affinity-graph /
  repulsion / optimizer modules for the t-SNE family.
- **Not ported:** the GPU rendering / plotting / video layer (algorithms only).

The [mlx-swift](https://github.com/ml-explore/mlx-swift) dependency is © Apple Inc.,
MIT-licensed.
