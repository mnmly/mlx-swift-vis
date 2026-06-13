# Porting notes: mlx-vis (Python) → MLXVis (Swift)

Validated against **mlx-swift 0.31.4**. Source of truth is the Python package
`mlx_vis` (one Swift file per Python module).

## Design decisions

- **API shape**: each method is a `public final class` with stored config and a
  `func fitTransform(_ x: MLXArray) -> MLXArray`. Input/output are `MLXArray`
  (no numpy/Apple-native), unlike the Python `np.ndarray` boundary.
- **No `eigh`**: mlx-swift 0.31.4 has `MLXLinalg.svd` but **no `eigh`**. The
  Python PCA preprocessing uses `mx.linalg.eigh(cov)` (ascending) then reverses.
  Covariance is symmetric PSD, so `svd(cov).2` (Vt) already gives eigenvectors in
  descending order. Use the shared `pcaReduce(_:dim:)` helper.
- **CPU index work**: where Python drops to numpy for index compaction
  (`np.nonzero`, edge pruning, `epochs_per_sample` scheduling), pull to Swift with
  `arr.asArray(Int32.self)` / `.asArray(Float.self)`, do the loop in plain Swift,
  and wrap results back with `MLXArray(swiftArray)`. This matches the skill's
  guidance and keeps stochastic-optimization parity intent without fighting the
  graph.

## Operation mapping (Python `mx.` → Swift)

| Python | Swift |
| --- | --- |
| `x[:, None]` | `x.expandedDimensions(axis: 1)` |
| `x[:, 1:]` | `x[0..., 1...]` |
| `x[:, :k]` | `x[0..., ..<k]` |
| `a < b` (elementwise) | `a .< b` (also `.>`, `.==`, `.>=`, `.!=`) |
| `a & b` (bool) | `a & b` |
| `~a` (bool not) | `.!a` |
| `mx.where(c, a, b)` | `MLX.where(c, a, b)` |
| `mx.exp/sqrt/abs/pow` | `MLX.exp` / `MLX.sqrt` / `.abs()` / `MLX.pow` |
| `mx.maximum/minimum` | `maximum` / `minimum` (free funcs) |
| `mx.clip(x, lo, hi)` | `clip(x, min: lo, max: hi)` |
| `x // y` (int floordiv) | `floorDivide(x, y)` |
| `mx.sum(x, axis=1, keepdims=True)` | `x.sum(axis: 1, keepDims: true)` |
| `mx.argsort(x, axis=1)` | `argSort(x, axis: 1)` |
| `mx.take_along_axis(x, i, 1)` | `takeAlong(x, i, axis: 1)` |
| `x[idx_array]` (fancy) | `x[idxArray]` |
| `np.repeat(arange(n), k)` | `repeated(MLXArray(Int32(0)..<Int32(n)), count: k, axis: 0)` |
| `mx.concatenate([...], axis=0)` | `concatenated([...], axis: 0)` |
| `mx.eye(n)` | use a comparison mask (see KNN) or `MLXArray.eye(n)` |
| `x.at[idx].add(v)` | `x.at[idx].add(v)` (also `.subtract`, `.multiply`); for columns `x.at[0..., j].add(v)` |
| `mx.zeros_like(x)` | `MLXArray.zeros(like: x)` |
| `mx.random.normal((n,k))` | `MLXRandom.normal([n, k])` |
| `mx.random.randint(0,n,(m,))` | `MLXRandom.randInt(0 ..< n, [m])` |
| `mx.random.seed(s)` | `MLXRandom.seed(UInt64(s))` |
| `mx.linalg.svd(a, stream=mx.cpu)` | `MLXLinalg.svd(a, stream: .cpu)` → `(U, S, Vt)` |
| `mx.fft.rfft2 / irfft2` | `MLXFFT.rfft2` / `MLXFFT.irfft2` (import MLXFFT) |
| `arr.item()` | `arr.item(Float.self)` / `.item(Bool.self)` |
| `bool(mx.all(x))` | `all(x).item(Bool.self)` |

## Numerical parity

`tools/gen_fixtures.py` (run under the Python project's `uv`) emits
`Tests/MLXVisTests/Fixtures/parity.json` from the reference, captured at the
**deterministic** boundaries that every method shares. `ParityTests.swift` checks:

| Stage | Tolerance | Notes |
| --- | --- | --- |
| `normalizeInput` (standard, minmax) | abs < 1e-4 | exact |
| `findABParams` | abs < 1e-3 | exact |
| brute-force KNN indices | exact (0 mismatches) | anisotropic fixture → no ties |
| brute-force KNN distances (euclid + squared) | abs < 1e-3 / 1e-2 | |
| PCA | rel < 1e-3 | compared via the sign/rotation-invariant pairwise-distance matrix; fixture has a clear spectral gap so the top-2 subspace is well-conditioned |

Stochastic embeddings (the optimization loops) are **not** parity-checked — RNG isn't
numpy-identical, so they can't bit-match; they're covered by shape/NaN tests.

## Performance (compile) + memory

- **t-SNE / DREAMS repulsive kernel is `compile`d** (fixed-shape, called every
  epoch). Measured ~**1.5× faster** vs eager at 3000×30 in Release; the win grows
  with n since the O(n²) kernel dominates. The surrounding matmuls aren't sped by
  compile, so don't expect more than ~2×.
- **UMAP's per-edge SGD step is intentionally NOT compiled** — its scatter-add has a
  variable-length index (active-edge set changes per epoch). `compile(shapeless:)`
  cannot shape-infer the scatter (`Scatter Sum cannot infer output shapes`), and
  non-shapeless would recompile every epoch. The step is cheap relative to the graph
  build anyway.
- **Async KNN pipelining**: `bruteForceKNN` schedules each distance chunk with
  `asyncEval` (not a blocking `eval`), so chunk N+1's matmul overlaps chunk N's
  readback and the large `(chunk × n)` intermediate is freed before the next
  iteration (bounded peak memory). Measurably faster on the large-n path.
- **UMAP scheduler** keeps its per-epoch active-edge index sets host-side
  (`[[Int32]]`) and materializes each `MLXArray` lazily inside the loop, instead of
  holding `nEpochs` index arrays on-GPU for the whole optimization — flat peak memory.

### No memory leak — verified across the whole module

`BenchmarkTests.testMemoryStabilityAllMethods` loops **every** method + KNN/NNDescent
5× each and asserts `Memory.activeMemory` stays flat. With each iteration wrapped in
`autoreleasepool` and a `Stream.defaultStream(Device.gpu).synchronize()` before
measuring (so deferred frees complete and the read doesn't race MLX's async
deallocation), every method holds **flat at ~82 KB** across all fits — zero
accumulation:

```
KNN/NNDescent/UMAP/TSNE/PaCMAP/LocalMAP/TriMap/DREAMS/CNE/MMAE
  activeMemory: [81936, 81936, 81936, 81936, 81936]   (constant)
```

Gotcha learned: a naïve `activeMemory` snapshot right after `clearCache()` is a
**false leak signal** — undrained autorelease temporaries and in-flight async frees
make it bounce non-monotonically (e.g. `[81936, 3041740, 1173916, 439500, ...]`). A
real leak grows monotonically and unbounded; the bounded bounce was measurement
artifact, removed by the autoreleasepool + stream-sync above. The larger cache/peak
footprint is MLX's reusable buffer pool, not a leak; long-lived consumers can bound it
with `Memory.set(cacheLimit:)`.

- Run benchmarks in **Release** (`-configuration Release`); Debug is ~5× slower.

## Build / test

- Scheme is the **package name** `MLXVis`, not the directory name.
- `xcodebuild -scheme MLXVis -destination 'platform=macOS' -derivedDataPath .xcdd build`
- `... test` for the XCTest target. (`swift test` cannot load MLX's metallib.)
