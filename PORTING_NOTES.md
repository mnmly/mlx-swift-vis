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
- **FFT repulsive gate (`Repulsion.swift`) is `n >= 4000`, 2D.** The FFT (FIt-SNE)
  path is O(n + ng²·log ng) with a fixed grid-cost floor (grid size tracks the live
  embedding extent, not n), so it only beats the exact O(n²) path once n²-work exceeds
  that floor. Release measurement (50 iters, same data): exact vs FFT is **2.3× (n=6000),
  3.8× (n=10000), 5.1× (n=14000)** in FFT's favour, with the crossover ≈ 3000. The gate
  was originally 16000, which left a large "dead zone" (~4k–16k points) on the slow exact
  path; 4000 sits just past the crossover. FFT's interpolation error is validated down to
  n=3000 by `CoreTests.testFFTRepulsiveMatchesExact`, so small-n keeps the exact (cheaper
  and more accurate) path.
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

## Known bottleneck: NNDescent KNN dominates large-n t-SNE/DREAMS

For `n ≳ 50k`, a t-SNE / DREAMS / CNE fit is **almost entirely the NNDescent KNN build**
— not the optimization loop, which most people assume is "the layout". Measured in
Release (`d = 30`, `perplexity = 30` ⇒ `k = 3·perplexity = 90`):

| Phase                    | n = 50k | n = 100k | n = 150k (extrapolated) |
|--------------------------|---------|----------|-------------------------|
| PCA                      | ~0 s    | ~0 s     | ~0 s                    |
| **NNDescent KNN**        | **4.9 s** | **16.6 s** | **~30–50 s**          |
| `buildPMatrix` (affinity)| 0.12 s  | 0.62 s   | ~1 s                    |
| optimization (500 iters) | 2.6 s   | 5.4 s    | ~8 s                    |

The FFT repulsive path already makes the optimization O(n); the KNN build is the wall.

**Why it's slow (root cause).** NNDescent converges fast (6–7 iterations at both 50k
and 100k), so it is *not* iterating too much. The cost is **per-iteration work scaling
super-linearly** (~3× per-iter cost for 2× the points), which is memory-bandwidth
bound, not FLOP bound. In the first 2–3 iterations `updateFrac ≈ 1.0`, so `scale = 1`
⇒ `mcNew = jNN = k`, and the neighbor-of-neighbor join in `build()` materialises a
candidate matrix of width **≈ k² = 8100 columns**. At 100k that is multi-GB
`(n, 8100)` int32/float32 arrays being gathered/sorted every iteration; they blow past
cache and saturate unified-memory bandwidth. `NNDescent.swift`'s header already flags
that the port drops the Python scalability tricks ("No fp16 matmul / low-dim
random-projection early iters … No active-point pruning") — those omissions are what
bites here.

**Optimization 1 — fp16 candidate distance gather (DONE, default on).**
`NNDescent.useFP16Dists` (default `true`) casts the candidate data to fp16 for the
`gatherDists` dot products, halving the bytes of the dominant `(n, c, d)` working set.
Because that path is memory-bandwidth bound, the win grows with n and turns into a
*cliff* avoidance at the top end (fp32 there spills into heavy unified-memory pressure):

| n     | fp32     | fp16    | speedup |
|-------|----------|---------|---------|
| 50k   | 4.9 s    | 4.3 s   | 1.13×   |
| 100k  | 14.9 s   | 11.5 s  | 1.30×   |
| 150k  | 106.6 s  | 21.2 s  | **5.0×** |

Distances are only used to *rank* candidates, so recall is essentially unchanged
(0.9972 → 0.9969 at n=20k, k=90). **Critical detail:** naive fp16 is catastrophic for
large-magnitude inputs — recall collapsed to ~1.5% at |x|~400 (fp16 overflow / precision
loss). `build()` therefore normalises `x` by its max-abs into a fp16-safe range, runs the
whole (scale-invariant) descent in scaled units, and multiplies the returned distances
back by `distScale`. With that, recall holds at 0.9969 for |x| up to ~2000. Guarded by
`CoreTests.testNNDescentRecall` (> 0.80) and the leak guard.

**Optimization 2 — bound candidate width (`max_candidates`, pynndescent-style): TRIED,
REJECTED.** Capping the k² neighbor-of-neighbor join by trimming expansion depth (`jNN`)
craters convergence: recall fell to 0.85 (cap 1024) / 0.59 (cap 512), and because the
descent then failed the `delta` convergence test it ran *all 20* iterations and got
*slower*, not faster. The wide early joins are load-bearing for fast convergence from a
random init. Do not re-attempt naive width truncation; a faithful new/old-marked
`max_candidates` (join only newly-changed neighbors) would be needed, and it barely helps
the dominant early iterations (where every edge is new) — low ROI.

**Optimization 3 — active-point pruning** (restrict later iters to changed rows) remains
*low* impact: the tail iterations are already cheap because `scale` shrinks the candidate
width. Skip unless a future workload shows a heavy tail.

Any further change must keep the `NNDescent.build` / `computeKNN` API and pass
`CoreTests.testNNDescentRecall` + `BenchmarkTests.testMemoryStabilityAllMethods`.

## Build / test

- Scheme is the **package name** `MLXVis`, not the directory name.
- `xcodebuild -scheme MLXVis -destination 'platform=macOS' -derivedDataPath .xcdd build`
- `... test` for the XCTest target. (`swift test` cannot load MLX's metallib.)
