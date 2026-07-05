# ``MLXVis``

Pure-MLX dimensionality reduction for Apple Silicon — UMAP, t-SNE, PaCMAP,
LocalMAP, TriMap, DREAMS, CNE, MMAE, and NNDescent, running on the Metal GPU.

## Overview

`MLXVis` is a Swift port of the Python [mlx-vis](https://github.com/hanxiao/mlx-vis)
library. Every algorithm is reimplemented on top of
[mlx-swift](https://github.com/ml-explore/mlx-swift) — no scipy/sklearn
equivalents — so all heavy computation runs on the Metal GPU through MLX.

Each method is a class with stored configuration and a single
`fitTransform(_:)` entry point that takes and returns an `MLXArray` of shape
`(nSamples, nFeatures) → (nSamples, nComponents)`. There is no numpy boundary:
inputs and outputs are MLX arrays.

```swift
import MLX
import MLXVis

let x = MLXRandom.normal([10_000, 128])          // (n, d)
let y = UMAP(nComponents: 2, nNeighbors: 15).fitTransform(x)   // (n, 2)
```

For `n > 20000`, the methods automatically use approximate ``NNDescent`` for the
k-nearest-neighbor graph; below that they use exact GPU brute force. t-SNE and
DREAMS use an FFT-accelerated repulsive force for large 2-D embeddings.

### Cancelling a fit

`fitTransform(_:)` is synchronous but honors **cooperative cancellation**: run it
inside a `Task` and cancel that task, and the optimization loop stops at the next
epoch and returns the **best-so-far embedding** (correctly shaped, mean-centered —
it never throws). This suits "good enough, stop now" UX and abandoning a fit with
the wrong parameters.

```swift
let task = Task { UMAP(nComponents: 2, nNeighbors: 15).fitTransform(x) }
// … later, e.g. the user taps Cancel:
task.cancel()
let partial = await task.value        // best-so-far embedding, shape (n, 2)
```

Cancellation is checked every epoch (pair it with ``UMAP/onEpoch`` for progressive
previews). It cannot interrupt an in-flight MLX evaluation, so a long setup step —
the KNN build in particular — runs to completion before the check takes effect;
cancelling during setup returns the initial embedding once that step finishes.

## Topics

### Manifold learning

- ``UMAP``
- ``TSNE``
- ``PaCMAP``
- ``LocalMAP``
- ``TriMap``
- ``DREAMS``
- ``CNE``
- ``MMAE``

### Nearest neighbors

- ``computeKNN(_:k:method:returnEuclidean:randomState:verbose:)``
- ``KNNMethod``
- ``NNDescent``
- ``bruteForceKNN(_:k:returnEuclidean:)``

### Preprocessing

- ``PCA``
- ``normalizeInput(_:method:)``
- ``Normalization``

### t-SNE family building blocks

The shared, reusable pieces behind t-SNE and DREAMS — exposed for composition and
direct testing.

- ``buildPMatrix(_:_:perplexity:n:)``
- ``reverseEdgeValues(rows:cols:vals:n:)``
- ``TSNERepulsion``
- ``fftRepulsiveGrad(_:n:)``
