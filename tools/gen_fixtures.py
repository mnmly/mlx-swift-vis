#!/usr/bin/env python3
"""Generate numerical-parity fixtures from the Python mlx-vis reference.

Run from the Python project so `mlx_vis` is importable:

    cd ../../python/mlx-vis
    uv run python ../../swift/mlx-swift-vis/tools/gen_fixtures.py

Writes JSON to Tests/MLXVisTests/Fixtures/parity.json (relative to this script).

Only DETERMINISTIC stages are captured — these are the shared-core building blocks
every method depends on. Stochastic embeddings (UMAP/t-SNE/... optimization) cannot
bit-match across RNGs and are validated separately by shape/NaN tests.
"""
import json
import os
import numpy as np
import mlx.core as mx

from mlx_vis._knn import compute_knn
from mlx_vis.pca import PCA
from mlx_vis._normalize import normalize_input
from mlx_vis._umap.umap import UMAP

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Tests", "MLXVisTests", "Fixtures", "parity.json",
)

n, d, k = 200, 16, 10
rng = np.random.RandomState(1234)
# Anisotropic data with a clear spectral gap so the top-2 PCA subspace is
# well-conditioned (isotropic randn gives near-degenerate singular values, making
# the top-2 plane arbitrary and unstable across SVD implementations).
scales = np.array([8.0, 4.0] + [0.4] * (d - 2), dtype=np.float32)
X = (rng.randn(n, d).astype(np.float32) * scales)

knn_idx, knn_dist = compute_knn(X, k, method="brute", return_euclidean=True)
knn_idx_sq, knn_dist_sq = compute_knn(X, k, method="brute", return_euclidean=False)

pca2 = PCA(n_components=2).fit_transform(X)

norm_std = normalize_input(X.copy(), "standard")
norm_mm = normalize_input(X.copy(), "minmax")

a, b = UMAP._find_ab_params(1.0, 0.1)

# FFT repulsive parity: fixed 2D embedding + the reference's _fft_repulsive output.
from mlx_vis._tsne.tsne import TSNE
nf = 2000
rngf = np.random.RandomState(99)
Yf = (rngf.randn(nf, 2).astype(np.float32) * 10.0)
h = 1.0 / 3
fft_nodes = np.array([(0.5 + j) * h for j in range(3)], dtype=np.float32)
fft_denom = np.ones(3, dtype=np.float32)
for j in range(3):
    for kk in range(3):
        if kk != j:
            fft_denom[j] *= (fft_nodes[j] - fft_nodes[kk])
Zf, gradf = TSNE._fft_repulsive(mx.array(Yf), nf, fft_nodes, fft_denom, {})
mx.eval(Zf, gradf)

fixture = {
    "n": n, "d": d, "k": k,
    "X": X.reshape(-1).tolist(),
    "knn_indices": knn_idx.reshape(-1).astype(np.int32).tolist(),
    "knn_dists": knn_dist.reshape(-1).astype(np.float32).tolist(),
    "knn_dists_sq": knn_dist_sq.reshape(-1).astype(np.float32).tolist(),
    "pca2": np.asarray(pca2).reshape(-1).astype(np.float32).tolist(),
    "norm_standard": np.asarray(norm_std).reshape(-1).astype(np.float32).tolist(),
    "norm_minmax": np.asarray(norm_mm).reshape(-1).astype(np.float32).tolist(),
    "ab": [float(a), float(b)],
    "fft_n": nf,
    "fft_Y": Yf.reshape(-1).tolist(),
    "fft_Z": float(np.asarray(Zf).item()),
    "fft_grad": np.asarray(gradf).reshape(-1).astype(np.float32).tolist(),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(fixture, f)
print(f"wrote {OUT}: n={n} d={d} k={k} ab=({a:.4f},{b:.4f})")
