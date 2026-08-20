# MLXVis

Swift port of the Python [mlx-vis](https://github.com/hanxiao/mlx-vis) library —
pure-MLX dimensionality reduction (UMAP, t-SNE, PaCMAP, LocalMAP, TriMap, DREAMS,
CNE, MMAE, NNDescent) on Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift).

See `PORTING_NOTES.md` for the Python→Swift operation mapping, parity strategy,
and the performance/memory methodology.

## Build & test

The scheme is the package name **`MLXVis`** (not the directory name). MLX needs the
Metal-capable toolchain at runtime, so use **`xcodebuild`** — `swift build` / `swift
test` cannot load MLX's `default.metallib`.

```bash
xcodebuild -scheme MLXVis -destination 'platform=macOS' -derivedDataPath .xcdd build
xcodebuild -scheme MLXVis -destination 'platform=macOS' -derivedDataPath .xcdd test
# Release (benchmarks / leak guard are meaningful only here). `SWIFT_ENABLE_TESTABILITY`
# is required: the test target uses `@testable import`, which Release otherwise disables.
xcodebuild -scheme MLXVis -destination 'platform=macOS' -derivedDataPath .xcdd \
  -configuration Release test -only-testing:MLXVisTests/BenchmarkTests \
  SWIFT_ENABLE_TESTABILITY=YES
```

`BenchmarkTests.testMemoryStabilityAllMethods` is the leak guard — it loops every
method + KNN and asserts `Memory.activeMemory` stays flat. Keep it green; the module
ships leak-free.

## Documentation

`MLXVis` ships DocC-generated reference docs (see `Sources/MLXVis/Documentation.docc/`
and `Scripts/build_docs.sh`). **`///` doc comments on public symbols are published**
to the static site at https://mnmly.github.io/mlx-swift-vis/ and (with
`EMIT_LLMS_TXT=1`) into `docs/llms.txt`.

When you add or modify a `public` declaration:

- Write a `///` doc comment: one-sentence summary, then a paragraph if the *why* is
  non-obvious. Don't restate the signature.
- Document each parameter with `- Parameter name:` using the **internal** name when
  there's an external label (DocC warns otherwise).
- Cross-reference with double-backtick links, e.g. `` ``PCA/fitTransform(_:)`` ``.
  DocC link syntax is signature-sensitive: `foo(_:)` ≠ `foo(_:_:)`.
- Add new top-level public symbols under the right `## Topics` group in
  `Sources/MLXVis/Documentation.docc/MLXVis.md`. Topics are organized by *user task*,
  not alphabetically.

Verify (needs a swift.org toolchain for the `llms.txt` export; `~/.swiftly`'s
main-snapshot has the flags — Xcode's docc does not):

```bash
EMIT_LLMS_TXT=1 Scripts/build_docs.sh   # exit 0, no new link/parameter warnings
```

## Conventions

- One Swift file per Python module (`Sources/MLXVis/Methods/<Method>.swift`); shared
  primitives in `Sources/MLXVis/Core/`.
- No `eigh` in mlx-swift — use the shared `pcaReduce` (covariance SVD).
- Numerical parity is checked against fixtures from `tools/gen_fixtures.py` (run under
  the Python project's `uv`) in `MLXVisTests/ParityTests`.
