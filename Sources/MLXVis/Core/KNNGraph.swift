// A serializable k-NN graph: export the (expensive) neighbour search once, hand it
// back on later fits so the build is skipped entirely.
//
// For large `n` the NNDescent build dominates the wall clock of every method, yet the
// graph only depends on (data, k, method, seed) — not on any optimizer knob. Caching
// it across runs turns a re-layout into pure optimization time.
//
// Bit-identity is by construction: a graph stores *exactly* the two arrays
// `computeKNN` returned (same `distanceKind`, no re-normalisation, no sqrt/square
// round-trip), and the injected path returns them unchanged. The one remaining side
// effect of a build — a single split of MLX's global PRNG key, consumed by NNDescent's
// random initialization — is replayed on the injected path so every *downstream*
// random draw (negative sampling, spectral/random init, …) matches as well.
//
// That makes everything up to and including the initialization bit-identical, which is
// as far as identity can be claimed: the optimizers themselves use GPU scatter-add,
// whose float accumulation order is nondeterministic, so no fit repeats itself bit for
// bit with or without a cache. See KNNGraphTests.

import Foundation
import MLX
import MLXRandom

/// Which distance convention a ``KNNGraph`` stores.
///
/// Graphs are not converted between conventions: `sqrt` followed by squaring is not
/// bit-exact, so a graph may only be injected into a fit that asks for the same kind.
/// UMAP, PaCMAP/LocalMAP, TriMap and CNE use ``euclidean``; t-SNE and DREAMS use
/// ``squared``.
public enum KNNDistanceKind: UInt8, Sendable, Codable, Hashable {
    /// Euclidean distances (`returnEuclidean: true`).
    case euclidean = 0
    /// Squared Euclidean distances (`returnEuclidean: false`).
    case squared = 1
}

/// The k-NN build a given configuration will perform, and the identity a precomputed
/// ``KNNGraph`` must match to be usable for it.
///
/// Every method exposes one via `knnGraphSpec(nSamples:)`. It is `Hashable`, so it
/// doubles as the parameter half of a cache key — pair it with a fingerprint of the
/// input matrix (and of any PCA preprocessing, which happens *before* the k-NN build).
public struct KNNGraphSpec: Sendable, Hashable, Codable {
    /// Number of points.
    public let n: Int
    /// Neighbours per point, already clamped to the search's effective value.
    public let k: Int
    /// The resolved search method — never ``KNNMethod/auto``.
    public let method: KNNMethod
    /// Distance convention the consuming method needs.
    public let distanceKind: KNNDistanceKind
    /// Seed NNDescent will use (`nil` for the exact brute-force path, which is seedless).
    public let randomState: Int?

    /// Resolve a requested configuration into the build that will actually run:
    /// ``KNNMethod/auto`` is decided by `n`, `k` is clamped the way the search clamps
    /// it, and the seed is normalised to the value NNDescent would default to.
    ///
    /// - Parameters:
    ///   - n: Number of points.
    ///   - requestedK: Neighbours asked for.
    ///   - requestedMethod: Search strategy, possibly ``KNNMethod/auto``.
    ///   - distanceKind: Distance convention the consumer needs.
    ///   - randomState: Seed, or `nil` for NNDescent's default.
    public init(
        n: Int,
        k requestedK: Int,
        method requestedMethod: KNNMethod,
        distanceKind: KNNDistanceKind,
        randomState: Int?
    ) {
        let resolved = resolveKNNMethod(requestedMethod, n: n)
        self.n = n
        self.method = resolved
        self.k = resolved == .nndescent ? Swift.min(requestedK, n - 1) : requestedK
        self.distanceKind = distanceKind
        self.randomState = resolved == .nndescent ? (randomState ?? 42) : nil
    }
}

/// Why a precomputed ``KNNGraph`` was rejected.
public enum KNNGraphError: Error, CustomStringConvertible, Equatable {
    /// The graph covers a different number of points than the fit.
    case pointCountMismatch(expected: Int, found: Int)
    /// The graph has a different neighbour count `k` than the fit needs.
    case neighborCountMismatch(expected: Int, found: Int)
    /// The graph stores the other distance convention.
    case distanceKindMismatch(expected: KNNDistanceKind, found: KNNDistanceKind)
    /// The graph was built by a different search method than the fit would use.
    case methodMismatch(expected: KNNMethod, found: KNNMethod)
    /// The graph was built with a different NNDescent seed.
    case randomStateMismatch(expected: Int?, found: Int?)
    /// `indices`/`distances` do not hold exactly `n * k` values.
    case payloadSizeMismatch(field: String, expected: Int, found: Int)
    /// A serialized payload could not be decoded.
    case corruptPayload(String)

    public var description: String {
        switch self {
        case let .pointCountMismatch(e, f):
            return "k-NN graph covers \(f) points, fit needs \(e)"
        case let .neighborCountMismatch(e, f):
            return "k-NN graph has k=\(f), fit needs k=\(e)"
        case let .distanceKindMismatch(e, f):
            return "k-NN graph stores \(f) distances, fit needs \(e)"
        case let .methodMismatch(e, f):
            return "k-NN graph was built with \(f), fit would use \(e)"
        case let .randomStateMismatch(e, f):
            return "k-NN graph was built with randomState \(String(describing: f)), "
                + "fit uses \(String(describing: e))"
        case let .payloadSizeMismatch(field, e, f):
            return "k-NN graph \(field) holds \(f) values, expected \(e)"
        case let .corruptPayload(why):
            return "k-NN graph payload is corrupt: \(why)"
        }
    }
}

/// A precomputed k-nearest-neighbor graph, as a plain Swift value that can be cached
/// on disk and injected into later fits to skip the neighbour search.
///
/// Obtain one from ``computeKNNGraph(_:k:method:returnEuclidean:randomState:verbose:onIteration:)``,
/// or from a fit by setting the method's `exportKNNGraph` flag and reading its
/// `lastKNNGraph` afterwards. Inject one with a method's
/// `fitTransform(_:knnGraph:)` overload (which validates first and throws on a
/// mismatch) or, unchecked, via its `knnGraph` property.
///
/// A graph is only valid for the exact build it came from: the same input matrix
/// **after any PCA preprocessing the method applies**, the same `k`, the same search
/// method, the same distance convention and the same seed. ``validate(against:)``
/// checks everything but the data itself — fingerprinting the input is the caller's
/// job, and belongs in the cache key alongside ``KNNGraphSpec``.
///
/// ```swift
/// let umap = UMAP(nComponents: 2, nNeighbors: 15, randomState: 42, pcaDim: 64)
/// let spec = umap.knnGraphSpec(nSamples: x.dim(0))
///
/// if let cached = try? KNNGraph(serialized: Data(contentsOf: url)),
///    (try? cached.validate(against: spec)) != nil {
///     y = try umap.fitTransform(x, knnGraph: cached)     // cache hit: no k-NN build
/// } else {
///     umap.exportKNNGraph = true
///     y = umap.fitTransform(x)
///     try umap.lastKNNGraph?.serialized().write(to: url) // cache fill
/// }
/// ```
public struct KNNGraph: Sendable, Equatable, Codable {
    /// Number of points.
    public let n: Int
    /// Neighbours per point.
    public let k: Int
    /// Neighbour ids, row-major `n * k`, ascending by distance within a row.
    public let indices: [Int32]
    /// Neighbour distances, row-major `n * k`, in ``distanceKind`` units.
    public let distances: [Float]
    /// Distance convention of ``distances``.
    public let distanceKind: KNNDistanceKind
    /// The search method that produced this graph — never ``KNNMethod/auto``.
    public let method: KNNMethod
    /// Seed NNDescent used, or `nil` when built by the seedless brute-force path.
    public let randomState: Int?

    /// Wrap host-side arrays (e.g. read back from a cache) as a graph.
    ///
    /// - Parameters:
    ///   - n: Number of points.
    ///   - k: Neighbours per point.
    ///   - indices: Row-major `n * k` neighbour ids.
    ///   - distances: Row-major `n * k` neighbour distances.
    ///   - distanceKind: Convention of `distances`.
    ///   - method: Search method that produced the graph.
    ///   - randomState: Seed used, if any.
    /// - Throws: ``KNNGraphError/payloadSizeMismatch(field:expected:found:)`` if either
    ///   payload is not exactly `n * k` long.
    public init(
        n: Int,
        k: Int,
        indices: [Int32],
        distances: [Float],
        distanceKind: KNNDistanceKind,
        method: KNNMethod,
        randomState: Int?
    ) throws {
        let want = n * k
        guard indices.count == want else {
            throw KNNGraphError.payloadSizeMismatch(
                field: "indices", expected: want, found: indices.count)
        }
        guard distances.count == want else {
            throw KNNGraphError.payloadSizeMismatch(
                field: "distances", expected: want, found: distances.count)
        }
        self.n = n
        self.k = k
        self.indices = indices
        self.distances = distances
        self.distanceKind = distanceKind
        self.method = method
        self.randomState = randomState
    }

    /// Capture the `(indices, distances)` pair a k-NN search returned.
    ///
    /// Both arrays are evaluated and copied to the host verbatim — float32 survives the
    /// round-trip exactly, so re-injecting this graph reproduces the arrays bit for bit.
    ///
    /// - Parameters:
    ///   - indices: `(n, k)` neighbour ids.
    ///   - distances: `(n, k)` neighbour distances.
    ///   - distanceKind: Convention of `distances`.
    ///   - method: Search method that produced them.
    ///   - randomState: Seed used, if any.
    public init(
        indices: MLXArray,
        distances: MLXArray,
        distanceKind: KNNDistanceKind,
        method: KNNMethod,
        randomState: Int?
    ) {
        precondition(indices.ndim == 2 && distances.ndim == 2, "k-NN arrays must be 2-D")
        precondition(indices.shape == distances.shape, "k-NN indices/distances shape mismatch")
        let idx = indices.asType(.int32)
        let dst = distances.asType(.float32)
        eval(idx, dst)
        self.n = indices.dim(0)
        self.k = indices.dim(1)
        self.indices = idx.asArray(Int32.self)
        self.distances = dst.asArray(Float.self)
        self.distanceKind = distanceKind
        self.method = method
        self.randomState = randomState
    }

    /// The neighbour ids as an `(n, k)` int32 `MLXArray`.
    public var indicesArray: MLXArray { MLXArray(indices, [n, k]) }

    /// The neighbour distances as an `(n, k)` float32 `MLXArray`.
    public var distancesArray: MLXArray { MLXArray(distances, [n, k]) }

    /// Size of ``serialized()`` output, in bytes.
    public var serializedByteCount: Int { KNNGraph.headerSize + n * k * 8 }

    /// Check that this graph describes the build `spec` asks for.
    ///
    /// - Parameter spec: The build a method is about to perform, from its
    ///   `knnGraphSpec(nSamples:)`.
    /// - Throws: A ``KNNGraphError`` naming the first field that disagrees. Nothing is
    ///   truncated, padded or converted to make a mismatched graph fit.
    public func validate(against spec: KNNGraphSpec) throws {
        guard n == spec.n else {
            throw KNNGraphError.pointCountMismatch(expected: spec.n, found: n)
        }
        guard k == spec.k else {
            throw KNNGraphError.neighborCountMismatch(expected: spec.k, found: k)
        }
        guard distanceKind == spec.distanceKind else {
            throw KNNGraphError.distanceKindMismatch(
                expected: spec.distanceKind, found: distanceKind)
        }
        guard method == spec.method else {
            throw KNNGraphError.methodMismatch(expected: spec.method, found: method)
        }
        guard randomState == spec.randomState else {
            throw KNNGraphError.randomStateMismatch(expected: spec.randomState, found: randomState)
        }
        let want = n * k
        guard indices.count == want else {
            throw KNNGraphError.payloadSizeMismatch(
                field: "indices", expected: want, found: indices.count)
        }
        guard distances.count == want else {
            throw KNNGraphError.payloadSizeMismatch(
                field: "distances", expected: want, found: distances.count)
        }
    }

    // MARK: - Binary serialization

    // "MLXVKNN" + format version. Little-endian throughout (all supported platforms).
    private static let magic: [UInt8] = Array("MLXVKNN\u{01}".utf8)
    private static let headerSize = 8 + 8 + 8 + 1 + 1 + 1 + 8  // magic, n, k, method, kind, hasSeed, seed

    /// Encode as a flat little-endian binary blob: a fixed header followed by the raw
    /// int32 index plane and float32 distance plane.
    ///
    /// Preferred over `Codable` for large graphs — at `n = 261k, k = 15` this is ~31 MB
    /// of contiguous bytes, versus a JSON encoding orders of magnitude larger.
    public func serialized() -> Data {
        var data = Data(capacity: serializedByteCount)
        data.append(contentsOf: KNNGraph.magic)
        KNNGraph.appendLE(&data, Int64(n))
        KNNGraph.appendLE(&data, Int64(k))
        data.append(KNNGraph.methodCode(method))
        data.append(distanceKind.rawValue)
        data.append(randomState == nil ? 0 : 1)
        KNNGraph.appendLE(&data, Int64(randomState ?? 0))
        indices.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        distances.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        return data
    }

    /// Decode a blob produced by ``serialized()``.
    ///
    /// - Parameter serialized: Bytes from a previous ``serialized()`` call.
    /// - Throws: ``KNNGraphError/corruptPayload(_:)`` on a bad magic, an unknown enum
    ///   code, or a truncated payload.
    public init(serialized: Data) throws {
        let bytes = [UInt8](serialized)
        guard bytes.count >= KNNGraph.headerSize,
            Array(bytes[0 ..< 8]) == KNNGraph.magic
        else {
            throw KNNGraphError.corruptPayload("bad magic or truncated header")
        }
        let nV = Int(KNNGraph.readLE(bytes, 8))
        let kV = Int(KNNGraph.readLE(bytes, 16))
        guard nV >= 0, kV >= 0 else { throw KNNGraphError.corruptPayload("negative n/k") }
        guard let m = KNNGraph.method(code: bytes[24]) else {
            throw KNNGraphError.corruptPayload("unknown method code \(bytes[24])")
        }
        guard let kind = KNNDistanceKind(rawValue: bytes[25]) else {
            throw KNNGraphError.corruptPayload("unknown distance kind \(bytes[25])")
        }
        let seed: Int? = bytes[26] == 0 ? nil : Int(KNNGraph.readLE(bytes, 27))

        let count = nV * kV
        let base = KNNGraph.headerSize
        guard bytes.count == base + count * 8 else {
            throw KNNGraphError.corruptPayload(
                "expected \(base + count * 8) bytes, got \(bytes.count)")
        }
        var idx = [Int32](repeating: 0, count: count)
        var dst = [Float](repeating: 0, count: count)
        bytes.withUnsafeBytes { raw in
            idx.withUnsafeMutableBytes { out in
                if count > 0 {
                    out.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[base ..< base + count * 4]))
                }
            }
            dst.withUnsafeMutableBytes { out in
                if count > 0 {
                    let lo = base + count * 4
                    out.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[lo ..< lo + count * 4]))
                }
            }
        }
        try self.init(
            n: nV, k: kV, indices: idx, distances: dst,
            distanceKind: kind, method: m, randomState: seed)
    }

    private static func methodCode(_ m: KNNMethod) -> UInt8 {
        switch m {
        case .auto: return 0
        case .brute: return 1
        case .nndescent: return 2
        }
    }

    private static func method(code: UInt8) -> KNNMethod? {
        switch code {
        case 0: return .auto
        case 1: return .brute
        case 2: return .nndescent
        default: return nil
        }
    }

    private static func appendLE(_ data: inout Data, _ v: Int64) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    private static func readLE(_ bytes: [UInt8], _ offset: Int) -> Int64 {
        var v: UInt64 = 0
        for i in 0 ..< 8 { v |= UInt64(bytes[offset + i]) << (8 * i) }
        return Int64(bitPattern: v)
    }
}
