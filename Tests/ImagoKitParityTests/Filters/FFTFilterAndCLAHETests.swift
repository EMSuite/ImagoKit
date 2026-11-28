// The FFT filter and CLAHE against kikuchipy 0.13.1 (scikit-image 0.26.0 for CLAHE), from the four bundles under
// ImagoFiltersTests/Fixtures/kikuchipy (moved from Electron Microscopy Studio's SEMPatternFilterParityTests with equal sha256;
// Tests/Fixtures/README.md). Each bundle carries its input patterns (uint8, 60 x 60) and kikuchipy's own filters.
//  - `FFTFilter` in the frequency domain (shift on and off, kikuchipy's lowpass x highpass transfer function) and the spatial
//    domain (5 x 5 Gaussian, 3 x 4 asymmetric kernel): values before the cast within the bundle's 1e-5 relative or 1e-3 grey
//    levels, cast patterns within one grey level on at most 1 % of the pixels (measured at most 0.22 %, the asymmetric kernel on
//    nickel pattern (0, 0); kikuchipy's FFTs are float32). Pattern (0, 0) of nickel_ebsd_large through the guide's filter
//    (the app's figure F4) on at most 0.2 %.
//  - `AdaptiveHistogramEqualization.equalize`: exact (three kernel / clip limit / bin settings).
// The filters read kikuchipy's own transfer function and Gaussian kernel from the bundle (`transfer_function`,
// `window_gaussian_5x5_std1`), the inputs its runs used.
// Red after a change to the shift layout, a kernel flip or tap offset, the CLAHE quantisation, padding, clip redistribution or
// float32 accumulation: whole grey levels move across the patterns.
import CryptoKit
import Foundation
import ImagoKitTestSupport
import ImagoCore
import ImagoFilters
import Testing

/// One reference bundle: manifest.json plus raw little-endian arrays (C order, no header), as the app's
/// tools/parity/sem/lib/refbundle.py writes them.
struct KikuchipyBundle {
    struct Entry: Decodable, Sendable {
        let file: String
        let dtype: String
        let shape: [Int]
        let sha256: String
        let rtol: Double
        let atol: Double
    }

    struct Manifest: Decodable, Sendable {
        let caseName: String
        let arrays: [String: Entry]
        enum CodingKeys: String, CodingKey {
            case arrays
            case caseName = "case"
        }
    }

    struct Loaded: Sendable {
        let name: String
        let shape: [Int]
        let values: [Double]
        let rtol: Double
        let atol: Double
    }

    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    static let names = ["fft-filter-kikuchipy-h5ebsd", "fft-filter-nickel-ebsd-large", "clahe-kikuchipy-h5ebsd", "clahe-nickel-ebsd-large"]

    let directory: URL
    let manifest: Manifest

    init(_ name: String) throws {
        directory = FixtureDirs.fixtures(of: "ImagoFiltersTests").appendingPathComponent("kikuchipy").appendingPathComponent(name)
        manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard manifest.caseName == "m2/" + name else { throw Failure(description: "\(name): manifest names \(manifest.caseName)") }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The array by name, its bytes checked against the manifest's sha256 and count, widened to Double (exact for every dtype
    /// the bundles hold).
    func array(_ name: String) throws -> Loaded {
        guard let entry = manifest.arrays[name] else { throw Failure(description: "\(manifest.caseName): no array \(name)") }
        let data = try Data(contentsOf: directory.appendingPathComponent(entry.file))
        let digest = Self.sha256(data)
        guard digest == entry.sha256 else {
            throw Failure(description: "\(manifest.caseName)/\(name): sha256 \(digest), manifest \(entry.sha256)")
        }
        let count = entry.shape.reduce(1, *)
        let size: Int
        switch entry.dtype {
        case "uint8": size = 1
        case "float32": size = 4
        case "float64": size = 8
        default: throw Failure(description: "\(manifest.caseName)/\(name): dtype \(entry.dtype)")
        }
        guard data.count == count * size else { throw Failure(description: "\(manifest.caseName)/\(name): \(data.count) bytes") }
        let values: [Double] = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [Double] in
            (0..<count).map { i -> Double in
                switch size {
                case 1: return Double(raw[i])
                case 4: return Double(Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self))))
                default: return Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: i * 8, as: UInt64.self)))
                }
            }
        }
        return Loaded(name: name, shape: entry.shape, values: values, rtol: entry.rtol, atol: entry.atol)
    }

    /// The bundle's input patterns as uint8: the data, the pattern count and the pattern shape (the last two axes).
    func patterns() throws -> (data: [UInt8], count: Int, rows: Int, columns: Int, shape: [Int]) {
        let a = try array("patterns")
        let rows = a.shape[a.shape.count - 2], columns = a.shape[a.shape.count - 1]
        return (a.values.map { UInt8($0) }, a.values.count / (rows * columns), rows, columns, a.shape)
    }
}

/// numpy.testing.assert_allclose with the bundle's tolerance: |a − b| <= atol + rtol · |b|, b the reference.
private func expectClose(_ got: [Double], _ reference: KikuchipyBundle.Loaded, _ label: String) {
    #expect(got.count == reference.values.count, "\(label): \(got.count) values, reference \(reference.values.count)")
    var bad = 0, worst = 0.0, worstIndex = -1
    for (i, (a, b)) in zip(got, reference.values).enumerated() {
        let excess = abs(a - b) - (reference.atol + reference.rtol * abs(b))
        if !(excess <= 0) {
            bad += 1
            if excess > worst || worstIndex < 0 { worst = excess; worstIndex = i }
        }
    }
    #expect(bad == 0, "\(label): \(bad) of \(reference.values.count) outside rtol \(reference.rtol), atol \(reference.atol); worst at \(worstIndex)")
}

/// Cast patterns: every pixel within one grey level of the reference, at most `fraction` of them off at all.
private func expectWithinOneGrey(_ ours: [UInt8], _ reference: KikuchipyBundle.Loaded, fraction: Double, _ label: String) {
    #expect(ours.count == reference.values.count, "\(label): count")
    var off = 0, worst = 0
    for (a, b) in zip(ours, reference.values) {
        let d = abs(Int(a) - Int(b))
        if d > 0 { off += 1 }
        worst = max(worst, d)
    }
    #expect(worst <= 1, "\(label): a pixel is \(worst) grey levels off")
    #expect(Double(off) <= fraction * Double(reference.values.count), "\(label): \(off) of \(reference.values.count) pixels one grey level off")
}

@Test(arguments: KikuchipyBundle.names)
func everyBundleFileHasItsManifestHash(_ name: String) throws {
    let bundle = try KikuchipyBundle(name)
    let files = try FileManager.default.contentsOfDirectory(atPath: bundle.directory.path).filter { $0 != "manifest.json" && !$0.hasPrefix(".") }
    #expect(Set(files) == Set(bundle.manifest.arrays.values.map(\.file)), "\(name): the files are the manifest's arrays")
    for key in bundle.manifest.arrays.keys { _ = try bundle.array(key) }
}

// MARK: - FFT filter

private func fftRuns(_ bundle: KikuchipyBundle) throws -> [(name: String, domain: FFTFilter.Domain)] {
    let transfer = try bundle.array("transfer_function").values
    let gaussian = try bundle.array("window_gaussian_5x5_std1").values
    let asymmetric: [Double] = [0, 1, 2, 3, 4, 5, 6, 7, -1, -2, 0.5, 1.5]
    return [("frequency_shift", FFTFilter.Domain.frequency(transferFunction: transfer, shift: true)),
            ("frequency_noshift", FFTFilter.Domain.frequency(transferFunction: transfer, shift: false)),
            ("spatial_gaussian", FFTFilter.Domain.spatial(kernel: gaussian, rows: 5, columns: 5)),
            ("spatial_asymmetric", FFTFilter.Domain.spatial(kernel: asymmetric, rows: 3, columns: 4))]
}

@Test(arguments: ["fft-filter-kikuchipy-h5ebsd", "fft-filter-nickel-ebsd-large"])
func fftFilterMatchesKikuchipy(_ name: String) throws {
    let bundle = try KikuchipyBundle(name)
    let p = try bundle.patterns()
    let pixels = p.rows * p.columns
    for run in try fftRuns(bundle) {
        var values: [Double] = [], cast: [UInt8] = []
        for i in 0..<p.count {
            let one = Array(p.data[(i * pixels)..<((i + 1) * pixels)])
            values += try FFTFilter.rescaledValues(one, rows: p.rows, columns: p.columns, domain: run.domain)
            cast += try FFTFilter.filter(one, rows: p.rows, columns: p.columns, domain: run.domain)
        }
        let reference = try bundle.array("\(run.name)_values")
        #expect(reference.shape == p.shape)
        expectClose(values, reference, "\(name) \(run.name) values")
        try expectWithinOneGrey(cast, bundle.array(run.name), fraction: 0.01, "\(name) \(run.name)")
    }
}

/// The app's north-star figure F4: pattern (0, 0) of nickel_ebsd_large through kikuchipy's lowpass (cutoff 22, width 10) x
/// highpass (cutoff 3, width 2) transfer function, shift on, as kikuchipy's pattern; and the `Micrograph` call gives the same.
@Test func fftFilterOfNickelPatternZeroAsKikuchipy() throws {
    let bundle = try KikuchipyBundle("fft-filter-nickel-ebsd-large")
    let p = try bundle.patterns()
    let domain = try FFTFilter.Domain.frequency(transferFunction: bundle.array("transfer_function").values, shift: true)
    let one = Array(p.data[0..<(p.rows * p.columns)])
    let ours = try FFTFilter.filter(one, rows: p.rows, columns: p.columns, domain: domain)
    try expectWithinOneGrey(ours, bundle.array("frequency_shift"), fraction: 0.002, "F4")
    let image = try Micrograph(width: p.columns, height: p.rows, pixels: one.map { Float($0) })
    let filtered = try FFTFilter.filter(image, storedAs: UInt8.self, domain: domain)
    #expect(filtered.pixels == ours.map { Float($0) })
}

// MARK: - CLAHE

@Test(arguments: ["clahe-kikuchipy-h5ebsd", "clahe-nickel-ebsd-large"])
func adaptiveHistogramEqualizationMatchesScikitImageExactly(_ name: String) throws {
    let bundle = try KikuchipyBundle(name)
    let p = try bundle.patterns()
    let pixels = p.rows * p.columns
    let runs: [(name: String, kernel: (rows: Int, columns: Int)?, clip: Double, bins: Int)] = [
        ("default", nil, 0, 128), ("k10x12_clip0p01_nbins64", (10, 12), 0.01, 64), ("k7_clip0p05_nbins256", (7, 7), 0.05, 256),
    ]
    for run in runs {
        var ours: [Double] = []
        for i in 0..<p.count {
            let one = Array(p.data[(i * pixels)..<((i + 1) * pixels)])
            ours += try AdaptiveHistogramEqualization.equalize(one, rows: p.rows, columns: p.columns, kernel: run.kernel,
                                                               clipLimit: run.clip, bins: run.bins).map { Double($0) }
        }
        let reference = try bundle.array(run.name)
        #expect(reference.shape == p.shape)
        expectClose(ours, reference, "\(name) \(run.name)")
    }
}
