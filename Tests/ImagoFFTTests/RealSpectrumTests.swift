// PowerSpectrum's real-input path (ImagoKit#94) against the complex transform it replaced, on odd, even, radix and Bluestein
// sizes, and the Hermitian mirror that makes it hold half the memory.
import Foundation
import ImagoCore
import ImagoFFT
import Testing

/// Deterministic pixels with structure on both axes and an offset (a DC bin far above the rest, like a micrograph).
private func pixels(_ w: Int, _ h: Int) -> [Float] {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(w * 131 + h)
    return (0..<(w * h)).map { i in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let noise = Double(state >> 40) / Double(1 << 24)
        let x = Double(i % w), y = Double(i / w)
        return Float(100 + 20 * cos(2 * .pi * 5 * x / Double(w)) + 10 * sin(2 * .pi * 3 * y / Double(h)) + noise)
    }
}

/// The complex transform's centred |F|² (the 0.10.0 computation), in Double.
private func complexPower(_ p: [Float], _ w: Int, _ h: Int, _ precision: Precision) -> [Double] {
    var re: [Double], im: [Double]
    switch precision {
    case .float32:
        let fft = FFT2D(nx: w, ny: h)!
        var r = p, i = [Float](repeating: 0, count: w * h)
        fft.transform(re: &r, im: &i, forward: true)
        (re, im) = (r.map(Double.init), i.map(Double.init))
    case .float64:
        let fft = FFT2DDouble(nx: w, ny: h)!
        re = p.map(Double.init)
        im = [Double](repeating: 0, count: w * h)
        fft.transform(re: &re, im: &im, forward: true)
    }
    var out = [Double](repeating: 0, count: w * h)
    for r in 0..<h {
        let sr = (r + h - h / 2) % h
        for c in 0..<w {
            let i = sr * w + (c + w - w / 2) % w
            out[r * w + c] = re[i] * re[i] + im[i] * im[i]
        }
    }
    return out
}

// 1 × n and n × 1, odd and even, vDSP (16, 48), exact radix (45, 75) and Bluestein (127, 131) lengths.
private let sizes: [(Int, Int)] = [(1, 7), (6, 1), (16, 16), (17, 16), (16, 17), (45, 48), (127, 75), (131, 127)]

@Test(arguments: sizes, Precision.allCases)
func theRealPathIsTheComplexTransform(_ size: (Int, Int), _ precision: Precision) throws {
    let (w, h) = size
    let p = pixels(w, h)
    let s = try PowerSpectrum.compute(Micrograph(width: w, height: h, pixels: p), precision: precision)
    let want = complexPower(p, w, h, precision)
    let peak = want.max()!
    let worst = zip(s.power.doubles, want).map { abs($0 - $1) }.max()!
    // The upstream float32 band (1e-5 of the maximum) and Float64's 1e-12, the pins' bands in SpectrumTests.
    let band = precision == .float32 ? 1e-5 : 1e-12
    #expect(worst <= band * peak, "\(w) × \(h): \(worst / peak) of the maximum, band \(band)")
}

@Test(arguments: sizes, Precision.allCases)
func theSpectrumIsExactlyCentrosymmetric(_ size: (Int, Int), _ precision: Precision) throws {
    // A real image's |F|² at (r, c) equals that at (−r, −c) about the zero frequency; the mirror makes it exact, bit for bit.
    let (w, h) = size
    let s = try PowerSpectrum.compute(Micrograph(width: w, height: h, pixels: pixels(w, h)), precision: precision)
    let cx = w / 2, cy = h / 2
    var mismatches = 0
    for r in 0..<h {
        for c in 0..<w {
            let mr = 2 * cy - r, mc = 2 * cx - c
            guard (0..<h).contains(mr), (0..<w).contains(mc) else { continue }
            if s.power[r * w + c] != s.power[mr * w + mc] { mismatches += 1 }
        }
    }
    #expect(mismatches == 0, "\(w) × \(h): \(mismatches) bins differ from their mirror")
}

@Test(arguments: Precision.allCases)
func magnitudeIsTheRootOfPower(_ precision: Precision) throws {
    let s = try PowerSpectrum.compute(Micrograph(width: 45, height: 17, pixels: pixels(45, 17)), precision: precision)
    switch (s.magnitude, s.power) {
    case (.float32(let m), .float32(let p)): #expect(m == p.map { $0.squareRoot() })
    case (.float64(let m), .float64(let p)): #expect(m == p.map { $0.squareRoot() })
    default: Issue.record("magnitude and power in different precisions")
    }
}

@Test(arguments: [(16, 8), (17, 9), (256, 64)], Precision.allCases)
func identicalRowsHaveNoPowerOffTheCentreRow(_ size: (Int, Int), _ precision: Precision) throws {
    // Every row the same: the transform is zero off the centre row. The real-input split leaves rounding residue there
    // (3e-11 against a DC of 1.6e8 at 16 × 8, macEM#685); below u²·max |F|² it is not a resolved value and reads 0, as the
    // complex transform gave, so the display draws it as no power. Red before: residue.
    let (w, h) = size
    let pixels = (0..<(w * h)).map { Float(100 + 10 * cos(2 * Double.pi * 3 * Double($0 % w) / Double(w))) }
    let s = try PowerSpectrum.compute(Micrograph(width: w, height: h, pixels: pixels), precision: precision)
    let off = (0..<h).filter { $0 != h / 2 }.flatMap { r in (0..<w).map { r * w + $0 } }
    #expect(off.allSatisfy { s.power[$0] == 0 }, "\(w) × \(h): \(off.map { s.power[$0] }.max()!) off the centre row")
    #expect(s.power[(h / 2) * w + w / 2] > 0, "the DC has power")
    #expect(s.power[(h / 2) * w + w / 2 + 3] > 0, "the planted cosine has power")
}
