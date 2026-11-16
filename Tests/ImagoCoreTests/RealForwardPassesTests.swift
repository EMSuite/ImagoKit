// ImagoKit#101: the real-input transform's row step runs in passes of row pairs, so its scratch is an eighth of the image,
// not half of it twice. Every line goes through the same code whatever the pass, so any split is the one-pass result bit
// for bit: odd and even heights (the last row alone), vDSP, exact radix and Bluestein lengths, both precisions.
import Foundation
import Testing
@testable import ImagoCore

@Suite struct RealForwardPassesTests {
    /// Deterministic pixels with structure on both axes.
    private static func pixels(_ w: Int, _ h: Int) -> [Double] {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D &+ UInt64(w * 977 + h)
        return (0..<(w * h)).map { i in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return 50 + 8 * cos(Double(i % w) * 0.7) + 3 * sin(Double(i / w) * 0.3) + Double(state >> 40) / Double(1 << 24)
        }
    }

    // Heights 1, 2, odd and even; widths radix 2 (16), exact radix (45) and Bluestein (31).
    static let sizes: [(Int, Int)] = [(16, 1), (16, 2), (16, 7), (45, 10), (31, 13), (45, 33)]

    /// Red before: `forward` had no passes (`pairsPerPass` did not exist).
    @Test(arguments: sizes) func anySplitIsTheOnePassTransformFloat(_ size: (Int, Int)) throws {
        let (w, h) = size
        let fft = try #require(FFT2D(nx: w, ny: h))
        let x = Self.pixels(w, h).map(Float.init)
        let pairs = (h + 1) / 2
        let one = RealFFT2D.forward(fft, x, pairsPerPass: pairs)
        for perPass in [1, 2, 3, max(1, pairs - 1)] {
            let split = RealFFT2D.forward(fft, x, pairsPerPass: perPass)
            #expect(split.re.map(\.bitPattern) == one.re.map(\.bitPattern), "\(w) × \(h), \(perPass) pairs a pass")
            #expect(split.im.map(\.bitPattern) == one.im.map(\.bitPattern), "\(w) × \(h), \(perPass) pairs a pass")
        }
        let defaulted = fft.realForward(x)
        #expect(defaulted.re.map(\.bitPattern) == one.re.map(\.bitPattern))
        #expect(defaulted.im.map(\.bitPattern) == one.im.map(\.bitPattern))
    }

    @Test(arguments: sizes) func anySplitIsTheOnePassTransformDouble(_ size: (Int, Int)) throws {
        let (w, h) = size
        let fft = try #require(FFT2DDouble(nx: w, ny: h))
        let x = Self.pixels(w, h)
        let pairs = (h + 1) / 2
        let one = RealFFT2D.forward(fft, x, pairsPerPass: pairs)
        for perPass in [1, 2, 3, max(1, pairs - 1)] {
            let split = RealFFT2D.forward(fft, x, pairsPerPass: perPass)
            #expect(split.re.map(\.bitPattern) == one.re.map(\.bitPattern), "\(w) × \(h), \(perPass) pairs a pass")
            #expect(split.im.map(\.bitPattern) == one.im.map(\.bitPattern), "\(w) × \(h), \(perPass) pairs a pass")
        }
    }

    /// The default pass: an eighth of the pairs, at least enough lines for the parallel bands, never more than all of them.
    @Test func theDefaultPassIsAnEighth() {
        #expect(RealFFT2D.rowPairsPerPass(pairs: 9716, columns: 19433) == 1215)
        #expect(RealFFT2D.rowPairsPerPass(pairs: 2048, columns: 4096) == 257)
        #expect(RealFFT2D.rowPairsPerPass(pairs: 8, columns: 16) == 8)
        #expect(RealFFT2D.rowPairsPerPass(pairs: 1, columns: 1) == 1)
    }
}
