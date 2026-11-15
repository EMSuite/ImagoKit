// The drift series and the aligned sum stop between frames when their task is cancelled (#80).
import Foundation
import ImagoAlignment
import ImagoCore
import ImagoKitTestSupport
import Testing

@Suite struct AlignmentCancellationTests {
    static let frames = (0..<3).map { noise(32, 32, seed: UInt64(10 + $0)) }

    @Test func theDriftSeriesStops() async {
        for reference in [DriftReference.first, .mean, .previous] {
            #expect(await throwsCancellation { try DriftSeries.estimate(Self.frames, reference: reference) })
        }
    }

    @Test func theAlignedSumStops() async {
        let shifts = [PixelShift](repeating: .zero, count: Self.frames.count)
        for method in [AlignmentMethod.fourier, .bilinear] {
            #expect(await throwsCancellation { try AlignedSum.sum(Self.frames, shifts: shifts, method: method) })
        }
    }
}
