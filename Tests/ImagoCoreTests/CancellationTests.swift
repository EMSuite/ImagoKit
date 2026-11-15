// The long operations of ImagoCore stop when their task is cancelled (#80): a cancelled call throws `CancellationError` instead of
// running to the end. Each image is large enough to reach the check (a fill and a flood check every 64 Ki steps).
import Foundation
import ImagoCore
import ImagoKitTestSupport
import Testing

@Suite struct CancellationTests {
    static let side = 300

    @Test func theRollingBallStops() async {
        let image = noise(64, 48, seed: 1).pixels
        #expect(await throwsCancellation { try RollingBall.background(image: image, rows: 48, columns: 64, radius: 5) })
    }

    @Test func morphologyUntilStableStops() async {
        let mask = (0..<(64 * 64)).map { $0 % 7 != 0 }
        #expect(await throwsCancellation { try BinaryMorphology.erosion(mask, width: 64, height: 64, iterations: 0) })
        #expect(await throwsCancellation { try BinaryMorphology.dilation(mask, width: 64, height: 64, iterations: 3) })
    }

    @Test func labellingAndTheWatershedStop() async {
        let n = Self.side * Self.side
        let mask = [Bool](repeating: true, count: n)
        #expect(await throwsCancellation { try Segmentation.label(mask, width: Self.side, height: Self.side) })
        let image = noise(Self.side, Self.side, seed: 2).pixels.map(Double.init)
        var seeds = [Int32](repeating: 0, count: n)
        seeds[0] = 1; seeds[n - 1] = 2
        let markers = seeds
        #expect(await throwsCancellation {
            try Segmentation.watershed(image: image, markers: markers, rows: Self.side, columns: Self.side)
        })
        #expect(await throwsCancellation {
            try Segmentation.watershed(image: image, markers: markers, rows: Self.side, columns: Self.side, watershedLine: true)
        })
    }

    @Test func theDistanceTransformsStop() async {
        let mask = (0..<(64 * 64)).map { $0 % 11 != 0 }
        #expect(await throwsCancellation { try DistanceTransform.euclidean(foreground: mask, rows: 64, columns: 64) })
        #expect(await throwsCancellation { try DistanceTransform.euclideanWithIndices(foreground: mask, rows: 64, columns: 64) })
    }

    /// A call from a task that is not cancelled runs as before.
    @Test func anUncancelledCallRuns() async throws {
        let mask = (0..<(64 * 64)).map { $0 % 7 != 0 }
        let task = Task { try BinaryMorphology.erosion(mask, width: 64, height: 64, iterations: 2) }
        #expect(try await task.value.count == mask.count)
    }
}
