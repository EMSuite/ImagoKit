// ImagoKit#101 (macEM#686): the live heap a spectrum adds at its peak. An image handed over to `PowerSpectrum.compute` is freed
// after the transform's row step, and `SpectrumDisplay.make` turns a handed-over spectrum's power into its values in place.
// A sampler reads the heap's bytes in use while the call runs; the least peak of five runs is kept. Other suites run in this
// process at the same time and their allocations raise a run's peak (CI read 1.51 where the arrays held come to 1⅛), so the
// bound sits below the old rise with room above the new one: other suites can only raise an old run's reading above 2.
import Darwin
import Foundation
import ImagoCore
import ImagoFFT
import Synchronization
import Testing

/// The live heap (malloc's bytes in use over every zone).
private func heapInUse() -> Int {
    var stats = malloc_statistics_t()
    malloc_zone_statistics(nil, &stats)
    return Int(stats.size_in_use)
}

/// The highest live heap a sampling thread has seen, and whether it should go on.
private final class HeapSampler: Sendable {
    let peak: Atomic<Int>
    let running = Atomic<Bool>(true)
    let done = DispatchSemaphore(value: 0)
    init(_ start: Int) { peak = Atomic(start) }
}

/// The most the live heap rose above its level at the start while `body` ran, sampled every 100 µs on another thread.
private func peakRise<T>(_ body: () throws -> T) rethrows -> (rise: Int, result: T) {
    let start = heapInUse()
    let sampler = HeapSampler(start)
    Thread.detachNewThread {
        while sampler.running.load(ordering: .acquiring) {
            let now = heapInUse()
            if now > sampler.peak.load(ordering: .relaxed) { sampler.peak.store(now, ordering: .relaxed) }
            usleep(100)
        }
        sampler.done.signal()
    }
    let result = try body()
    let end = heapInUse()
    sampler.running.store(false, ordering: .releasing)
    sampler.done.wait()
    return (max(sampler.peak.load(ordering: .relaxed), end) - start, result)
}

@Suite(.serialized) struct SpectrumPeakMemoryTests {
    // 6144 × 4096: one full-size Float32 array is 96 MiB, far above what the rest of the suite allocates at once.
    static let w = 6144, h = 4096
    static var fullSize: Int { w * h * MemoryLayout<Float>.stride }

    private static func image() throws -> Micrograph {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        let pixels = [Float](unsafeUninitializedCapacity: w * h) { p, n in
            for i in 0..<(w * h) {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                p[i] = 100 + Float(state >> 40) / Float(1 << 24)
            }
            n = w * h
        }
        return try Micrograph(width: w, height: h, pixels: pixels)
    }

    /// Red before: the image stayed until the call returned, beside the half spectrum and a full-size row scratch, then the
    /// column scratch — two full-size arrays above the input throughout. Now the row step adds the half spectrum and an
    /// eighth, and the input goes before the column step: about 1⅛ (the bound is 1.75).
    @Test func aHandedOverImageGoesAfterTheRowStep() throws {
        var rises: [Int] = []
        for _ in 0..<5 {
            var image: Micrograph? = try Self.image()
            let (rise, spectrum) = try peakRise { try PowerSpectrum.compute(image.take()!) }
            #expect(image == nil && spectrum.width == Self.w)
            rises.append(rise)
        }
        let least = Double(rises.min()!) / Double(Self.fullSize)
        #expect(least < 1.75, "the spectrum rose \(least) full-size arrays above the handed-over image (was 2)")
    }

    /// Red before: the display's values were a new array beside the power (one full-size array more). Now they are the power's
    /// own storage.
    @Test func aHandedOverSpectrumBecomesItsDisplay() throws {
        var rises: [Int] = []
        for _ in 0..<5 {
            var spectrum: PowerSpectrum? = try PowerSpectrum.compute(Self.image())
            let (rise, display) = peakRise { SpectrumDisplay.make(spectrum.take()!) }
            #expect(spectrum == nil && display.values.count == Self.w * Self.h)
            rises.append(rise)
        }
        let least = Double(rises.min()!) / Double(Self.fullSize)
        #expect(least < 0.5, "the display rose \(least) full-size arrays above the handed-over spectrum (was 1)")
    }
}
