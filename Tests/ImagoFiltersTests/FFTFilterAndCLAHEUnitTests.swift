// The FFT filter and CLAHE on closed forms and refusals (ImagoKit#106); their kikuchipy and scikit-image parity is
// ImagoKitParityTests/Filters/FFTFilterAndCLAHETests.swift.
import Foundation
import ImagoCore
import ImagoFilters
import Testing

@Suite struct FFTFilterAndCLAHEUnitTests {
    /// A 5 x 7 ramp with a bump, so a shifted read shows.
    private static let rows = 5, columns = 7
    private static let image: [Double] = (0..<35).map { Double($0) + ($0 == 17 ? 10 : 0) }

    private func expectClose(_ got: [Double], _ want: [Double], _ label: String) {
        #expect(got.count == want.count, "\(label): count")
        let worst = zip(got, want).map { abs($0 - $1) }.max() ?? .infinity
        #expect(worst <= 1e-9, "\(label): worst \(worst)")
    }

    @Test func aTransferFunctionOfOnesKeepsTheImage() throws {
        let ones = [Double](repeating: 1, count: Self.image.count)
        for shift in [false, true] {
            expectClose(try FFTFilter.frequencyFiltered(Self.image, rows: Self.rows, columns: Self.columns, transferFunction: ones,
                                                        shift: shift), Self.image, "shift \(shift)")
        }
    }

    /// kernel[a, b] reads image[clamp(i + a − l), clamp(j + b − l)], l = 1 for a 3 x 3 kernel: the corner taps read one pixel
    /// up-left and one down-right, the edges replicated.
    @Test func aSpatialKernelReadsTheEdgeReplicatedNeighbour() throws {
        func clamped(_ dr: Int, _ dc: Int) -> [Double] {
            (0..<(Self.rows * Self.columns)).map { k -> Double in
                let r = min(max(k / Self.columns + dr, 0), Self.rows - 1), c = min(max(k % Self.columns + dc, 0), Self.columns - 1)
                return Self.image[r * Self.columns + c]
            }
        }
        var topLeft = [Double](repeating: 0, count: 9), bottomRight = topLeft
        topLeft[0] = 1
        bottomRight[8] = 1
        expectClose(try FFTFilter.spatialFiltered(Self.image, rows: Self.rows, columns: Self.columns, kernel: topLeft, kernelRows: 3,
                                                  kernelColumns: 3), clamped(-1, -1), "top-left tap")
        expectClose(try FFTFilter.spatialFiltered(Self.image, rows: Self.rows, columns: Self.columns, kernel: bottomRight,
                                                  kernelRows: 3, kernelColumns: 3), clamped(1, 1), "bottom-right tap")
        expectClose(try FFTFilter.spatialFiltered(Self.image, rows: Self.rows, columns: Self.columns, kernel: [1], kernelRows: 1,
                                                  kernelColumns: 1), Self.image, "1 x 1 kernel")
    }

    /// The second call runs on the cached plan (FFTSurface's cache) and gives the same bits as the first.
    @Test func aRepeatedFilterRepeatsItsNumbers() throws {
        let h = (0..<Self.image.count).map { 1 / (1 + Double($0 % 6)) }
        let first = try FFTFilter.frequencyFiltered(Self.image, rows: Self.rows, columns: Self.columns, transferFunction: h, shift: true)
        let again = try FFTFilter.frequencyFiltered(Self.image, rows: Self.rows, columns: Self.columns, transferFunction: h, shift: true)
        #expect(first == again)
    }

    @Test func fftFilterRefusals() throws {
        let pixels = [UInt8](repeating: 1, count: 25)
        #expect(throws: FFTFilterError.shape) {
            try FFTFilter.filter(pixels, rows: 5, columns: 5, domain: .frequency(transferFunction: [Double](repeating: 1, count: 16)))
        }
        #expect(throws: FFTFilterError.shape) {
            try FFTFilter.filter(pixels, rows: 5, columns: 5, domain: .spatial(kernel: [1, 2, 3], rows: 2, columns: 2))
        }
        #expect(throws: FFTFilterError.shape) {
            try FFTFilter.filter(pixels, rows: 5, columns: 5, domain: .spatial(kernel: [], rows: 0, columns: 3))
        }
        // A zero transfer function leaves a flat image, which has no range to rescale.
        let ramp = (0..<25).map { UInt8($0) }
        #expect(throws: FFTFilterError.nonFinite) {
            try FFTFilter.filter(ramp, rows: 5, columns: 5, domain: .frequency(transferFunction: [Double](repeating: 0, count: 25)))
        }
        let half = try Micrograph(width: 2, height: 1, pixels: [0.5, 1])
        let over = try Micrograph(width: 2, height: 1, pixels: [300, 1])
        let ones = FFTFilter.Domain.frequency(transferFunction: [1, 1])
        #expect(throws: FFTFilterError.pixelValues) { try FFTFilter.filter(half, storedAs: UInt8.self, domain: ones) }
        #expect(throws: FFTFilterError.pixelValues) { try FFTFilter.filter(over, storedAs: UInt8.self, domain: ones) }
    }

    @Test func theMicrographCallsKeepTheCalibrationAndTheNumbers() throws {
        let ramp = (0..<64).map { UInt16(($0 * 37) % 64 * 1000) }
        let axis = AxisCalibration(scale: 0.5, unit: "nm", origin: 3)
        let image = try Micrograph(width: 8, height: 8, pixels: ramp.map { Float($0) }, x: axis, y: axis)
        let domain = FFTFilter.Domain.spatial(kernel: [1, 2, 1, 2, 4, 2, 1, 2, 1], rows: 3, columns: 3)
        let filtered = try FFTFilter.filter(image, storedAs: UInt16.self, domain: domain)
        let filteredArray = try FFTFilter.filter(ramp, rows: 8, columns: 8, domain: domain)
        #expect(filtered.pixels == filteredArray.map { Float($0) })
        #expect(filtered.x == axis && filtered.y == axis)
        let equalised = try AdaptiveHistogramEqualization.equalize(image, storedAs: UInt16.self, kernel: (4, 4), clipLimit: 0.01)
        let equalisedArray = try AdaptiveHistogramEqualization.equalize(ramp, rows: 8, columns: 8, kernel: (4, 4), clipLimit: 0.01)
        #expect(equalised.pixels == equalisedArray.map { Float($0) })
        #expect(equalised.x == axis && equalised.y == axis)
    }

    @Test func claheClipHistogramRedistributesEveryExcessCount() {
        var hist = [10, 0, 0, 1, 7]
        AdaptiveHistogramEqualization.clipHistogram(&hist, limit: 4)
        #expect(hist.reduce(0, +) == 18)
        #expect(hist.allSatisfy { $0 <= 4 })
    }

    @Test func claheDefaultKernelIsKikuchipys() {
        let k = AdaptiveHistogramEqualization.defaultKernel(rows: 60, columns: 80)
        #expect(k.rows == 20 && k.columns == 15)
    }

    @Test func claheRefusals() {
        let pixels = (0..<64).map { UInt8($0 * 3) }
        #expect(throws: AdaptiveHistogramEqualizationError.shape) {
            try AdaptiveHistogramEqualization.equalize(pixels, rows: 8, columns: 7)
        }
        #expect(throws: AdaptiveHistogramEqualizationError.shape) {
            try AdaptiveHistogramEqualization.equalize(pixels, rows: 8, columns: 8, kernel: (0, 4))
        }
        #expect(throws: AdaptiveHistogramEqualizationError.parameter("clip limit")) {
            try AdaptiveHistogramEqualization.equalize(pixels, rows: 8, columns: 8, kernel: (4, 4), clipLimit: -1)
        }
        #expect(throws: AdaptiveHistogramEqualizationError.parameter("bins")) {
            try AdaptiveHistogramEqualization.equalize(pixels, rows: 8, columns: 8, kernel: (4, 4), bins: 0)
        }
        #expect(throws: AdaptiveHistogramEqualizationError.shape) {
            try AdaptiveHistogramEqualization.equalize(pixels.map { UInt32($0) }, rows: 8, columns: 8, kernel: (4, 4))
        }
    }
}
