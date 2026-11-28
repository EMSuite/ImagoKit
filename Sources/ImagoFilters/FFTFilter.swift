// SPDX-License-Identifier: GPL-3.0-only
//
//  FFTFilter.swift (ImagoFilters)
//  An image filtered through its 2D FFT: kikuchipy 0.13.1's `EBSD.fft_filter` in both of its domains, ported step by step in
//  float64 for one image of any size (kikuchipy runs it on every pattern of a scan; nothing here is specific to a pattern).
//  - "frequency": `kikuchipy.pattern._pattern.fft_filter` — the image's 2D FFT (`scipy.fft.fft2`), multiplied by a transfer
//    function of the image's shape (with `shift`, the function is laid out for the zero frequency at the centre, upstream's
//    `fftshift` / `ifftshift` pair), the real part of the inverse FFT.
//  - "spatial": `kikuchipy.filters.fft_barnes._fft_filter` (Barnes et al. 2016) — the image padded by edge replication, the
//    kernel flipped into the corner of a zero array of the padded shape, their FFTs multiplied, the inverse cropped: the
//    correlation of the image with the kernel under nearest-edge reads. Upstream pads to SciPy's `next_fast_len`; here the
//    padded shape is exactly image + kernel − 1 per axis (the extra zeros upstream adds are never read by the crop), and the
//    FFT is the kit's exact-length `FFT2DDouble` (any length), its plans shared through `FFTSurface`'s plan cache.
//  Both rescale the filtered image to the integer type's range and cast by truncation (`kikuchipy.pattern.chunk.fft_filter`
//  → `rescale_intensity`).
//
//  Numbers: kikuchipy runs the FFTs in float32; these are float64 (the reference path). The rescaled values agree within
//  1e-5 relative or 1e-3 grey levels, the cast images within one grey level (measured on all 4125 patterns of kikuchipy's
//  nickel_ebsd_large: 0.005 % of the pixels, at most 2 of a pattern). Moved from Electron Microscopy Studio
//  (Sources/TEMSEM/Patterns/PatternFilters.swift) without a changed number (ImagoKit#106).
//
//  kikuchipy is GPL-3.0-or-later, taken under GPL-3.0 (NOTICE).
//

import Foundation
import ImagoCore

/// Why an FFT filter cannot run.
public nonisolated enum FFTFilterError: Error, Equatable, Sendable, LocalizedError {
    /// A count does not match `rows * columns`, a transfer function is not the image's shape, a kernel's count is not its
    /// `rows * columns`, or a size is below 1.
    case shape
    /// The FFT for this shape could not be set up.
    case fftSetup
    /// The filtered image is flat or not finite, so it cannot be rescaled.
    case nonFinite
    /// A `Micrograph` pixel is not a whole number in the stored integer type's range.
    case pixelValues

    public var errorDescription: String? {
        switch self {
        case .shape: "The image's values or the filter do not match the image's size."
        case .fftSetup: "The FFT for this image size could not be set up."
        case .nonFinite: "The filtered image is flat or not finite, so it cannot be rescaled."
        case .pixelValues: "The image's pixels are not whole numbers in the range of the stored type."
        }
    }
}

/// kikuchipy 0.13.1's `EBSD.fft_filter(transfer_function, function_domain, shift)` for one image, in float64.
public nonisolated enum FFTFilter {
    /// kikuchipy's `function_domain`, with the filter it takes.
    public enum Domain: Sendable, Equatable {
        /// `function_domain="frequency"`: `transferFunction` has the image's shape, row-major; with `shift`, its centre
        /// (`shape // 2`) multiplies the zero frequency.
        case frequency(transferFunction: [Double], shift: Bool = false)
        /// `function_domain="spatial"`: a `rows × columns` kernel, row-major, correlated with the image under nearest-edge reads.
        case spatial(kernel: [Double], rows: Int, columns: Int)
    }

    // MARK: - Frequency domain

    /// `kikuchipy.pattern._pattern.fft_filter(image, transfer_function, shift)`: Re(IFFT(FFT(image) · H)), H laid out as
    /// upstream lays it: with `shift`, the transfer function's centre (`shape // 2`) multiplies the zero frequency.
    public static func frequencyFiltered(_ image: [Double], rows: Int, columns: Int, transferFunction h: [Double],
                                         shift: Bool = false) throws -> [Double] {
        guard rows >= 1, columns >= 1, image.count == ImagoSize.area(rows, columns), h.count == image.count else {
            throw FFTFilterError.shape
        }
        guard let fft = PlanCache.shared.plan(nx: columns, ny: rows) as FFT2DDouble? else { throw FFTFilterError.fftSetup }
        var re = image
        var im = [Double](repeating: 0, count: image.count)
        fft.transform(re: &re, im: &im, forward: true)
        // fftshift moves frequency u to (u + n/2) mod n; the product there is undone by ifftshift, so frequency u is
        // multiplied by H[(u + n/2) mod n] (H[u] without the shift).
        let sy = shift ? rows / 2 : 0, sx = shift ? columns / 2 : 0
        for u in 0..<rows {
            let hu = (u + sy) % rows
            for v in 0..<columns {
                let g = h[hu * columns + (v + sx) % columns]
                re[u * columns + v] *= g
                im[u * columns + v] *= g
            }
        }
        fft.transform(re: &re, im: &im, forward: false, scaleInverse: true)
        return re
    }

    // MARK: - Spatial domain (Barnes)

    /// `kikuchipy.filters.fft_barnes._fft_filter(image, …)` with `_fft_filter_setup(image.shape, window)`: the image correlated
    /// with the `kernelRows × kernelColumns` kernel under nearest-edge reads, through the padded FFT. Output pixel (i, j) sums
    /// kernel[a, b] · image[clamp(i + a − ly), clamp(j + b − lx)] with l = k − (k − 1)/2 − 1 per axis (`_offset_before_fft`).
    public static func spatialFiltered(_ image: [Double], rows: Int, columns: Int, kernel: [Double], kernelRows wy: Int,
                                       kernelColumns wx: Int) throws -> [Double] {
        guard rows >= 1, columns >= 1, image.count == ImagoSize.area(rows, columns),
              wy >= 1, wx >= 1, kernel.count == ImagoSize.area(wy, wx) else { throw FFTFilterError.shape }
        let (fy, overflowY) = rows.addingReportingOverflow(wy - 1)
        let (fx, overflowX) = columns.addingReportingOverflow(wx - 1)
        guard !overflowY, !overflowX, let padded = ImagoSize.area(fy, fx) else { throw FFTFilterError.shape }
        let ay = (wy - 1) / 2, ax = (wx - 1) / 2      // `_offset_after_ifft`, and the replicated rows/columns after the image
        guard let fft = PlanCache.shared.plan(nx: fx, ny: fy) as FFT2DDouble? else { throw FFTFilterError.fftSetup }
        // `_pad_image`: the image at the origin, its last row/column replicated (k − 1)/2 times after it, its first row/column
        // in the remaining `_offset_before_fft` rows/columns at the far end (they wrap around to just before the image).
        func source(_ p: Int, _ n: Int, _ after: Int) -> Int { p < n ? p : (p < n + after ? n - 1 : 0) }
        var re = [Double](repeating: 0, count: padded)
        for p in 0..<fy {
            let r = source(p, rows, ay)
            for q in 0..<fx { re[p * fx + q] = image[r * columns + source(q, columns, ax)] }
        }
        var im = [Double](repeating: 0, count: padded)
        // `_pad_window`: the kernel flipped in both axes, in the top-left corner.
        var kre = [Double](repeating: 0, count: padded)
        for a in 0..<wy { for b in 0..<wx { kre[a * fx + b] = kernel[(wy - 1 - a) * wx + (wx - 1 - b)] } }
        var kim = [Double](repeating: 0, count: padded)
        fft.transform(re: &re, im: &im, forward: true)
        fft.transform(re: &kre, im: &kim, forward: true)
        for k in 0..<padded {
            let (a, b, c, d) = (re[k], im[k], kre[k], kim[k])
            re[k] = a * c - b * d
            im[k] = a * d + b * c
        }
        fft.transform(re: &re, im: &im, forward: false, scaleInverse: true)
        var out = [Double](repeating: 0, count: rows * columns)
        for i in 0..<rows { for j in 0..<columns { out[i * columns + j] = re[(ay + i) * fx + ax + j] } }
        return out
    }

    // MARK: - EBSD.fft_filter

    /// kikuchipy 0.13.1's `EBSD.fft_filter(transfer_function, function_domain, shift)` for one integer image, before the
    /// cast: filtered in `domain`, then rescaled from its own minimum and maximum to the type's range (0 to `P.max`).
    public static func rescaledValues<P: FixedWidthInteger & UnsignedInteger>(
        _ pixels: [P], rows: Int, columns: Int, domain: Domain
    ) throws -> [Double] {
        let p = pixels.map { Double($0) }
        let filtered: [Double]
        switch domain {
        case let .frequency(transferFunction, shift):
            filtered = try frequencyFiltered(p, rows: rows, columns: columns, transferFunction: transferFunction, shift: shift)
        case let .spatial(kernel, kernelRows, kernelColumns):
            filtered = try spatialFiltered(p, rows: rows, columns: columns, kernel: kernel, kernelRows: kernelRows,
                                           kernelColumns: kernelColumns)
        }
        // `rescale_intensity(filtered, dtype_out)`: the image's own minimum and maximum to 0 ... P.max.
        guard let imin = filtered.min(), let imax = filtered.max(), imin.isFinite, imax.isFinite else {
            throw FFTFilterError.nonFinite
        }
        let omin = 0.0, omax = Double(P.max)      // `skimage.util.dtype.dtype_range` of an unsigned type
        let scale = (omax - omin) / (imax - imin)
        guard scale.isFinite else { throw FFTFilterError.nonFinite }
        return filtered.map { ($0 - imin) * scale + omin }
    }

    /// kikuchipy 0.13.1's `EBSD.fft_filter(transfer_function, function_domain, shift)` for one integer image: `rescaledValues`
    /// cast back to the image's type by truncation, as `astype` does.
    public static func filter<P: FixedWidthInteger & UnsignedInteger>(
        _ pixels: [P], rows: Int, columns: Int, domain: Domain
    ) throws -> [P] {
        let values = try rescaledValues(pixels, rows: rows, columns: columns, domain: domain)
        let top = Double(P.max)
        // Every value is already in the type's range (the rescale put the minimum and maximum there); the clamp guards the
        // last bit of rounding.
        return try values.map { v -> P in
            guard v.isFinite else { throw FFTFilterError.nonFinite }
            return P(min(max(v, 0), top))
        }
    }

    /// kikuchipy 0.13.1's `EBSD.fft_filter(transfer_function, function_domain, shift)` of a `Micrograph` whose pixels are the
    /// stored values of an unsigned integer type `P` (an 8-bit or 16-bit pattern read into Float32, exact): the filtered
    /// image in the same type's values, its calibration kept. A pixel that is not a whole number in `P`'s range is refused.
    public static func filter<P: FixedWidthInteger & UnsignedInteger>(
        _ image: Micrograph, storedAs _: P.Type, domain: Domain
    ) throws -> Micrograph {
        guard let pixels: [P] = image.storedIntegers() else { throw FFTFilterError.pixelValues }
        let out = try filter(pixels, rows: image.height, columns: image.width, domain: domain)
        return try Micrograph(width: image.width, height: image.height, pixels: out.map { Float($0) }, x: image.x, y: image.y)
    }
}

extension Micrograph {
    /// The pixels as the unsigned integer type `P` they were stored in, or nil when one is not a whole number in its range.
    func storedIntegers<P: FixedWidthInteger & UnsignedInteger>() -> [P]? {
        var out = [P]()
        out.reserveCapacity(pixels.count)
        for v in pixels {
            guard let p = P(exactly: v) else { return nil }
            out.append(p)
        }
        return out
    }
}
