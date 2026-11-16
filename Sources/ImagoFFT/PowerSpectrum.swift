// SPDX-License-Identifier: GPL-3.0-only
//
//  PowerSpectrum.swift (ImagoFFT)
//  The centred Fourier spectrum of an image or of a region of it — magnitude and power, with the frequency axes in the
//  image's reciprocal unit.
//
//  CONVENTIONS, NumPy's: `F = fftshift(fft2(x))`; the zero frequency at index n/2 (integer division) on each axis, for odd
//  and even n alike; the axis value at index i is (i − n/2) · (1 / (n · d)), d each axis's own scale, so a non-square or
//  anisotropic image has different spacings along x and y. Uncalibrated, d = 1 and the unit is "1/px". Nothing is
//  windowed, padded or mean-removed.
//
//  MEMORY (ImagoKit#94): the image is real, so the transform is the real-input one (NumPy's rfft2, `RealFFT2D`): the half
//  spectrum, squared, mirrored into the centred full power by Hermitian symmetry (bin (r, c) is the conjugate of
//  bin (−r, −c)). `magnitude` is computed from `power` when read. Two full-size arrays at the peak, one held; the complex
//  transform held four at its peak (the 375 MP alite SEM image, macEM#579). The image is consumed (ImagoKit#101): handed over,
//  its pixels go after the transform's row step: the peak is 2⅛ full-size arrays (the input, the half spectrum and an eighth of
//  row scratch), then two, where it was three throughout (the input beside two; macEM#686).
//

import Foundation
import ImagoCore

/// An array a product holds in the precision it was computed in.
public nonisolated enum SpectrumValues: Sendable, Equatable {
    case float32([Float])
    case float64([Double])

    public var count: Int {
        switch self {
        case .float32(let v): v.count
        case .float64(let v): v.count
        }
    }

    /// Element `i` as a Double (exact for either case).
    public subscript(i: Int) -> Double {
        switch self {
        case .float32(let v): Double(v[i])
        case .float64(let v): v[i]
        }
    }

    /// Every element as a Double.
    public var doubles: [Double] {
        switch self {
        case .float32(let v): v.map(Double.init)
        case .float64(let v): v
        }
    }

    /// The precision the values were computed in.
    public var precision: Precision {
        switch self {
        case .float32: .float32
        case .float64: .float64
        }
    }
}

/// Why a spectrum cannot be computed.
public nonisolated enum PowerSpectrumError: Error, Equatable, Sendable {
    case crop(PixelCrop)
    case fftSetup(width: Int, height: Int)
}

public nonisolated struct PowerSpectrum: Sendable {
    public let width: Int
    public let height: Int
    /// |F|², centred, row-major, x fastest; the zero frequency at (height/2, width/2).
    public let power: SpectrumValues
    /// |F| = √|F|², same layout; computed from `power` on each read (a full-size array: read it once).
    public var magnitude: SpectrumValues {
        switch power {
        case .float32(let p): .float32(p.map { $0.squareRoot() })
        case .float64(let p): .float64(p.map { $0.squareRoot() })
        }
    }
    /// The frequency at each column and row, in `unitsX` / `unitsY`.
    public let axisX: [Double]
    public let axisY: [Double]
    /// The frequency step along each axis, 1/(n·d) (the value NumPy's fftfreq multiplies by).
    public let spacingX: Double
    public let spacingY: Double
    /// "1/<unit>" of the image's axes ("nm" for an axis in "1/nm"), or "1/px" when an axis is uncalibrated.
    public let unitsX: String
    public let unitsY: String
    /// The region transformed, nil for the whole image. Its own size sets the frequency spacing.
    public let crop: PixelCrop?
    public let precision: Precision

    /// The spectrum of `image`, or of `crop` of it. `image` is consumed: a caller that hands over its only reference (a
    /// temporary, or `consume`) has its pixels freed as soon as the transform has read them, before the column step and the
    /// power exist (ImagoKit#101); a caller that keeps the image keeps it, as before.
    public static func compute(_ image: consuming Micrograph, crop: PixelCrop? = nil,
                               precision: Precision = .float32) throws -> PowerSpectrum {
        let source: Micrograph
        if let crop {
            guard let c = image.cropped(crop) else { throw PowerSpectrumError.crop(crop) }
            _ = consume image                   // the whole image goes before the region is transformed
            source = c
        } else {
            source = consume image
        }
        let w = source.width, h = source.height
        let (dx, ux) = LengthUnit.frequencySampling(source.x)
        let (dy, uy) = LengthUnit.frequencySampling(source.y)
        let pixels = source.pixels
        _ = consume source                      // `pixels` is now the only reference this function holds
        let power: SpectrumValues
        switch precision {
        case .float32:
            guard let fft = FFT2D(nx: w, ny: h) else { throw PowerSpectrumError.fftSetup(width: w, height: h) }
            power = .float32(centredPower(fft, consume pixels, w: w, h: h))
        case .float64:
            guard let fft = FFT2DDouble(nx: w, ny: h) else { throw PowerSpectrumError.fftSetup(width: w, height: h) }
            let doubles = pixels.map(Double.init)
            _ = consume pixels
            power = .float64(centredPower(fft, consume doubles, w: w, h: h))
        }
        return PowerSpectrum(width: w, height: h, power: power,
                             axisX: frequencies(w, d: dx), axisY: frequencies(h, d: dy),
                             spacingX: 1.0 / (Double(w) * dx), spacingY: 1.0 / (Double(h) * dy),
                             unitsX: ux, unitsY: uy, crop: crop, precision: precision)
    }

    /// NumPy's `fftshift(fftfreq(n, d))`: the integer index times 1/(n·d), in that order of operations.
    public static func frequencies(_ n: Int, d: Double) -> [Double] {
        let val = 1.0 / (Double(n) * d)
        return (0..<n).map { Double($0 - n / 2) * val }
    }

    /// The centred |F|² of the real `h × w` `pixels`: the half spectrum (`h × (w/2 + 1)`), squared, then fftshift with
    /// the missing half mirrored. Output (r, c) takes bin (sr, sc) = ((r + h − h/2) mod h, (c + w − w/2) mod w), which for
    /// sc > w/2 is the conjugate of the half spectrum's bin ((h − sr) mod h, w − sc): the same power. The self-mirrored
    /// columns take one half of their rows from the other, so the result is exactly centrosymmetric.
    static func centredPower<P: LineTransformPlan>(_ plan: P, _ pixels: consuming [P.Scalar], w: Int, h: Int) -> [P.Scalar] {
        let m = w / 2 + 1
        // |F|² of the half spectrum (half a full-size array beside the transform's one, which goes at the end of this block).
        // `pixels` is handed on: the transform frees it after its row step.
        let half: [P.Scalar]
        do {
            let (re, im) = plan.realForward(consume pixels)
            half = re.withUnsafeBufferPointer { r in im.withUnsafeBufferPointer { i in
                [P.Scalar](unsafeUninitializedCapacity: r.count) { out, n in
                    for k in 0..<r.count { out[k] = r[k] * r[k] + i[k] * i[k] }
                    n = r.count
                }
            } }
        }
        var power = [P.Scalar](repeating: 0, count: w * h)
        let shiftX = w - w / 2, shiftY = h - h / 2
        half.withUnsafeBufferPointer { hb in
            power.withUnsafeMutableBufferPointer { pb in
                let src = hb.baseAddress!, dst = pb.baseAddress!
                FFT2D.forEachBand(count: h, length: w, scratchCount: 0) { rows, _ in
                    for r in rows {
                        let sr = (r + shiftY) % h, mirror = (h - sr) % h
                        let row = src + sr * m, mirrored = src + mirror * m, out = dst + r * w
                        for c in 0..<w {
                            // Columns 0 and (even w) w/2 are their own mirror: the lower rows read the upper rows' bins
                            // too, so every bin equals its mirror bit for bit.
                            let sc = (c + shiftX) % w, msc = (w - sc) % w
                            out[c] = sc < m && !(sc == msc && sr > mirror) ? row[sc] : mirrored[msc]
                        }
                    }
                }
            }
        }
        return power
    }
}
