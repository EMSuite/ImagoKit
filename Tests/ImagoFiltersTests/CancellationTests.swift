// The long operations of ImagoFilters stop when their task is cancelled (#80): a cancelled call throws `CancellationError` instead of
// running every pass.
import Foundation
import ImagoCore
import ImagoFilters
import ImagoKitTestSupport
import Testing

@Suite struct FilterCancellationTests {
    @Test func theDenoisersStop() async {
        let image = noise(32, 24, seed: 3).pixels.map { Double($0) / 100 }
        #expect(await throwsCancellation { try Denoise.totalVariationChambolle(image, rows: 24, columns: 32) })
        #expect(await throwsCancellation { try Denoise.totalVariationBregman(image, rows: 24, columns: 32) })
        for fast in [true, false] {
            #expect(await throwsCancellation {
                try Denoise.nonLocalMeans(image, rows: 24, columns: 32, patchSize: 3, patchDistance: 2, fastMode: fast)
            })
        }
    }

    @Test func theHRTEMFilterStops() async {
        let image = noise(64, 64, seed: 4)
        #expect(await throwsCancellation { try HRTEMFilter.filter(pixels: image.pixels, width: 64, height: 64) })
    }

    @Test func theGeometricPhaseAnalysisStops() async {
        let lattice = PlantedLattice("uniform", n: 64)
        let image = lattice.image, g1 = lattice.g1, g2 = lattice.g2
        #expect(await throwsCancellation { try GeometricPhaseAnalysis.analyze(image, rows: 64, columns: 64, g1: g1, g2: g2) })
    }
}
