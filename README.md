# PressLoupe

PressLoupe is a local workbench for magnified ink-on-paper press proofs. Register repeat photographs to the same sheet, inspect the ink and paper, then retain measurements with their source images.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19978`. Set `PRESSLOUPE_PORT` to choose a different local port. The named volume retains work between container starts. With Python 3.11 and the pinned packages installed, the native entry point is `./start.sh`. An existing image with the identical locked Python packages may be used for the build's dependency stage through `PRESSLOUPE_DEPS_IMAGE`.

## Inspect a proof

1. Load the four synthetic press proofs or import a photograph. The training sheets contain screened tint steps, registration crosses, a solid ink patch and a paper reference. Each has diffuse, polarized and raking views. Under Captures, mark four corners, set the sheet dimensions and register any repeat views to that plane.
2. Use Sampling regions to outline the print and unprinted paper. Under Measure, select **Halftone dot census**. Choose a paper-background radius and a contrast threshold. Teal outlines enclose complete, size-qualified ink components. Amber pixels mark unresolved ink. The table records area, position, elongation and an orientation for elongated components. Physical areas require calibrated dimensions.
3. Compare ink color, estimated coverage, small appearance defects, glare, directional relief, texture or hairline networks. Measure a control-strip cross section, print rule or path and inspect the source and measured support. Keep an approved color reference or compare saved results between press pulls.

### Dual-illuminant cut swatch

For a spot color whose physical room keeps a single continuous paper cut (for example 6 × 8 mm), photograph the signing sheet twice under the standard lamp and the workshop warm lamp, register both captures to the same plane, and keep the separately photographed target swatch for the same color number. **Plan cut swatch** scans axis-aligned cut rectangles in both orientations and ranks them by the worst of the two same-lamp ΔE76 values against the target. Every lamp fingerprint is computed only from pixels that both lamp captures see inside the same normalized rectangle, that are on paper (inside the chosen region and outside exclusions), supported, unclipped and free of a specular reflection in that lamp. Reflection, out-of-paper and pixels one lamp cannot see are dropped per lamp; they are never averaged across lamps or replaced by pixels from another position.

Approving a plan freezes the physical swatch in the Reference library with one fingerprint per illuminant, the cut rectangle geometry and the per-lamp eligible-pixel counts. A later **Reference match** against a new press sheet compares each lamp only with the same lamp on the frozen swatch. A missing lamp is reported explicitly as not captured and the other lamp is never substituted, while earlier single-capture references and measurements keep working unchanged.

4. Evidence exports a dated report or ZIP containing saved measurements and optional original photographs. The received-evidence viewer validates the declared members and opens a local copy.

Inputs include single-frame PNG, JPEG, WebP and TIFF up to 24 MiB and 12 megapixels. A proof holds up to ten captures and 24 sampling regions. The registered plane spans at most 1,800 pixels on its longest edge. The dot census reports the area in the registered image and the first 600 component rows. Joined dots, marks that cross a missing boundary and very weak dots require inspection of the original. It measures visible ink appearance rather than the original printing plate.

The local StampDesk workbench supplied the image registration, evidence archive and ordinary optical inspection libraries. PressLoupe adds a screened press-sheet generator, halftone ink-component measurement and its native table, CSV, overlay and summary.
