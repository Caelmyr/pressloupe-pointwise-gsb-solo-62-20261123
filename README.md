# PressLoupe

PressLoupe is a local workbench for magnified ink-on-paper press proofs. Register repeat photographs to the same sheet, inspect the ink and paper, then retain measurements with their source images.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19978`. Set `PRESSLOUPE_PORT` to choose a different local port. The named volume retains work between container starts. With Python 3.11 and the pinned packages installed, the native entry point is `./start.sh`. An existing image with the identical locked Python packages may be used for the build's dependency stage through `PRESSLOUPE_DEPS_IMAGE`.

## Inspect a proof

1. Load the four synthetic press proofs or import a photograph. The training sheets contain screened tint steps, registration crosses, a solid ink patch and a paper reference. Each has diffuse, polarized and raking views. Under Captures, mark four corners, set the sheet dimensions and register any repeat views to that plane.
2. Use Sampling regions to outline the print and unprinted paper. Under Measure, select **Halftone dot census**. Choose a paper-background radius and a contrast threshold. Teal outlines enclose complete, size-qualified ink components. Amber pixels mark unresolved ink. The table records area, position, elongation and an orientation for elongated components. Physical areas require calibrated dimensions.
3. Compare ink color, estimated coverage, small appearance defects, glare, directional relief, texture or hairline networks. Measure a control-strip cross section, print rule or path and inspect the source and measured support. Keep an approved color reference or compare saved results between press pulls.
4. Evidence exports a dated report or ZIP containing saved measurements and optional original photographs. The received-evidence viewer validates the declared members and opens a local copy.

Inputs include single-frame PNG, JPEG, WebP and TIFF up to 24 MiB and 12 megapixels. A proof holds up to ten captures and 24 sampling regions. The registered plane spans at most 1,800 pixels on its longest edge. The dot census reports the area in the registered image and the first 600 component rows. Joined dots, marks that cross a missing boundary and very weak dots require inspection of the original. It measures visible ink appearance rather than the original printing plate.

The local StampDesk workbench supplied the image registration, evidence archive and ordinary optical inspection libraries. PressLoupe adds a screened press-sheet generator, halftone ink-component measurement and its native table, CSV, overlay and summary.
