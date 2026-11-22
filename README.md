# PressLoupe

PressLoupe is a local workbench for magnified ink-on-paper press proofs. Register repeat photographs to the same sheet, inspect the ink and paper, then retain measurements with their source images.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19978`. Set `PRESSLOUPE_PORT` to choose a different local port. The named volume retains work between container starts. With Python 3.11 and the pinned packages installed, the native entry point is `./start.sh`. An existing image with the identical locked Python packages may be used for the build's dependency stage through `PRESSLOUPE_DEPS_IMAGE`.

## Inspect a proof

1. Load the four synthetic press proofs or import a photograph. The training sheets contain screened tint steps, registration crosses, a solid ink patch and a paper reference. Each has diffuse, polarized and raking views. Under Captures, mark four corners, set the sheet dimensions and register any repeat views to that plane.
2. Use Sampling regions to outline the print and unprinted paper. Under Measure, select **Halftone dot census**. Choose a paper-background radius and a contrast threshold. The census fits one screen-lattice placement to the photographed dot positions (the same placement a pinhole cross and equal-pitch transparent grid would give), then assigns every detected ink pixel to exactly one lattice footprint. Teal outlines enclose counted footprints and amber pixels mark unresolved ink. When two lattice placements explain the pinhole positions almost equally, the lower-residual placement is used and the ambiguity is recorded. Paper white is never added to a footprint and one ink pixel can never be counted in two dots. The table records area, position, elongation, orientation, lattice site and attribution origin. Physical areas require calibrated dimensions.
3. After a magnified re-check, open an unresolved ink group in the saved census and re-read it as one merged dot or split it across its real lattice landings (squeezed pairs and triplets only). The saved pixel evidence is re-partitioned, so on reopen the dot count, every footprint position and area, the overlay and the exported CSV all match the new attribution; reverting to automatic restores the original reading. Boundary-crossing ink and groups spanning too many lattice sites (solid patches, rules, smears) cannot be split.
4. Compare ink color, estimated coverage, small appearance defects, glare, directional relief, texture or hairline networks. Measure a control-strip cross section, print rule or path and inspect the source and measured support. Keep an approved color reference or compare saved results between press pulls.
5. Evidence exports a dated report or ZIP containing saved measurements and optional original photographs. The received-evidence viewer validates the declared members and opens a local copy.

Inputs include single-frame PNG, JPEG, WebP and TIFF up to 24 MiB and 12 megapixels. A proof holds up to ten captures and 24 sampling regions. The registered plane spans at most 1,800 pixels on its longest edge. The dot census reports footprint area in the stored registered image and the first 600 dot rows. Joined dots, marks that cross a missing boundary, solid patches and very weak dots require inspection of the original. It measures visible ink appearance rather than the original printing plate.

The local StampDesk workbench supplied the image registration, evidence archive and ordinary optical inspection libraries. PressLoupe adds a screened press-sheet generator, halftone ink-component measurement and its native table, CSV, overlay and summary.
