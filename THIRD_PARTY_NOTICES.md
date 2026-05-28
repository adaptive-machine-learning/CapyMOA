# Third-Party Notices

OpenMOA is licensed under BSD-3-Clause. The following third-party components are included in, derived from, or required by the source package.

## JPype pickle workaround

- File: `src/openmoa/_pickle.py`
- Source: JPype `jpype/pickle.py`, commit `653ccffd1df46e4d472217d77f592326ae3d3690`
- License: Apache-2.0
- Notice: the original Apache-2.0 header is retained in the file; the Apache license text is included under `LICENSES/moa-jar/Apache-2.0.txt`.

## Bundled MOA runtime jar

- File: `src/openmoa/jar/moa.jar`
- Configured source URL: see `invoke.yml` (`moa_url`)
- SHA256: `b14be3c1df87aa5bf37f24c9a35258ab1f9a941897e61294701c43c0141dc2b7`
- Purpose: Java runtime backend for MOA-based learners and generated Java stubs.
- License status: the jar bundles MOA/WEKA and multiple Java dependencies. Maven metadata inside the jar reports GPL-3.0, LGPL, Apache-2.0, BSD-style, CUP, RngPack, Javolution, ThoughtWorks, and FPL/optimization license terms among bundled components. Extracted license and notice files are included under `LICENSES/moa-jar/`.

## Generated Java stubs

- Paths: `src/moa-stubs`, `src/com-stubs`
- Source: generated from `src/openmoa/jar/moa.jar` using `stubgenj`
- Purpose: Python type checking and IDE completion for Java classes used through JPype.

## Datasets

The repository working tree may contain benchmark datasets under `data/` and tiny fixtures under `tests/data/`. The build configuration excludes benchmark data from built distributions unless explicitly included by package metadata. Dataset source notes are in `data/README.md`; any dataset with unclear redistribution terms should remain outside release artifacts.
