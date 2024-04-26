# Updating CapyMOA's `moa.jar` version
1. Update `invoke.yml`
2. Remove the old `moa.jar` with `python -m invoke build.clean-moa`.
3. Attempt to download the new `moa.jar` with `python -m invoke build.download-moa`.
3. Update `tests/test_moajar.py` with the updated sha256 hash.
