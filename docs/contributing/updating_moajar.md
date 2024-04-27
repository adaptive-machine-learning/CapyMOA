# Updating CapyMOA's `moa.jar` version
1. Upload the new version of MOA to the CapyMOA Dropbox. Please name the new
   version with the date, e.g. `240412_moa.jar` (`yymmdd_moa.jar`) so we may 
   rollback easily if needed.
2. Update `invoke.yml`'s `moa_url` to point to the new version of MOA.
   * Must be the complete URL
3. Remove the old `moa.jar` with `python -m invoke build.clean-moa`.
4. Attempt to download the new `moa.jar` with `python -m invoke build.download-moa`.
5. Update `tests/test_moajar.py` with the updated sha256 hash.
   * macOS: ```shasum -a 256 moa_jar_file.jar```
