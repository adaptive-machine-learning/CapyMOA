# Updating CapyMOA's `moa.jar` version

This document describes how to change the version of MOA that the CapyMOA
project uses. **It is only intended for developers who are contributing to
CapyMOA**. Before you start, make sure you have the following:

* You have installed the development dependencies and have an editable install
  of CapyMOA. If you have not, follow the instructions in the [installation guide](../installation.rst).

* You must **NOT** have set the environment variables that would
  override the default `moa.jar` location. `CAPYMOA_MOA_JAR` must be unset.

## Refreshing the `moa.jar`

When a developer wants to replace the `moa.jar` in their local CapyMOA as a
consequence of pulling or rebasing changes, they can run the following command:

```console
python -m invoke refresh-moa
```

## Changing Project MOA Version

When a developer needs to update the version of MOA that the capymoa project uses
they need to follow these steps:

1. **Upload the new version of MOA to the CapyMOA Dropbox.**

   Please name the new version with the date, e.g. `240412_moa.jar`
   (`yymmdd_moa.jar`) so we may rollback easily if needed.
2. **Update `invoke.yml`'s `moa_url` to point to the new version of MOA.**

   This file tells capymoa where to download the `moa.jar` from during the
   packaging process.
   * Must be the complete URL.
3. **Remove the old `moa.jar` with `python -m invoke refresh-moa`.**
4. **Update `tests/test_moajar.py` with the updated sha256 hash.**
   * macOS: ```shasum -a 256 moa_jar_file.jar```
   * linux: ```sha256sum moa_jar_file.jar```

   `tests/test_moajar.py` is used to avoid using an outdated version of MOA by
   mistake. It will generate a warning for the user if the hash of the `moa.jar`
   does not match the one in the file. The tests on Github Actions will fail
   if the hash does not match the file downloaded from the URL in `invoke.yml`.

5. **Verify that capymoa is pointing to the new version of MOA by running:**

   ```console
   $ python -c "import capymoa; capymoa.about()"
   CapyMOA 0.2.0
      CAPYMOA_DATASETS_DIR: ...
      CAPYMOA_MOA_JAR:      .../CapyMOA/src/capymoa/jar/moa.jar
      CAPYMOA_JVM_ARGS:     ['-Xmx8g', '-Xss10M']
      JAVA_HOME:            ...
      MOA version:          A SHA256 hash of the actual
      JAVA version:         ...
   ```

   In particular, check the if the hash matches the one you calculated in the
   previous step. If it does not, you should double-check the URL in
   `invoke.yml` and re-run `python -m invoke build.clean-moa` and `python -m
   invoke build.download-moa`.

6. Your pull request should include the changes to `invoke.yml`, and
   `tests/test_moajar.py`.
