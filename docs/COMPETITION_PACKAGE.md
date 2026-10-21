# Competition package - 21 October 2026

Assignment Version 1.5 (15 October), pages 1-2, requires one source/artifact ZIP
with the starter repository structure and no data files or notebooks. Page 9
names the seven mandatory submission paths. The competition closes on
22 October 2026 at 11:59 PM; the separate Moodle report and statements are due
24 October at 11:59 PM. The conformance freeze was 20 October at 11:59 PM.
Local verification does not establish the status of remote CI at that freeze.

`2023CS10076.zip` at the repository root is the primary competition archive.
Run `python scripts/verify_competition.py` to build and verify it. This retains
the existing builder and its report/artifact hash guards. The small existing
`docs/REPORT.pdf` is retained as an input to those guards; its inclusion does
not replace the separate Moodle submission. No scoring or training policy is
changed by packaging. Prior root ZIP bytes are backed up locally under
`runs/competition_package/`, outside the upload.

All starter source/configuration/documentation paths from the original upload
commit `6b6a70b` are retained. Raw toy/transfer data, like all other datasets,
are excluded; their directory paths are retained as empty ZIP entries.
`data/README.md` is retained. Tests need local fixture files, so extraction
verification copies only the supplied toy/transfer fixtures into a disposable
extracted project after confirming the archive itself contains no data.
Generated logs, caches, research outputs, old ZIPs, neural training histories,
credentials and intermediate models are excluded. Small provenance and
experiment summaries remain in `docs/`; historical facts are not removed.

The extracted-project verification loads the frozen main and both neural
artifacts, checks deterministic finite candidate-restricted rankings and
fusion, and trains/loads C1 models on all three supplied synthetic benchmarks.
The complete toy smoke workflow runs only in that disposable project; its
toy-trained main artifact never replaces the repository's production model.
See the local `runs/competition_package/verification.json` and logs for actual
outcomes. No portal action is performed.

The earlier fresh main-model refit and its byte equality are summarized in
`REPRODUCTION_VERIFICATION.json`. Packaging checks that saved result against
the current artifact and exercises the reproduction CLI in dry-run mode. It
does not rerun the expensive main-model fit or claim a new independent fit.
The verified environment was Windows/Python 3.13.7/LightGBM 4.7.0; execution
on staff Ubuntu 24.04.3/Python 3.12.3 is not independently verified here.

Historical TEST/public-data exposure and the unverified permission claim remain
in `COMPLIANCE.md`. Local test success is not a certification of historical
academic-integrity compliance. Remote freeze-time CI status is not verified.
