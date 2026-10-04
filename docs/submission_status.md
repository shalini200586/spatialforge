# Submission status

Honest audit of what is ready, what is implemented but not validated, and what needs data from outside this repository. Per-requirement detail: [compliance_matrix.md](compliance_matrix.md).

## READY

- **LiDAR pipeline execution** on three real captures (PARTIAL results by design, deterministic: two full runs on `single_room` gave identical geometry and a byte-identical plan).
- **Canonical `property.json` schema 1.0** (validated by tests, committed schema file kept in sync) and `run_report.json` with 12 standard stages and timings, identical across tiers.
- **Product plan** (`plan.png`) with dimensions, openings, partial-result badge.
- **One-command CLI**: `process --tier lidar|video|photos|auto`, status model (success / partial / failure), structured warnings and errors, overwrite protection, staged output, `--version`, `doctor`.
- **Drift ablation** with safe fallback, reported honestly ([drift_ablation.md](drift_ablation.md)): accepted on one capture, rejected on two.
- **Damage pipeline** (crack and stain candidates, surface mapping, metric extents, concealed-risk rules `CDR-001..006`, scope line items).
- **Photo and video software paths** (covered by synthetic and mocked tests; video also run once on real footage, see below).
- **Benchmark harness**: ground-truth schema and example, metrics, gates as configuration, repeatability, incumbent comparison, fix-loop freeze/compare, tested.
- **Reproducibility tooling**: `scripts/setup.ps1`, `smoke_test.ps1`, `reproduce.ps1`, `build_submission.py`, `doctor`, model download script (prints repository and commit), `THIRD_PARTY.md`.
- **Documentation**: README, capture protocol, device matrix, compliance matrix, technical report, benchmark report template, demo checklist.
- **Incremental Git history**: one commit per ticket.

## PARTIAL / NOT VALIDATED

- Real Tier-1 **photos**: never run on a real photo set.
- A real ordinary **walkthrough video**: never run (the one real video was a LiDAR scanning pass: PARTIAL, weak tracking).
- Actual **tape/laser accuracy** of every tier.
- **Opening** detection recall and the **2 cm** opening-width gate (the sample opening interval is 0.75-1.55 m).
- **Ceiling** height **1.5 cm** gate.
- **Repeatability** (ceiling spread 1 cm, wall 1 cm or 0.5%).
- **Photo wall length +/-8%**, **photo footprint +/-8%**, photo whole-property stitching on real rooms.
- **Video wall length +/-3%**.
- **Confidence-interval calibration**; the interval on every measurement (top-level LiDAR wall lengths carry none).
- **Damage** precision and recall; concealed-risk rules on real damage.
- **Capture protocol** followed by a non-engineer.
- Technical report PDF: see below.

## MISSING EXTERNAL DATA

- Tape/laser **ground truth** for any property.
- **Incumbent** (consumer scanning app) export for two rooms.
- A **same-property, all-three-tier** benchmark (3+ rooms plus a connector).
- A **furnished room with real damage**.
- A **repeat capture**.
- Therefore also: the real **before/after fix** (STATUS: PENDING REAL BENCHMARK) and the **raw benchmark data** deliverable.

## Other open items

- **No LICENSE file** has been chosen for SpatialForge (owner decision). Third-party licenses are listed in `THIRD_PARTY.md` as to be verified; the depth model's license is explicitly unverified.
- The commit history's author e-mail is the git config's; nothing in tracked files contains a personal path, e-mail or credential (audited with `git grep`).
