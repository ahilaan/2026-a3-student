# Rank-4 development campaign after submission 203

Historical campaign record. The 3,000-tree blend described below is not the
current 4,400-tree artifact; `CURRENT_MODEL.json` identifies the present blend.
Portal ranks/scores are observations at that time, not current standing.

Submission 203 replaced 177 as the portal's selected best, with live score
0.5116 and rank 5. Fourth place was 0.5117 at that observation. The next
Track B band begins at DEV nDCG 0.5575. C2 feedback is available only from
the portal; new local work does not open Fold1 TEST.

The next candidate averages two augmented 1,200-tree TRAIN+DEV refits with
seeds 0 and 23, then blends with the existing plain 600-tree refit. The
predeclared DEV grid uses augmented fractions .90, .85, .80, .75, .70, .65,
and .60. Select the largest fraction reaching DEV nDCG .5575. DEV supplied
gradients, so this is optimization of Track B, not unseen-query validation.
scripts/select_dev_blend.py records all results and the selected fraction.
The grid selected augmented weight .70: DEV nDCG .55795013. Weight .75
reached .55712421 and missed the target. The native final artifact therefore
weights seed-0 augmented1200, seed-23 augmented1200 and plain600 at
.35, .35 and .30 respectively. Its C2 result is unknown before submission.

A separate robustness comparison uses the two seed-0 study models that
excluded the same 600 TRAIN queries before fitting and augmentation.
scripts/robust_blend_study.py compares clean, scaled and feature-dropped
versions of those queries. The highest mean in its .60--1.00 grid is
.457513 at augmented weight .65, compared with .456915 at .80. These small
differences reuse development queries. They are a single-seed proxy, not
validation of the full-data seed ensemble or a guarantee about hidden C2.

Subsequent portal results inform submission selection. Historical TEST
exposure remains documented in docs/COMPLIANCE.md; these new comparisons
do not undo that history.
