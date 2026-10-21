# Separate Moodle submission - due 24 October 2026, 11:59 PM

Version 1.5, page 12, Section 6 requires the following after the competition:

1. Source code and README covering fusion, pointwise/pairwise learning,
   LambdaMART and the C1 recipe. Preserve the exact competition source snapshot.
2. A 2-3-page report: `docs/REPORT.pdf` is the existing three-page report.
   Review it before Moodle upload against the frozen configuration, current
   C1 behavior and disclosures. `docs/FINAL_C1_PASS.md` supplies the later
   C1 study evidence; it is a supporting record, not the 2-3-page report.
   The report must compare all three models' DEV nDCG@10, plot at least one
   LambdaMART sweep, and analyze generalization using grouped TRAIN CV.
3. A paragraph explaining the final model and accuracy/generalization tradeoff.
   The paragraph below records the frozen configuration.
4. AI-use and code-provenance statements. The statements in README.md and
   docs/SUBMISSION_NOTES.md, together with COMPLIANCE.md, NEURAL_PROVENANCE.md
   and FUSION_PROVENANCE.md, are the current source material. The student must
   review them for completeness, including any sources consulted outside this
   session; no separate course-overview template has been verified here.

Final model paragraph: The submitted LambdaMART is a native 4,400-tree blend
of two augmented TRAIN+DEV fits, with 136 raw features and 136 within-query
average-tie percentiles. The components have 2,000 trees/127 leaves/seed 0 and
2,400 trees/63 leaves/seed 23 and weights 0.65/0.35. Both use learning rate
0.05, minimum leaf size 100, L2 10, extra randomized splits, 63 bins and 0.8
feature/row sampling. Query-relative features and augmentation target scale
robustness, while regularization and grouped TRAIN comparisons inform the
accuracy/generalization tradeoff. DEV contributes gradients to the main and
pairwise fits and selects pointwise epochs, so those DEV measurements are not
independent generalization estimates. Historical TEST-label/public-data access
and portal selection exposure remain disclosed; current TRAIN+DEV refitting
does not erase them. C1 separately fits only its supplied TRAIN/validation
inputs, choosing between raw and query-percentile rankers using validation
nDCG@10 with a raw fallback and shared soft time budget; hidden performance
is unknown. The neural configurations are recorded in CURRENT_NEURAL.json.

This task prepares the competition ZIP only and submits nothing to Moodle.
