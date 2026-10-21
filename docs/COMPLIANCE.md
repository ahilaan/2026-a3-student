# Training data and provenance

The active main-model campaign uses only Fold1 TRAIN and DEV. Fold1 TEST is
excluded from fitting, augmentation, preprocessing statistics, validation and
checkpoint selection. TRAIN+DEV fitting is declared; DEV refit scores are trained-on.

The previous submission #177 used TEST labels, contrary to the assignment's
rules. The earlier statement that permission existed was incorrect; TEST use
was prohibited from the beginning. Its exact ZIP, artifacts, source snapshot and hash
manifest are preserved locally under runs/compliant_campaign/archive/pre_policy_change.
That archive is excluded from submission packages and is not an input to the
new main-model training. Its history has not been rewritten.

Historical hyperparameter research included TEST access. The first TRAIN+DEV-only
refit reuses that model family but discards all old main-model weights and fits
from scratch. Subsequent local studies reserve 600 TRAIN queries before fitting
and augmentation. These are development comparisons, not a claim that past
experimentation was independent of TEST. Portal feedback also informs selection.
Refitting alone does not undo that earlier access or establish that the whole
experimental process satisfies the assignment's rules.

Pointwise and the retained pairwise artifacts were trained without TEST labels.
Their hashes and provenance are verified before packaging.

Historical C1 research under `runs/c1_research/` also downloaded and used
public LTR benchmark labels (including their evaluation splits). Its report
records a claimed TA clarification but no independently verified official
written amendment. That claim does not establish permission under the original
PDF. The final optimization pass does not open those datasets or reuse their
results for selection. Records remain preserved. The baseline transfer parser
already included a saturation change motivated by that research; this pass
does not claim the earlier process was unaffected by those labels.

Current TRAIN+DEV-only fitting, exact retained-component assembly, and a fresh
end-to-end refit are distinct claims. See `FINAL_C1_PASS.md` for the measured
verification status. A new refit cannot erase historical selection exposure.
