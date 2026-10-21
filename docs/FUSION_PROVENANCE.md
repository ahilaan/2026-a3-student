# Current fusion: equal-weight CombSUM over three channels

The current rule min-max normalizes BM25 whole-document, BM25 title and
LMIR Dirichlet whole-document scores within each candidate query, then adds
the three normalized scores. LMIR JM has weight zero. Constants and ties
retain the starter semantics; no relevance labels are used during inference.

Selection used TRAIN only. A two-start, two-sweep coordinate search over
channel weights/powers found TRAIN nDCG .30062935 / MAP .12323512.
The rule was then simplified using a predeclared one-standard-error rule:
among equal-weight textbook CombSUM subsets, keep the most channels whose
TRAIN nDCG and MAP are both within one query standard error of the fine rule;
break ties by .8*nDCG+.2*MAP. Only the three-channel subset qualified.
Its TRAIN nDCG is .29998692 / MAP .12294964; the query standard errors of
the fine rule are .00274925 and .00180422 respectively.

All 6,000 TRAIN queries are reused for tuning. These are repeated development
comparisons, not an independent test. DEV was not loaded by either selection
script. The decision was frozen before scripts/report_fusion_dev.py reported
DEV through the actual production entry point. Fold1 TEST is excluded.

Reproduce selection with scripts/tune_fusion_train.py followed by
scripts/select_simple_fusion_train.py. The production constants are in
submission/ranking.py; scripts/report_fusion_dev.py measures that frozen rule.
Required handwritten RRF, CombSUM and CombMNZ functions remain unchanged.
