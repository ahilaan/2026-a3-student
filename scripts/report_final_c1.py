"""Render measured final-pass evidence as reports and flat CSV (no datasets)."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/final_c1_pass'


def read(path):return json.loads((OUT/path).read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--draft',action='store_true',help='Show verified work while the fresh refit is running')
    args=ap.parse_args()
    import numpy as np
    fits=read('study/fits.json');summary=read('study/summary.json')
    decision=read('decision.json');regression=read('regressions.json')
    assembly=read('assembly/verification.json')
    fresh=read('fresh/verification.json') if (OUT/'fresh/verification.json').exists() else None
    if not args.draft and (fresh is None or not (OUT/'stress_adaptive/stress_dense500k.json').exists()):
        raise RuntimeError('Fresh reproduction and final dense stress must finish before finalizing')
    if fresh is None:
        fresh_text=('Fresh staged refit is still running. The complete 2,000-tree/127-leaf component '
                    'matches its original SHA-256; the 63-leaf component is in progress. '
                    'Full end-to-end fresh reproduction is not yet claimed.')
    else:
        fresh_text=(f"Fresh staged refit: byte equality **{fresh['byte_equal']}**; prediction equality "
                    f"**{fresh['prediction_equal']}**; ranking equality **{fresh['ranking_equal']}**; "
                    f"maximum score difference {fresh['maximum_prediction_difference']}; "
                    f"{fresh['trees']} trees/{fresh['features']} features; {fresh['seconds']:.2f} seconds.")
    verified_version=read('environment.json')['lightgbm']
    pending=[]
    if fresh is None:pending.append('fresh full-model reproduction')
    if not (OUT/'stress_adaptive/stress_dense500k.json').exists():pending.append('final adaptive dense stress')
    if not (OUT/'packaging.json').exists():pending.append('ZIP packaging')
    status=('Pending: '+', '.join(pending)+'. Completed measurements below are verified.') if args.draft else 'Final measured results and verification.'
    preflight=read('preflight.json');edges=read('edges.json')
    baseline=[r for r in fits if r['config']=='baseline']
    all_fits=fits+read('confirmation/fits.json')
    paired={ (r['dataset'],r['fold']):r for r in all_fits if r['config']=='baseline' }
    columns=['dataset','family','fold','config','train_rows','inner_rows','outer_rows','train_ndcg','inner_ndcg','outer_ndcg',
             'inner_outer_gap','fit_seconds','total_seconds','best_iteration','saved_trees','model_bytes','peak_process_bytes','success']
    with (OUT/'experiments.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['phase']+columns+['paired_outer_delta','params','metadata'])
        writer.writeheader()
        for index,r in enumerate(all_fits):
            row={k:r[k] for k in columns}
            row.update(phase='primary' if index<len(fits) else 'confirmation',
                       paired_outer_delta=r['outer_ndcg']-paired[(r['dataset'],r['fold'])]['outer_ndcg'],
                       params=json.dumps(r['params'],sort_keys=True),metadata=json.dumps(r['metadata'],sort_keys=True))
            writer.writerow(row)
    baseline_rows=[]
    for name in dict.fromkeys(r['dataset'] for r in baseline):
        rows=[r for r in baseline if r['dataset']==name]
        baseline_rows.append([name,len(rows),f"{np.mean([r['train_ndcg'] for r in rows]):.6f}",
                              f"{np.mean([r['inner_ndcg'] for r in rows]):.6f}",
                              f"{np.mean([r['outer_ndcg'] for r in rows]):.6f}",
                              f"{np.std([r['outer_ndcg'] for r in rows]):.6f}",
                              f"{np.mean([r['total_seconds'] for r in rows]):.2f}"])
    candidate_rows=[]
    for config in dict.fromkeys(r['config'] for r in fits):
        rows=[r for r in fits if r['config']==config]
        deltas=[r['outer_ndcg']-paired[(r['dataset'],r['fold'])]['outer_ndcg'] for r in rows]
        candidate_rows.append([config,len(rows),f"{np.mean([r['outer_ndcg'] for r in rows]):.6f}",
                               f'{np.mean(deltas):+.6f}',f'{min(deltas):+.6f}',
                               f"{np.mean([r['total_seconds'] for r in rows]):.2f}"])
    strategy_rows=[[r['strategy'],f"{r['outer_mean']:.6f}",f"{r['outer_std']:.6f}",
                    f"{r['mean_seconds']:.2f}",f"{r['worst_regression']:+.6f}",f"{r['successes']}/{r['conditions']}"]
                   for r in summary['strategies']]
    stresses=[json.loads(p.read_text()) for p in sorted((OUT/'stress').glob('*.json'))]
    adaptive_stresses=[json.loads(p.read_text()) for p in sorted((OUT/'stress_adaptive').glob('*.json'))]
    profiles=read('profile_summary.json')
    profile_table=table(['Procedure','Outer conditions','Outer mean','Paired delta','Worst delta','Mean API s'],
                        [[r['policy'],r['conditions'],f"{r['outer_mean']:.6f}",f"{r['paired_mean_delta']:+.6f}",
                          f"{r['worst_delta']:+.6f}",f"{r['mean_seconds']:.2f}"] for r in profiles])
    confirmation=read('confirmation/decisions.json')
    confirmation_table=table(['Confirmation condition','Outer folds','Mean selector delta','Worst delta','Rank choices'],
                             [[name,len(rows),f"{np.mean([r['delta'] for r in rows]):+.6f}",
                               f"{min(r['delta'] for r in rows):+.6f}",sum(r['selected']=='query_ranks' for r in rows)]
                              for name in sorted(set(r['dataset'] for r in confirmation))
                              for rows in [[r for r in confirmation if r['dataset']==name and r['policy']=='raw_rank_selection']]])
    adaptive_stress_table=table(['TRAIN rows','D','Density','Total recipe s','Peak MiB','Chosen D','Candidate fits','Loads'],
                               [[r['rows'],r['width'],r['background_density'],f"{r['seconds']:.2f}",
                                 f"{r['peak_working_set_bytes']/2**20:.1f}",r['model_features'],
                                 len(r.get('candidate_fits',[])) if 'candidate_fits' in r else 'not instrumented',r['loadable']]
                                for r in adaptive_stresses])
    c1_regression=read('final_recipe_regression.json')
    c1_regression_table=table(['Provided transfer set','Original vali','Final vali','Final repeat byte equal'],
                              [[r['dataset'],f"{r['baseline']['ndcg@10']:.9f}",f"{r['final']['ndcg@10']:.9f}",
                                r['same_seed_repeat_byte_equal']] for r in c1_regression])
    compact={'decision':decision,'strategy_comparison':summary,'protocol':read('study/protocol.json'),
             'environment':read('environment.json'),'small_query_optimization':read('small_query_optimization.json'),
             'reproduction':{'fresh':fresh,'retained_component_assembly':assembly},
             'regression':regression,'preflight':preflight,'stress':stresses,'edges':edges,
             'adaptive_stress':adaptive_stresses,'production_api_agreement':read('production_api_agreement.json'),
             'engineered_inference':read('engineered_inference.json'),'final_toy_regression':c1_regression}
    for name in ['confirmation/decisions.json','confirmation/protocol.json','profiles/results.json']:
        if (OUT/name).exists():compact[name]=read(name)
    (ROOT/'docs/FINAL_C1_RESULTS.json').write_text(json.dumps(compact,indent=2),encoding='utf-8')
    stress_rows=[[r['rows'],r['width'],r['background_density'],f"{r['seconds']:.2f}",
                 f"{r['peak_working_set_bytes']/2**20:.1f}",r['trees'],r['loadable']] for r in stresses]
    report=f'''# {'DRAFT: ' if args.draft else ''}Final optimization and compliance pass - 21 October 2026

{status}

## Decision

{decision['recommendation']}

{decision['reason']}

Hidden C1 performance is unknown. No leaderboard submission was made.

## Compliance and current-model reproduction

`scripts/train_model.py --config current-main` orchestrates the existing full
trainer and native-tree blender. The legacy `basic` and `submitted` modes are
unchanged in behavior; `submitted` is explicitly labeled as the historical
200-tree configuration. Outputs under `submission/` are rejected by the new
current-main mode. Fresh fits and retained-component assembly are distinct.

```bash
python scripts/train_model.py --config current-main --train data/train --dev data/dev --use-dev --work-dir runs/current_reproduction --out runs/current_reproduction/model.txt
```

The actual fresh run used `runs/final_c1_pass/fresh` instead of
`runs/current_reproduction`. Only released TRAIN and DEV were read. Source
hashes, raw cached arrays, labels, query IDs, within-query transforms and the
TRAIN-then-DEV combined matrix were audited before fitting. Evidence:
`runs/final_c1_pass/fresh/input_audit.json`.

Architecture: two native LambdaRank components, 2,000 trees/127 leaves/seed 0
and 2,400 trees/63 leaves/seed 23. The blend scales leaf outputs by .65/.35
and concatenates trees. Both fit 958,671 original TRAIN+DEV rows/8,000
queries plus one augmented copy (1,917,342 fit rows/16,000 query blocks).
Their 1,200-tree save/reload boundaries are preserved; continuation resets
training state as in the actual workflow. Hyperparameters: learning rate .05,
minimum leaf 100, L2 10, feature/row fractions .8, bagging every round,
extra randomized splits, 63 bins, deterministic row-wise training, four
threads and all explicit RNG seeds equal to the component seed. No stopping
or gradients use held-out data. Original plus average-tie query percentiles
give 272 features. Augmentation RNG is `76420 + seed`: Gaussian noise .02
times query feature range, per-query log-normal column scaling sigma .5,
and 14-40 dropped columns, followed by the same inference transform.

Retained-component assembly: byte equality **{assembly['byte_equal']}**;
prediction equality **{assembly['prediction_equal']}**; ranking equality
**{assembly['ranking_equal']}** on {assembly['comparison_queries']} TRAIN queries.
{fresh_text}
Exact commands and hashes are in the respective `verification.json` files.
These results apply to local LightGBM {verified_version}; Ubuntu/Python 3.12
cross-platform byte identity was not tested. Matching hyperparameters alone
are not treated as a reproduction result.

README and SUBMISSION_NOTES now state current splits, augmentation, component
settings, tuning exposure, the correct standard entry point and C1 behavior.
C1_REPRODUCIBILITY and BLEND_SELECTION are explicitly historical. The old
public-data study's permission claim is labeled unverified and excluded from
this pass. Known historical TEST-label and public LTR-label usage is preserved
in COMPLIANCE and the original research records. A new refit cannot erase that
selection exposure or establish that the historical process complied. AI and
code-provenance disclosures remain. The existing assignment PDF/report were
not rewritten to conceal history.

## Baseline protocol and results

There are 39 condition/fold pairs: three provided toy sets (one outer split
each), three generated datasets (three outer folds each), and nine MSLR TRAIN
size/feature conditions (three outer folds each). For toy sets, provided TRAIN
is divided into gradient/stopping queries; provided validation is outer-only.
Generated datasets use widths 17/90/300, query counts 240/360/240, densities
1/.1/.02 and labels with 2/3/5 levels, seeds 171/172/173. They include singleton
and zero-label queries. MSLR pools contain 300/900/2,250 complete released TRAIN
queries sampled with seed 193, with leading 40/90/136-column subsets. Outer
grouped folds use seed 41; one fifth of available queries is inner validation
using seed 1200+fold. All models use seed 0. No row-random splitting occurs.
Nested pools and feature variants share queries; these are correlated
conditions, not 39 independent replications or a significance estimate.

Original baseline: LambdaRank, 15 leaves, minimum leaf 50, L2 1, rate .05,
feature/row fractions .8, 255 bins, truncation 30, maximum 800 rounds and
patience 50. All 39 baseline models saved/reloaded with finite predictions and
matched the standard ranking interface. Baseline fitting time total:
{sum(r['fit_seconds'] for r in baseline):.2f} seconds.

'''+table(['Condition','Folds','Train nDCG','Inner nDCG','Outer nDCG','Outer SD','Mean procedure s'],baseline_rows)+f'''

Three full-input synthetic staff-interface runs are separately recorded in
`baseline_api.json`; their supplied validation scores are mechanical checks,
not independent outer scores. Per-fit best iterations, saved trees, model
bytes, matrix bytes, process-memory high-water marks, train/inner/outer scores
and gaps are recorded in `study/fits.json` and `experiments.csv`.

## Staged experiments

Candidates were registered in `study/protocol.json` before fitting. Screening
uses nine representatives: all toy sets, first fold of each generated set,
and first full-width fold at each MSLR size. Practical finalists (baseline,
31 leaves, regularized 31 leaves, fast15) were predetermined and then evaluated
on every condition. The query-transform extension was registered based only
on screening inner scores, before reviewing outer results; its fixed gate is
recorded in `study/transform_extension.json`. Every meaningful override is in
the protocol and each fit's parameter dictionary. No Cartesian search was run.

'''+table(['Candidate','Fits','Outer mean','Mean paired delta','Worst delta','Mean procedure s'],candidate_rows)+f'''

Candidate means with different fit counts use different subsets and are not
directly comparable; paired deltas use each candidate's matching baseline.
Screening covers 7/15/31/63 leaves, minimum leaves 20/50/100/200, L2 0/1/5/10,
L1/minimum gain, feature fractions .7/.8/1, row sampling, rates .03/.05/.1,
up to 1,200 rounds, patience 20/35/50/75, 63/127/255 bins, LambdaRank
truncations 10/13/20/default30 and diagnostic rank_xendcg. Raw dense and CSR
handling and the existing query-percentile schema are tested. Data statistics
include queries, rows, dimensions, density, label distribution, multigrade
queries, documents/query and constant columns. The simple adaptive rule uses
31 leaves for at least 400 gradient queries, otherwise baseline. No complex
metadata rule is fitted to outer scores.

## Complete strategy comparison

The fixed alternative is chosen using inner scores only. Validation search
compares baseline/31/regular31 and retains the simpler earlier candidate unless
inner nDCG improves by more than .001. The hybrid compares baseline with the
size-rule choice. Raw/rank selection uses the same inner-only tie rule. Strategy
runtime sums the measured constituent procedures; it excludes source loading
and is not a fresh-process API benchmark. The actual end-to-end procedure,
two-candidate selection and 200-tree race/continuation are separately measured
in `profiles/results.json`, including parsing. All outer outcomes measure the
entire corresponding inner-driven procedure, not the best outer candidate.

'''+table(['Strategy','Outer nDCG','SD','Mean procedure s','Worst delta','Loadable'],strategy_rows)+f'''

Family-specific deltas and selection decisions are in `study/summary.json`
and `decision.json`. Model-selection/stopping never consume outer labels.
Small correlated-fold differences do not establish statistical significance.
No ensemble or dataset-specific clipping/removal schema is deployed without
evidence. LightGBM already filters unsplittable constant columns; keeping their
schema positions preserves inference. The NaN/finite-overflow behavior is
tested using generated input. Explicit infinities/malformed labels fail with
context. Transformations requiring unavailable sidecars are avoided.

### Actual API allocation comparison

'''+profile_table+f'''

These nine representatives are the first fold of each generated condition,
all three toy sets, and the full raw width at each MSLR pool size. They are a
subset of the 39 primary conditions, not additional independent replications.
The two-model race gives each candidate at most 200 trees and continues only
the inner winner. It does not reliably improve outer accuracy. Baseline and
raw/rank chosen artifacts agree byte-for-byte with the in-memory study in all
nine cases. The deployed production implementation was rerun on all nine and
also matches the evaluated selector's chosen model bytes. Normal-case behavior
is therefore verified through the actual parser, training and serialization.
Emergency-budget allocation is tested separately; production gives the raw
baseline its full shared deadline before considering optional features.

### Confirmation before deployment

'''+confirmation_table+f'''

These 12 additional outer folds were registered based only on inner screening,
before any outer summaries were reviewed. The 1,800-query MSLR pool is disjoint
from the initial 2,250-query pool: shuffled TRAIN IDs [2250:4050], seed 193.
Outer folds use seed 84. Generated cases use 360 queries each, seeds 471/472/473:
90 columns with label-preserving per-query positive scales and shifts; 40
columns with monotone exponential scales; and 300 sparse columns with random
three-grade labels. The last case intentionally has no reliable feature signal.
The disjoint MSLR mean change is +0.009717. The large scale-shift gain is a
constructed stress condition and is not a prediction of hidden C1 improvement.
Weak/noisy labels regress by 0.008150 on average; worst confirmation change
is -0.015667. No weak-label heuristic was added after viewing those results.
All predeclared initial and confirmation gates passed (decision.json).

## Selected C1 implementation

`submission/recipe.py` retains the original shallow LambdaRank parameters.
It loads TRAIN/validation once, saves the completed raw baseline, and scores
its validation predictions using the staff stable-tie/zero-relevance convention.
Only if sufficient shared time remains does it append complete-query average
percentiles and fit the same ranker again. A validation improvement above .001
is required to atomically replace the raw artifact. No validation gradients,
train+validation refit, learned preprocessing statistics, external files,
dataset identities, hard-coded feature positions or ensembles are used.
Feature names encode the raw/query-percentile schema, already supported by the
unchanged prepare/rerank interface. Only one selected native model is loaded.

The raw baseline retains the entire 960-second soft deadline. Optional work
requires max(30 seconds, three times measured baseline cost) remaining. Query
feature generation checks the shared deadline between original queries while
reserving max(30 seconds, twice baseline cost) for fitting/saving. Abandoning
optional feature generation leaves the raw model intact. Candidate boosting
uses the same shared deadline and public best-iteration stopping mechanism.
The .001 tie threshold and two-candidate policy were frozen before outer
evaluation; they were not tuned against confirmation outcomes.

## Runtime, memory and reliability

Original fixed baseline, before the representation selector:

'''+table(['TRAIN rows','D','Background density','Recipe s','Peak working set MiB','Trees','Loads'],stress_rows)+f'''

Selected final adaptive policy:

'''+adaptive_stress_table+f'''

Stress inputs have five dense signal columns, generated noisy relevance,
100 documents/query, 50,000 separate validation rows, 300 columns and comments.
Both parsing passes, training and saving are timed; generation is excluded.
Peak memory is process working set, not an enforced 8 GB cap. Measurements
were made on Windows, with other local work running; staff machine speed,
memory accounting and distributions may differ. Edge cases additionally cover
5,000-document groups, 5,000 tiny queries, zeros/constants and validation-only
indices, with group sums verified. NaN, malformed/non-finite values,
interleaved IDs, ties, singleton queries and forced-expiry behavior are tested.

The production soft timer starts before import/parsing and targets 960 seconds,
reserving max(15 seconds, twice recent iteration cost). Public LightGBM
EarlyStopException retains the best completed validation iteration. A valid
first-tree fallback is written before further iterations; atomic replacement
keeps it until final serialization succeeds. The raw candidate remains byte-equal
to the original when the timer is inactive. This cannot preempt
parsing, dataset construction or a single native iteration, and does not
guarantee recovery from a killed/failed staff process. Emergency time-based
tree counts depend on machine load; normal fixed-seed runs are repeatable.

The original recipe was also verified to fail on a 10,001-document query
because of LightGBM's native 10,000-document limit. Only oversized queries
are split into contiguous training blocks of at most 10,000 documents, never
mixing original queries. In this case stopping uses a custom metric on whole
original validation queries, including stable ties and zero-relevance queries.
Cross-block training pairs are omitted, a known approximation. Normal query
groups and normal validation behavior remain unchanged. The native failure,
group-boundary invariants, staff metric agreement and successful 10,001-row
training/validation are recorded in native_group_limit.json and targeted tests.

For two-document queries, optional training percentiles use an exact vectorized
equivalent of the existing inference transform. The 5,000-tiny-query workload
fell from 43.41 to 2.88 seconds with identical model bytes. Float32/float64,
ties, missing values and non-finite ordering match the existing transform;
three generated production cases also retain identical model bytes.
Evidence: `small_query_optimization.json`. A minimal staff-module isolation
check confirms the recipe has no dependency on the offline experiment scripts.

## Regression, tests and packaging

All protected artifact/source hashes are unchanged except the intentionally
modified recipe. On {regression['comparison_queries']} authorized TRAIN queries
({regression['comparison_documents']} documents), main scores/order/IDs and
fusion outputs match exactly. Neural scores match through the staff loader.
Final repeated-seed C1 artifacts match byte-for-byte on all three provided
transfer sets. Two retain the original raw model; the 250-feature set selects
the query-percentile model. These supplied-validation scores are mechanical
selection checks, not outer generalization evidence:

'''+c1_regression_table+f'''

Evidence: `regressions.json`, `final_recipe_regression.json`.

Real DEV preflight: main nDCG {preflight['ltr_ndcg@10']:.6f}, pointwise
{preflight['pointwise_ndcg@10']:.6f}, pairwise {preflight['pairwise_ndcg@10']:.6f},
BM25 {preflight['bm25_ndcg@10']:.6f}. DEV supplies main/pairwise gradients and
pointwise selection: these are regression/fitting measurements, not new tests.
Largest-query inference {preflight['largest_query_seconds']:.3f} seconds;
prepare {preflight['prepare_seconds']:.3f} seconds; preflight passes
{preflight['passes']}.

Test logs are in `runs/final_c1_pass/`: actual-artifact suite **126 passed,
1 deselected** (the excluded end-to-end test expects a toy-trained main model),
isolated toy-trained full suite **127 passed**, plus focused reproducibility,
fallback, selector and module-isolation tests.
The isolated copy protects production weights. Packaging verification is
recorded in `packaging.json`; the reviewed ZIP is separate from the original
archive, contains all seven required entries, excludes raw datasets/runs,
and must remain below 50 MB. No automatic submission occurs.
'''
    (ROOT/'docs/FINAL_C1_PASS.md').write_text(report,encoding='utf-8')
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print('Wrote docs/FINAL_C1_PASS.md, runs/final_c1_pass/REPORT.md and experiments.csv')


if __name__=='__main__':main()
