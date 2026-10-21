"""Generate the required three-page report from verified model/CV evidence.

Report-only authoring dependencies: pip install reportlab pypdf pymupdf.
No dataset is read by this command. CV and preflight inputs are measurements.
"""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def make_report(record, preflight, cv_path, output=None, neural_record=None):
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from pypdf import PdfReader

    cv_path = Path(cv_path)
    cv = json.loads(cv_path.read_text())
    if len(cv.get('fits', [])) != 5 or not cv.get('summary'):
        raise ValueError('A completed five-fold CV report with summary is required')
    if cv.get('split') != 'Fold1 TRAIN only':
        raise ValueError('Report generalization evidence must use TRAIN-only CV')
    model_path = ROOT / record['model']
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Model does not match report input')
    model_hashes = [v for k, v in preflight['artifact_sha256'].items() if k.endswith('model.txt')]
    if model_hashes != [record['sha256']]:
        raise ValueError('Preflight measurement belongs to another main model')
    audit_path = ROOT/'runs/compliant_campaign/training_cache_audit.json'
    audit_hash = None
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        expected_counts = {'train': (723412, 6000), 'dev': (235259, 2000)}
        if audit.get('TEST_opened') is not False or not audit.get('combined_exactly_train_then_dev'):
            raise ValueError('Input audit does not verify the released splits')
        for split, (rows, queries) in expected_counts.items():
            item = audit['verified_splits'][split]
            if ((item['rows'], item['queries']) != (rows, queries)
                    or not item['raw_labels_queries_match_source']
                    or not item['query_transforms_recomputed_and_equal']):
                raise ValueError('Input audit is incomplete')
        shutil.copy2(audit_path, ROOT/'docs/TRAINING_AUDIT.json')
        audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    output = Path(output or ROOT/'output/pdf/2023CS10076_report.pdf')
    output.parent.mkdir(parents=True, exist_ok=True)
    W, H = A4
    ink, accent, muted, rule = [HexColor(v) for v in ('#172734', '#176879', '#536472', '#d7e0e5')]
    c = Canvas(str(output), pagesize=A4)
    c.setTitle('A3 Rank Fusion and Learning to Rank - Ahilaan Saxena')
    c.setAuthor('Ahilaan Saxena - 2023CS10076')
    margin, width = 44, W-88

    def text(s, y, size=10, color=ink, leading=14):
        p = Paragraph(s, ParagraphStyle('body', fontName='Helvetica', fontSize=size,
                      leading=leading, textColor=color))
        _, height = p.wrap(width, H)
        if y-height < 55:
            raise ValueError('Report page would overflow')
        p.drawOn(c, margin, y-height)
        return y-height-10

    def heading(title, page):
        c.setFillColor(ink); c.setFont('Helvetica-Bold', 20)
        c.drawString(margin, H-54, title)
        c.setFont('Helvetica', 10); c.setFillColor(muted)
        c.drawString(margin, H-74, 'Ahilaan Saxena | 2023CS10076 | COL764 Assignment 3')
        c.setStrokeColor(rule); c.line(margin, H-88, W-margin, H-88)
        c.setFont('Helvetica', 9)
        c.drawString(margin, 33, 'Sources: assignment specification and starter repository; local run receipts.')
        c.drawRightString(W-margin, 33, f'{page} / 3')
        return H-107

    def label(s, y):
        c.setFillColor(accent); c.setFont('Helvetica-Bold', 12)
        c.drawString(margin, y, s)
        return y-15

    def axes(x, y, w, h, low, high, title):
        c.setFillColor(ink); c.setFont('Helvetica-Bold', 11)
        c.drawString(x-35, y+h+27, title)
        c.setFont('Helvetica', 9); c.setFillColor(muted)
        c.drawString(x, y+h+10, 'Mean nDCG@10 (higher is better)')
        for i in range(5):
            value = low+(high-low)*i/4
            yy = y+h*i/4
            c.setStrokeColor(rule); c.line(x, yy, x+w, yy)
            c.drawRightString(x-8, yy-3, f'{value:.2f}')
        return lambda value: y+h*(value-low)/(high-low)

    y = heading('Methods and DEV measurements', 1)
    y = label('Ranking methods and data declarations', y)
    y = text('Fusion is handwritten, equal-weight min-max CombSUM over BM25 whole-document, '
             'BM25 title and LMIR Dirichlet whole-document. LMIR JM is disabled. '
             'The rule was selected on TRAIN only; RRF and CombMNZ retain their textbook definitions.', y)
    y = text('The main artifact is native LightGBM LambdaRank. Its declared feature schema is '
             f'<b>{escape(preflight["tree_feature_schema"])}</b>. '
             'It is fitted on 958,671 original TRAIN+DEV rows from 8,000 queries. '
             'Any augmented copies stay within complete query groups. The artifact loads without retraining.', y)
    pairwise = ('Pairwise is a plain 136-256-128-1 ReLU MLP, assembled from two relevance-label-trained '
                '128-64 networks within the staff width limits.' if neural_record is None else
                'Pairwise is a '+('fresh' if neural_record['args'].get('fresh') else 'refined')+' plain '+ '-'.join(map(str, [136]+neural_record['hidden']+[1]))+
                ' ReLU MLP within the staff width limits.')
    y = text('Pointwise uses handwritten squared-error training with TRAIN gradients and DEV epoch selection. '
             +pairwise+' It uses TRAIN+DEV gradients and frozen TRAIN-only standardization; '
             'no model-score distillation is used.', y)
    values = [('Pointwise', preflight['pointwise_ndcg@10']),
              ('Pairwise', preflight['pairwise_ndcg@10']), ('LambdaMART', preflight['ltr_ndcg@10'])]
    x, yy, w, h = 78, 281, 213, 181
    scale = axes(x, yy, w, h, 0, max(.7, max(v for _, v in values)+.08), 'Loaded-artifact DEV accuracy')
    for i, (name, value) in enumerate(values):
        xx = x+(i+.5)*w/len(values)
        c.setFillColor(accent if i == 2 else HexColor('#98aeb7'))
        c.rect(xx-22, yy, 44, scale(value)-yy, stroke=0, fill=1)
        c.setFillColor(ink); c.setFont('Helvetica-Bold', 9)
        c.drawCentredString(xx, scale(value)+7, f'{value:.5f}')
        c.setFont('Helvetica', 9); c.drawCentredString(xx, yy-17, name)
    curve = record.get('dev_fixed_counts')
    if not curve:
        curve = []
        for name in ['aug63_600', 'aug63_1200']:
            r = json.loads((ROOT/f'runs/compliant_campaign/{name}.json').read_text())
            curve.append({'trees': r['args']['rounds'], 'ndcg': r['dev_trained_on_ndcg']})
    dx, dw = 365, 176
    low = min(r['ndcg'] for r in curve)-.01
    high = max(r['ndcg'] for r in curve)+.01
    ds = axes(dx, yy, dw, h, low, high, 'DEV versus tree count')
    mn, mx = min(r['trees'] for r in curve), max(r['trees'] for r in curve)
    dp = lambda n: dx+10+(dw-20)*(n-mn)/max(1, mx-mn)
    c.setStrokeColor(accent); c.setFillColor(accent); c.setLineWidth(1.7)
    points = [(dp(r['trees']), ds(r['ndcg'])) for r in curve]
    for a, b in zip(points[:-1], points[1:]): c.line(*a, *b)
    for r, (a, b) in zip(curve, points):
        c.circle(a, b, 3, fill=1, stroke=0)
        c.setFont('Helvetica', 9); c.drawCentredString(a, yy-17, str(r['trees']))
    c.drawCentredString(dx+dw/2, yy-32, 'Boosting trees (count)')
    y = 229
    curve_caption = ('Right plot: this single-model refit at fixed tree counts.'
                     if record.get('dev_fixed_counts') else
                     'Right plot: a single seed-0 augmented baseline, not the final blend.')
    y = text(curve_caption+' DEV supplied gradients; this curve measures fitted DEV accuracy.',
             y, size=9, leading=12)
    y = text('<b>Interpretation:</b> DEV supplied gradients to the main and pairwise models, '
             'and stopping feedback to pointwise. These are development measurements, '
             'not independent generalization estimates. Loaded-artifact preflight also verifies '
             'finite scores, deterministic inference and the candidate/latency contract.', y)
    text('Fold1 TEST is excluded from the current fitting, augmentation, preprocessing and local selection. '
         'Earlier experiments consulted TEST in violation of the rules; refitting does not undo that '
         'history. The limitations are recorded accurately in docs/COMPLIANCE.md.', y, size=9, leading=12)
    c.showPage()

    summary = cv['summary']
    best = max(summary, key=lambda r: r['validation_mean'])
    y = heading('Grouped cross-validation', 2)
    y = text('Five query-disjoint folds use released TRAIN only. All documents for a query stay together. '
             'Query-relative features use no labels or corpus-fitted statistics. For augmented runs, '
             'augmentation occurs after the fold split. Each fit runs to the largest fixed tree count; '
             'reported folds are not used for early stopping.', y)
    signature = cv.get('signature', {})
    model_args = record.get('args', {})
    is_exact = (not record.get('constituent_receipts')
                and signature.get('config') == model_args.get('config')
                and signature.get('augment') == model_args.get('augment')
                and signature.get('normalization', 'rank') == model_args.get('normalization', 'rank')
                and signature.get('rank_only') == model_args.get('rank_only'))
    y = text(f'Source: <b>{escape(cv_path.name)}</b>. '
             + ('This is the current model-family sweep.' if is_exact else
                f'This is a {cv["params"]["num_leaves"]}-leaf baseline sweep, not CV of the exact final blend. '
                'Subsequent model changes also use reserved TRAIN development queries and declared DEV/portal feedback.'), y)
    if model_args.get('rounds', 0) > max(r['trees'] for r in summary):
        y = text(f'The final {model_args["rounds"]}-tree capacity exceeds this sweep. '
                 'It was chosen using fitted DEV and portal development feedback; '
                 'the reported CV does not validate that later capacity.', y)
    xx, yy, w, h = 83, 410, W-139, 166
    low = min(r['validation_mean']-r['validation_std'] for r in summary)-.015
    high = max(r['train_mean'] for r in summary)+.015
    scale = axes(xx, yy, w, h, low, high, 'Fixed tree-count sweep: training versus unseen queries')
    lo_count, hi_count = min(r['trees'] for r in summary), max(r['trees'] for r in summary)
    px = lambda n: xx+12+(w-24)*(n-lo_count)/max(1, hi_count-lo_count)
    for key, color in [('train_mean', muted), ('validation_mean', accent)]:
        c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.6)
        points = [(px(r['trees']), scale(r[key])) for r in summary]
        for a, b in zip(points[:-1], points[1:]): c.line(*a, *b)
        for a, b in points: c.circle(a, b, 2.5, fill=1, stroke=0)
    for r in summary:
        xval = px(r['trees'])
        c.setStrokeColor(accent)
        c.line(xval, scale(r['validation_mean']-r['validation_std']),
               xval, scale(r['validation_mean']+r['validation_std']))
        c.setFont('Helvetica', 9); c.setFillColor(muted)
        c.drawCentredString(xval, yy-15, str(r['trees']))
    c.drawCentredString(xx+w/2, yy-32, 'Boosting trees (count)')
    c.setFillColor(muted); c.drawString(xx+210, yy+h+11, 'Training')
    c.setFillColor(accent); c.drawString(xx+265, yy+h+11, 'Validation mean +/- fold SD')
    y = 346
    columns = [(margin, 'Trees'), (margin+85, 'Train mean'), (margin+188, 'Validation mean +/- SD'),
               (margin+372, 'Val / train')]
    c.setFillColor(ink); c.setFont('Helvetica-Bold', 9)
    for pos, title in columns: c.drawString(pos, y, title)
    y -= 22
    for r in summary:
        c.setFont('Helvetica', 10)
        cells = [str(r['trees']), f'{r["train_mean"]:.5f}',
                 f'{r["validation_mean"]:.5f} +/- {r["validation_std"]:.5f}',
                 f'{r["validation_train_ratio"]:.4f}']
        for (pos, _), value in zip(columns, cells): c.drawString(pos, y, value)
        y -= 21
    y -= 13
    y = text(f'The highest mean in this sweep is <b>{best["validation_mean"]:.5f}</b> '
             f'at {best["trees"]} trees. Its validation-to-training ratio is '
             f'{best["validation_train_ratio"]:.4f}. Growing training accuracy faster than '
             'validation accuracy exposes memorization of training queries. Fold standard deviations '
             'describe variation across these five query partitions, not a confidence guarantee.', y)
    text('The CV sweep and the reserved-query probes are development comparisons. Repeated selection, '
         'trained-on DEV measurements and portal feedback must not be represented as a fresh held-out test.', y)
    c.showPage()

    y = heading('Configuration and reproducibility', 3)
    y = label('Final configuration and tradeoff', y)
    def constituents(r, weight=1.):
        paths = r.get('constituent_receipts', [])
        if not paths: return [(weight, r)]
        pieces = []
        for i, path in enumerate(paths):
            pieces.extend(constituents(json.loads((ROOT/path).read_text()),
                          weight*(r['first_weight'] if i == 0 else 1-r['first_weight'])))
        return pieces
    leaves = constituents(record)
    descriptions = []
    for weight, item in leaves:
        a, p = item['args'], item['params']
        relative = 'percentile' if a.get('normalization', 'rank') == 'rank' else 'min-max'
        descriptions.append(f'{weight:.3f} x {a["rounds"]} trees, {p["num_leaves"]} leaves, '
                            f'seed {a["seed"]}, '+('augmented' if a['augment'] else 'original rows')+
                            ', '+(relative+'-only splits' if a['rank_only'] else 'raw and '+relative+' splits')+
                            ', '+('randomized thresholds' if p.get('extra_trees') else 'greedy thresholds'))
    y = text(escape('; '.join(descriptions))+'.', y)
    p = leaves[0][1]['params']
    y = text(f'Leaf-model settings: learning rate {p["learning_rate"]}, minimum leaf rows '
             f'{p["min_data_in_leaf"]}, L2 penalty {p["lambda_l2"]}, feature fraction '
             f'{p["feature_fraction"]}, bagging fraction {p["bagging_fraction"]}. '
             'Feature and row sampling with fixed seeds regularize the fits.', y)
    y = text('The configuration trades fitted DEV accuracy against generalization and robustness. '
             'Augmentation perturbs only TRAIN/DEV feature rows and retains relevance labels; '
             'query-relative transforms reduce reliance on absolute feature scales. Blending averages '
             'label-trained tree predictions within a native LightGBM artifact. More fitted DEV '
             'accuracy alone does not establish better accuracy on unseen queries.', y)
    y = text('Track C1 uses the generic recipe in submission/recipe.py: it reads only the two supplied '
             'benchmark input files, derives feature width from their schema, trains with a fixed seed '
             'and a training deadline, and writes a native model. Inference uses the artifact schema; '
             'it does not identify hidden datasets or branch on query IDs.', y)
    y = label('How to reproduce and verify', y-12)
    y = text('Install the grading requirements, prepare the released TRAIN/DEV files, then run the '
             'ordered reproduction commands in docs/CURRENT_MODEL.json. The main artifact SHA-256 is:<br/>'
             f'<font size="8">{record["sha256"][:32]}<br/>{record["sha256"][32:]}</font>', y)
    neural_doc = 'docs/CURRENT_NEURAL.json' if neural_record else 'docs/NEURAL_PROVENANCE.md'
    y = text('The neural settings and original TRAIN standardizer are documented in '
             +neural_doc+'. Use the unchanged staff artifact loader to check the plain MLPs. '
             'Run the artifact regression checks and scripts/preflight.py on released DEV. '
             'The toy smoke test belongs in a disposable copy because it overwrites the main artifact.', y)
    y = text(f'Preflight: prepare {preflight["prepare_seconds"]:.3f} seconds; largest-query rerank '
             f'{preflight["largest_query_seconds"]:.3f} seconds. The measured largest-query call is '
             'below the 2-second cap; the staff grader remains authoritative for the whole-run limit.', y)
    if audit_hash:
        y = text('The input audit matched cached features, labels, query IDs and recomputed transforms '
                 'for all 6,000 TRAIN and 2,000 DEV queries. The combined cache matches TRAIN then DEV '
                 'exactly. Source hashes and checks are in docs/TRAINING_AUDIT.json.', y)
    y = label('AI use and code provenance', y-12)
    text('Codex assisted with implementation, experiments, testing, documentation and authorized portal '
         'submissions; ChatGPT assisted with assignment analysis. Formula/code sources are the assignment '
         'and starter repository. LightGBM implements the permitted LambdaMART step; required fusion, '
         'pair construction and neural losses are handwritten. No external LTR implementation or teacher '
         'scores supervise the neural models. The student must understand and verify these declarations.', y)
    c.save()
    if len(PdfReader(str(output)).pages) != 3:
        raise ValueError('Expected exactly three report pages')
    (ROOT/'docs').mkdir(exist_ok=True)
    shutil.copy2(output, ROOT/'docs/REPORT.pdf')
    evidence = {'split': cv['split'], 'folds': 5, 'params': cv.get('params'),
                'summary': summary, 'source': str(cv_path),
                'source_sha256': hashlib.sha256(cv_path.read_bytes()).hexdigest(),
                'dev_fixed_count_curve': curve}
    (ROOT/'docs/CV_RESULTS.json').write_text(json.dumps(evidence, indent=2))
    manifest = {'model_sha256': record['sha256'], 'cv_source_sha256': evidence['source_sha256'],
                'artifact_sha256': {Path(k).name:v for k,v in preflight['artifact_sha256'].items()},
                'report_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'pages': 3}
    if audit_hash:
        manifest['training_audit_sha256'] = audit_hash
    (ROOT/'docs/REPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-receipt', required=True)
    ap.add_argument('--preflight', required=True)
    ap.add_argument('--cv-report', default=str(ROOT/'runs/optimization/query_extra_cv.json'))
    args = ap.parse_args()
    print(make_report(json.loads(Path(args.model_receipt).read_text()),
                      json.loads(Path(args.preflight).read_text()), args.cv_report))


if __name__ == '__main__':
    main()
