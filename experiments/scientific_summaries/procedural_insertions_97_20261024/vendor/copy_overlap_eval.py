"""Required copy-overlap evidence for scientific summary evaluations."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
try:
    from .ngram_overlap import SourceIndex,audit_summary,VERSION
except ImportError:
    from ngram_overlap import SourceIndex,audit_summary,VERSION


def compact(rows):
    generated=[r for r in rows if r.get('audit')]
    report=dict(paper_slots=len(rows),audited_generated_outputs=len(generated),
        generation_failures_or_empty_outputs=len(rows)-len(generated),
        all_fields_strict_five_word_pass=sum(max(c['longest_contiguous_match_words'] for c in r['audit']['categories'].values())<=5 for r in generated),
        all_fields_six_word_tolerance_pass=sum(max(c['longest_contiguous_match_words'] for c in r['audit']['categories'].values())<=6 for r in generated))
    for category in ['narrative','evidence','metadata']:
        values=[r['audit']['categories'][category] for r in generated]
        report[category]=dict(
            strict_five_word_pass=sum(v['longest_contiguous_match_words']<=5 for v in values),
            borderline_six_only=sum(v['longest_contiguous_match_words']==6 for v in values),
            violation_seven_plus=sum(v['longest_contiguous_match_words']>=7 for v in values),
            maximum_contiguous_match_words=max((v['longest_contiguous_match_words'] for v in values),default=0),
            mean_longest_match_words=statistics.mean(v['longest_contiguous_match_words'] for v in values) if values else 0,
            mean_covered_fraction_six_plus=statistics.mean(v['covered_word_fraction_by_minimum_run']['6'] for v in values) if values else 0,
            mean_covered_fraction_seven_plus=statistics.mean(v['covered_word_fraction_by_minimum_run']['7'] for v in values) if values else 0,
            pooled_covered_fraction_six_plus=sum(v['covered_words_by_minimum_run']['6'] for v in values)/max(1,sum(v['words'] for v in values)),
            pooled_covered_fraction_seven_plus=sum(v['covered_words_by_minimum_run']['7'] for v in values)/max(1,sum(v['words'] for v in values)),
            evidence_fragments_over_five_words=sum(v['evidence_fragments_over_five_words'] for v in values))
    return report


def markdown(report):
    lines=['## Source-copy overlap audit','',
        'Exact normalized contiguous word overlap against each complete paper source. '
        'Five words are allowed, six are borderline, seven or more are flagged. '
        'Narrative prose, evidence quotes and bibliographic metadata are reported separately. '
        'Coverage counts each summary word once per threshold within each fragment. '
        'Separate statements are never concatenated to create matches.','',
        '| Condition | Audited outputs | Narrative ≤5 | Narrative exactly 6 | Narrative ≥7 | Longest narrative match | Mean narrative coverage ≥6 | Evidence ≥7 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,v in report['conditions'].items():
        n=v['narrative'];e=v['evidence']
        lines.append(f"| {label} | {v['audited_generated_outputs']} | {n['strict_five_word_pass']} | {n['borderline_six_only']} | {n['violation_seven_plus']} | {n['maximum_contiguous_match_words']} | {100*n['mean_covered_fraction_six_plus']:.2f}% | {e['violation_seven_plus']} |")
    lines+=['','Primary counts use whitespace-delimited words after Unicode normalization and punctuation stripping; mathematical expressions, '
        'decimal numbers and hyphenated terms are not split into artificial extra words. A separate punctuation-split diagnostic is retained in the JSONL. '
        'NFKC normalization, casefolding, soft-hyphen removal and PDF line-end word joining are used. '
        'Offsets refer to normalized text and word positions. Technical names and ordinary scientific phrases can match literally; '
        'lexical overlap alone does not establish plagiarism. All output flags remain visible and no paper is removed from QA. '
        'Reasoning traces, instructions and reviewer verdicts are not summary prose. '
        'Bibliographic names and titles are included in the metadata diagnostics, not silently mixed into paraphrase statistics.','']
    return '\n'.join(lines)


def audit_conditions(papers,contexts,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    rows=[]
    for p in papers:
        source=p['fulltext'];index=SourceIndex(source)
        expected=p.get('fulltext_sha256',p.get('source_sha256'))
        if expected:assert expected==index.sha256
        for label,generated in contexts.items():
            row=generated[p['document_id']]
            audit=None
            if row.get('status')=='generated' or row.get('summary'):
                audit=audit_summary(source,row.get('summary'),row.get('judge_context'),index=index)
            rows.append(dict(condition=label,document_id=p['document_id'],source_sha256=index.sha256,
                status=row.get('status'),audit=audit))
    labels=list(contexts)
    report=dict(audit_version=VERSION,complete=True,elapsed_seconds=time.monotonic()-start,
        allowed_consecutive_words=5,borderline_consecutive_words=6,violation_from_consecutive_words=7,
        conditions={label:compact([r for r in rows if r['condition']==label]) for label in labels},
        qa_filtering=False,qa_predictions_or_gold_used=False)
    (output/'copy_overlap.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'copy_overlap_details.jsonl').open('w') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    csvrows=[]
    for row in rows:
        flat={k:row[k] for k in ['condition','document_id','source_sha256','status']}
        for c in ['narrative','evidence','metadata']:
            a=row['audit']['categories'][c] if row['audit'] else {}
            flat.update({c+'_longest_match':a.get('longest_contiguous_match_words'),
                c+'_classification':a.get('classification'),c+'_coverage_6plus':a.get('covered_word_fraction_by_minimum_run',{}).get('6'),
                c+'_coverage_7plus':a.get('covered_word_fraction_by_minimum_run',{}).get('7')})
        csvrows.append(flat)
    with (output/'copy_overlap.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    text=markdown(report);(output/'COPY_OVERLAP.md').write_text(text)
    result=output/'RESULTS.md'
    if result.exists():
        previous=result.read_text().split('\n## Source-copy overlap audit')[0]
        result.write_text(previous.rstrip()+'\n\n'+text)
    return report


def audit_cohort_root(root,papers,models):
    """Required sidecar for the older raw/corrected cohort evaluators."""
    root=Path(root);contexts={}
    for model in models:
        for phase in ['raw','corrected']:
            generated={}
            directory=root/'outputs/cohorts'/model/'documents'
            for paper in papers:
                record=json.loads((directory/paper['document_id']/'document.json').read_text())
                expected=record.get('source_sha256',record.get('fulltext_sha256'))
                if expected:assert expected==hashlib.sha256(paper['fulltext'].encode()).hexdigest()
                value=record.get(phase) or {}
                generated[paper['document_id']]=dict(value,status='generated' if value.get('summary') else 'generation_failed')
            contexts[model+'_'+phase]=generated
    return audit_conditions(papers,contexts,root/'outputs')
