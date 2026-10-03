import hashlib
import json
from pathlib import Path
from common import ROOT, VENDOR, load, write, digest, PROMPT

bundle = load(VENDOR / 'data/testset.json')
if digest((VENDOR / 'data/testset.json').read_text()) != 'a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261':
    raise ValueError('Test bundle changed')
sources = []
for paper in bundle['papers']:
    if digest(paper['fulltext']) != paper['fulltext_sha256']:
        raise ValueError('Source mismatch')
    sources.append({k: paper[k] for k in ('document_id', 'fulltext', 'fulltext_sha256', 'source')})
if len(sources) != 97 or sum(len(x['questions']) for x in bundle['papers']) != 970:
    raise ValueError('Wrong cohort size')
write(ROOT / 'inputs/eval_sources_only.json', sources)
training = Path('/e/fscratch/reformo/schuhmann1/scientific-distillation-1000')
papers = {x['document_id']: x for x in load(training / 'inputs/papers.json')}
selected = []
finals = sorted((training / 'outputs/documents').glob('*/final.json'))
for path in finals:
    obj = load(path)
    paper = papers[obj['document_id']]
    if any(x['fulltext_sha256'] == paper['source_sha256'] for x in sources):
        raise ValueError('Holdout used for tuning')
    selected.append(dict(document_id=paper['document_id'], fulltext=paper['fulltext'],
        fulltext_sha256=paper['source_sha256'], domain=paper['domain'], summary=obj['summary']))
    if len(selected) == 64:
        break
if len(selected) != 64:
    raise ValueError('Need 64 non-holdout throughput examples')
write(ROOT / 'inputs/throughput_sources_only.json', selected)
write(ROOT / 'inputs/experiment_manifest.json', dict(
    repository='LAION-AI/project-alexandria', commit='5aac4b5ba2a78b20637e8ab960fe79d01ecd769a',
    testset_sha256=hashlib.sha256((VENDOR/'data/testset.json').read_bytes()).hexdigest(),
    documents=[x['document_id'] for x in sources], paper_count=97, question_count=970,
    prompt_sha256=digest(PROMPT), generation_temperature=0.2, top_p=0.95, thinking=False,
    max_output_tokens=16000, max_context_tokens=65536, precision_ornith='BF16',
    precision_qwen38='official FP8', original_qwen38_reference='AutoRound mixed INT4',
    correction='one full source-only semantic correction plus bounded repository V3 structural/quote repair',
    audit='one source-only self-audit before and after correction; invalid audit retried once',
    failures='retained in 97-paper denominator; missing summary counted wrong for all ten MCQs',
    benchmark_tuning_sources='64 existing training papers, no eval QA or papers used for tuning',
    benchmark_output_budget=512, benchmark_note='bounded output probes followed by complete-paper timings',
    judge='Qwen/Qwen2.5-7B-Instruct', judge_temperature=0.5, judge_concurrency=4,
    judge_precision='BF16', judge_max_output_tokens=100, parser='historical_semicolon_v1',
    option_order='immutable exported order', bootstrap_resamples=10000, bootstrap_seed=250219413,
    training_contamination='all eval inputs and outputs in separate eval workspace'))
print('PREPARED', len(sources), len(selected), digest(PROMPT))
