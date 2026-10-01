"""Resumable Qwen3.8-27B parallel extraction; never reads summaries or QA answers."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'src'))
from project_alexandria.backends import OpenAICompatibleBackend
from project_alexandria.io import write_json_atomic
from project_alexandria.pipeline import ExtractionConfig, KnowledgeUnitPipeline
from project_alexandria.experiments.reproduce import knowledge_unit_context

MODEL = 'Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound'
REVISION = '4756e3e4871aefd8d7cd5b0f6155ae5490451c1e'


def resilient(pipeline, papers):
    # Reuse the historical batch-failure isolation, plus transport-error isolation.
    payloads = [{'text': p['fulltext'], 'title': '', 'abstract': ''} for p in papers]
    try:
        return pipeline.extract_many(payloads)
    except (ValueError, RuntimeError):
        if len(papers) > 1:
            midpoint = len(papers) // 2
            return resilient(pipeline, papers[:midpoint]) + resilient(pipeline, papers[midpoint:])
        for attempt in range(2):
            try:
                return pipeline.extract_many(payloads)
            except (ValueError, RuntimeError):
                if attempt == 1:
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-documents', type=int, default=0, help='Smoke-test limit; 0 means all pending.')
    parser.add_argument('--skip-id', action='append', default=[])
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_documents < 0:
        parser.error('Invalid batch size or limit')
    papers = json.loads((ROOT / 'data/papers.json').read_text(encoding='utf-8'))
    config = ExtractionConfig(mode='parallel', chunk_words=500, context_words=1000,
                              canonicalization_max_tokens=1800, parse_retries=1)
    backend = OpenAICompatibleBackend('qwen38', base_url=args.base_url, api_key='',
        max_tokens=2500, temperature=0.2, concurrency=10, timeout=900, thinking=False)
    pipeline = KnowledgeUnitPipeline(backend, config)
    path = ROOT / 'kus.json'
    output = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'model': MODEL, 'revision': REVISION, 'config': asdict(config),
        'decoding': {'temperature': 0.2, 'top_p': 0.95, 'max_tokens': 2500,
                     'canonicalization_max_tokens': 1800, 'thinking': False, 'concurrency': 10},
        'runtime': {'vllm': '0.27.1', 'tensor_parallel_size': 2, 'max_model_len': 32768,
                    'gpu_model': 'RTX 3090', 'gpus': 2},
        'documents': [], 'batches': [], 'elapsed_seconds': 0.0}
    if (output['config'] != asdict(config) or output['model'] != MODEL
            or output['revision'] != REVISION):
        raise ValueError('Incompatible extraction checkpoint')
    by_id = {p['document_id']: p for p in papers}
    for document in output['documents']:
        if document['document_id'] not in by_id:
            raise ValueError('Checkpoint contains a document absent from current corpus')
        if document['fulltext_sha256'] != by_id[document['document_id']]['fulltext_sha256']:
            raise ValueError('Fulltext changed after extraction: ' + document['document_id'])
    completed = {d['document_id'] for d in output['documents']}
    output['excluded_ids'] = args.skip_id
    output['corpus_sha256'] = hashlib.sha256((ROOT / 'data/papers.json').read_bytes()).hexdigest()
    remaining = [p for p in papers if p['document_id'] not in completed | set(args.skip_id)]
    if args.max_new_documents:
        remaining = remaining[:args.max_new_documents]
    (ROOT / 'data/kus').mkdir(exist_ok=True)
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start:start + args.batch_size]
        print('EXTRACTING', [p['document_id'] for p in batch], flush=True)
        started = time.monotonic()
        results = resilient(pipeline, batch)
        elapsed = time.monotonic() - started
        for paper, result in zip(batch, results):
            identifier = paper['document_id']
            record = {'document_id': identifier, 'fulltext_sha256': paper['fulltext_sha256'],
                      'result': result.to_dict(), 'judge_context': knowledge_unit_context(result)}
            output['documents'].append(record)
            write_json_atomic(str(ROOT / 'data/kus' / (identifier + '.json')), record)
        output['batches'].append({'documents': [p['document_id'] for p in batch], 'seconds': elapsed})
        output['elapsed_seconds'] += elapsed
        write_json_atomic(str(path), output)
        print('EXTRACTED', len(output['documents']), '/', len(papers), 'batch_seconds', round(elapsed, 1), flush=True)


if __name__ == '__main__':
    main()
