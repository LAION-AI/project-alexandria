"""Freeze a reproducible 100-paper fulltext/existing-summary convenience sample."""
import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIELDS = ('executive_summary', 'research_context', 'research_question_hypothesis',
          'methodological_details', 'procedures_architectures', 'key_results',
          'interpretation_implications', 'contradictions_limitations', 'claims',
          'data_code_availability', 'robustness_ablation_notes', 'ethical_considerations',
          'key_figures_tables', 'three_takeaways')
REVISION = '232547b7b938a6588ce5d9d7faa96384860b7a16'

def fetch(config, offset):
    url = ('https://datasets-server.huggingface.co/rows?dataset=laion%2FScientific-Summaries'
           f'&config={config}&split=train&offset={offset}&length=50')
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.load(response)['rows']
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

def main():
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DATA,
                        help='Use a new directory; never overwrite the frozen published corpus.')
    DATA = parser.parse_args().output_dir
    if (DATA / 'papers.json').exists():
        raise SystemExit('Frozen corpus already exists. Use it for reproduction, or select a new --output-dir.')
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'papers').mkdir(exist_ok=True)
    (DATA / 'summaries').mkdir(exist_ok=True)
    papers = []
    seen = set()
    for config, offsets in [('arxiv', [100, 10000, 100000, 500000, 1000000]),
                            ('bethgelab', [100, 10000, 50000, 100000, 200000])]:
        accepted = []
        for offset in offsets:
            rows = fetch(config, offset)
            random.Random(250219413 + offset).shuffle(rows)
            batch = []
            for wrapped in rows:
                row = wrapped['row']
                text = row.get('text_sanitized') or row.get('text_raw') or ''
                summary = '\n\n'.join(f'{field.replace("_", " ").title()}\n{row[field]}'
                                      for field in FIELDS if row.get(field))
                digest = hashlib.sha256(text.encode()).hexdigest()
                if (wrapped.get('truncated_cells') or not 8000 <= len(text) <= 30000
                        or len(text.split()) < 1500 or len(summary) < 2000
                        or row.get('oa_is_retracted') or digest in seen
                        or not row.get('source_title') or not row.get('key_results')):
                    continue
                # Prefer body text with explicit conclusions/results, avoiding abstract-only records.
                if not any(s in text.lower() for s in ('conclusion', 'discussion', 'references')):
                    continue
                identifier = f'{config}-{wrapped["row_idx"]}'
                paper = dict(row)
                paper.update(document_id=identifier, viewer_config=config,
                             viewer_row_index=wrapped['row_idx'], fulltext=text,
                             existing_summary=summary, fulltext_sha256=digest,
                             summary_sha256=hashlib.sha256(summary.encode()).hexdigest())
                batch.append(paper)
                seen.add(digest)
                if len(batch) == 10:
                    break
            accepted.extend(batch)
            print(config, offset, 'accepted', len(batch), flush=True)
        if len(accepted) < 50:
            offset = 250000
            while len(accepted) < 50:
                for wrapped in fetch(config, offset):
                    row = wrapped['row']
                    text = row.get('text_sanitized') or row.get('text_raw') or ''
                    summary = '\n\n'.join(f'{f.replace("_", " ").title()}\n{row[f]}' for f in FIELDS if row.get(f))
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if (wrapped.get('truncated_cells') or not 8000 <= len(text) <= 30000
                            or len(text.split()) < 1500 or len(summary) < 2000 or digest in seen
                            or not row.get('source_title') or not row.get('key_results')
                            or row.get('oa_is_retracted')):
                        continue
                    identifier = f'{config}-{wrapped["row_idx"]}'
                    paper = dict(row)
                    paper.update(document_id=identifier, viewer_config=config,
                                 viewer_row_index=wrapped['row_idx'], fulltext=text,
                                 existing_summary=summary, fulltext_sha256=digest,
                                 summary_sha256=hashlib.sha256(summary.encode()).hexdigest())
                    accepted.append(paper)
                    seen.add(digest)
                    if len(accepted) == 50:
                        break
                offset += 50
        papers.extend(accepted[:50])
    for paper in papers:
        identifier = paper['document_id']
        (DATA / 'papers' / f'{identifier}.txt').write_text(paper['fulltext'], encoding='utf-8')
        (DATA / 'summaries' / f'{identifier}.txt').write_text(paper['existing_summary'], encoding='utf-8')
    (DATA / 'papers.json').write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'dataset': 'laion/Scientific-Summaries', 'revision_observed': REVISION,
                'retrieval': 'HF dataset viewer rows API; response field hashes frozen locally',
                'seed': 250219413, 'sampling': '50 arxiv + 50 bethgelab; five offset pools each; length-filtered convenience sample',
                'summary_fields': FIELDS, 'documents': [
                    {k: p[k] for k in ('document_id', 'paper_id', 'viewer_config', 'viewer_row_index',
                                      'source_title', 'fulltext_sha256', 'summary_sha256')}
                    for p in papers]}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for shard in range(3):
        (DATA / f'shard-{shard}.json').write_text(json.dumps(
            [{'document_id': p['document_id'], 'title': p['source_title'],
              'paper_file': str(DATA / 'papers' / f'{p["document_id"]}.txt')}
             for i, p in enumerate(papers) if i % 3 == shard], indent=2), encoding='utf-8')
    print('saved', len(papers), 'papers', flush=True)

if __name__ == '__main__':
    main()
