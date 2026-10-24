"""Mine all 865 released training papers, with fixed document-level derivation/dev split."""
import concurrent.futures
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import time
from common import ROOT, load, read_jsonl, write, jsonl, digest
from ngram_overlap import SourceIndex, fragments
from passages import sentence_spans
from lexicon import options

TRAIN = Path('/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004/release/data/train.jsonl')
OLD = Path('/e/scratch/reformo/schuhmann1/scientific-hybrid-qwen08-gemma-e2b-97-20261023')


def init():
    global editor, procedural, rules
    import procedural as module
    procedural = module
    editor = procedural.LexicalRepair()
    rules = load(ROOT/'inputs/candidate_rules.json')['rules']


def process(row):
    index = SourceIndex(row['fulltext'])
    summary = json.loads(row['validated_final_summary_json'])
    doc = row['document_id']
    split = 'validation' if int(digest(doc)[:8],16) % 5 == 0 else 'derivation'
    cards = []
    copied_windows = []
    for text, path, category in fragments(summary):
        if category != 'narrative':
            continue
        for a,b in sentence_spans(text):
            sentence = text[a:b].strip()
            before = index.fragment(sentence, path, category)
            covered = before['covered_words_by_minimum_run']['6']
            if not covered:
                continue
            copied_windows.append((covered, sentence, path))
            for candidate in options(sentence, index, rules):
                cards.append(dict(candidate, document_id=doc, domain=row['domain'], split=split,
                                  path=path, kind='candidate_rule'))
    diagnostic = []
    saved_protected = procedural.PROTECTED
    saved_block = procedural.BLOCK_LEXNAMES
    saved_collocations = editor.collocations
    for mode in ['existing_guarded_dictionary', 'naive_dictionary_diagnostic']:
        if mode == 'naive_dictionary_diagnostic':
            procedural.PROTECTED = set()
            procedural.BLOCK_LEXNAMES = set()
            editor.collocations = []
        editor.lookup.cache_clear()
        for covered, text, path in sorted(copied_windows, key=lambda r:-r[0])[:3]:
            after, trace = editor.repair_fragment(text, index, 'synonyms_broad')
            if after != text:
                diagnostic.append(dict(document_id=doc, domain=row['domain'], split=split, path=path,
                    before=text, after=after, trace=trace, kind=mode, rule=mode,
                    pair_id=digest(text+'\0'+after)))
                break
        procedural.PROTECTED = saved_protected
        procedural.BLOCK_LEXNAMES = saved_block
        editor.collocations = saved_collocations
    return dict(document_id=doc, source_sha256=index.sha256, domain=row['domain'], split=split,
                candidates=cards, diagnostics=diagnostic)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    if (ROOT/'outputs/mining_complete.json').exists():
        return
    started = time.monotonic()
    cohort = read_jsonl(OLD/'inputs/cohort97.jsonl')
    test_ids = {r['document_id'] for r in cohort}
    test_hashes = {r['source_sha256'] for r in cohort}
    cards = defaultdict(list)
    diagnostic = defaultdict(list)
    inventory = []
    def rows():
        with TRAIN.open() as stream:
            for line in stream:
                row = json.loads(line)
                assert row['document_id'] not in test_ids
                yield row
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=init) as pool:
        for result in pool.map(process, rows(), chunksize=2):
            assert result['source_sha256'] not in test_hashes, 'Training/test source collision'
            inventory.append({k:v for k,v in result.items() if k not in ['candidates','diagnostics']})
            if len(inventory)%50==0:print('MINING_PAPERS',len(inventory),flush=True)
            for card in result['candidates']:
                cards[(card['rule'],card['split'])].append(card)
            for card in result['diagnostics']:
                diagnostic[card['kind']].append(card)
    selected = []
    for (rule, split), values in sorted(cards.items()):
        # At most one exemplar per paper/rule, then deterministic hash sampling.
        unique = {}
        for card in sorted(values,key=lambda c:c['pair_id']):
            unique.setdefault(card['document_id'],card)
        limit = 16 if split == 'derivation' else 8
        selected.extend(sorted(unique.values(),key=lambda c:c['pair_id'])[:limit])
    for kind, values in diagnostic.items():
        selected.extend(sorted(values,key=lambda c:c['pair_id'])[:150])
    jsonl(ROOT/'inputs/training_review.jsonl',selected)
    jsonl(ROOT/'inputs/training_candidates_all.jsonl',[c for values in cards.values() for c in values])
    jsonl(ROOT/'inputs/training_inventory.jsonl',inventory)
    write(ROOT/'outputs/mining_complete.json',dict(complete=True, training_papers=len(inventory),
        candidates=sum(len(v) for v in cards.values()), review_cards=len(selected),
        diagnostic_counts={k:len(v) for k,v in diagnostic.items()}, seconds=time.monotonic()-started,
        no_test_ids_or_hashes=True, split_counts=dict(Counter(r['split'] for r in inventory)),
        split='Document SHA256 modulo5: 20% validation; all865 released records used, no QA gold.'))
    print('MINING_COMPLETE',len(inventory),len(selected),flush=True)


if __name__ == '__main__':
    main()
