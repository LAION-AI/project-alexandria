"""Apply an exported frozen CPU rule bank to one structured summary."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from lexicon import repair, immutable
from ngram_overlap import SourceIndex, fragments, audit_summary
from passages import patch
from reference_validator import narrative_context, validate_summary
from critical_values import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()
    saved = json.loads(args.summary.read_text())
    original = saved.get('summary', saved)
    source = args.source.read_text()
    rules = json.loads(args.rules.read_text())['rules']
    index = SourceIndex(source)
    before = narrative_context(original)
    replacements = {}
    traces = []
    for text, path, category in fragments(original):
        if category == 'narrative':
            value, edits = repair(text, index, rules)
            if value != text:
                replacements[path, text] = value
                traces.append(dict(path=path, before=text, after=value, edits=edits))
    summary = patch(original, replacements)
    context = narrative_context(summary)
    protected = lambda s: [(t, p, c) for t, p, c in fragments(s) if c != 'narrative']
    rolled_back = False
    error = None
    try:
        assert immutable(before, context), 'Immutable scientific values changed'
        assert check(json.dumps(original, ensure_ascii=False), json.dumps(summary, ensure_ascii=False))['passed']
        assert protected(original) == protected(summary)
        try:
            validate_summary(json.dumps(original, ensure_ascii=False), source)
            valid = True
        except (ValueError, TypeError, KeyError):
            valid = False
        if valid:
            validate_summary(json.dumps(summary, ensure_ascii=False), source)
        audit = audit_summary(source, summary, index=index, expanded_diagnostic=False)
        original_audit = audit_summary(source, original, index=index, expanded_diagnostic=False)
        assert audit['categories']['narrative']['longest_contiguous_match_words'] <= original_audit['categories']['narrative']['longest_contiguous_match_words']
    except (AssertionError, ValueError, TypeError, KeyError) as exc:
        summary = original
        context = before
        rolled_back = True
        error = repr(exc)
        audit = audit_summary(source, original, index=index, expanded_diagnostic=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(summary=summary, judge_context=context, audit=audit,
        source_sha256=index.sha256, rolled_back=rolled_back, error=error, traces=traces,
        cpu_seconds=time.process_time()-started), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
