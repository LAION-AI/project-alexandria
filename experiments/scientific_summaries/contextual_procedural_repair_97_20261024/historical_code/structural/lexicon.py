"""Finite context-sensitive phrase changes; no unrestricted dictionary editing."""
import re
from collections import Counter
from common import named_values, negations, digest
from critical_values import check, LATEX
from ngram_overlap import tokenize, merged_length

MODALS = re.compile(r'\b(?:may|might|can|cannot|could|must|should|shall|will|would)\b', re.I)
SENSE_WORDS = set('significant significance statistically statistical insignificant nonsignificant positive negative sensitivity specificity precision accuracy recall bias variance error risk odds ratio correlation causal causality association null hypothesis hypotheses energy power force mass concentration expression activity fixed random randomized randomised independent dependent uncertainty uncertain possible probable robust stronger weaker increased decreased'.split())


def state(text, index):
    words = tokenize(text)[1]
    grams = index.grams(6)
    hits = [i for i in range(max(0, len(words)-5)) if tuple(words[i:i+6]) in grams]
    return words, hits, merged_length((i, i+6) for i in hits)


def immutable(before, after):
    return (check(before, after)['passed'] and named_values(before) == named_values(after)
            and negations(before) == negations(after)
            and Counter(w.lower() for w in MODALS.findall(before)) == Counter(w.lower() for w in MODALS.findall(after))
            and Counter(w for w in tokenize(before)[1] if w in SENSE_WORDS) == Counter(w for w in tokenize(after)[1] if w in SENSE_WORDS))


def options(text, index, rules):
    words, hits, covered = state(text, index)
    if not hits:
        return []
    raw = []
    for m in re.finditer(r'\S+', text):
        values = tokenize(m.group())[1]
        if len(values) > 1:
            return []
        if values:
            raw.append((m.start(), m.end(), values[0]))
    if [r[2] for r in raw] != words:
        return []
    matched = [(raw[i][0], raw[i+5][1]) for i in hits]
    protected = [m.span() for m in LATEX.finditer(text)]
    protected += [m.span() for m in re.finditer(r'`[^`]*`|"[^"\n]*"|“[^”\n]*”', text)]
    result = []
    for priority, rule in enumerate(rules):
        for match in re.finditer(rule['pattern'], text, re.I):
            a, b = match.span()
            if not any(a < d and b > c for c, d in matched):
                continue
            if any(a < d and b > c for c, d in protected):
                continue
            # Context is inspected, never replaced. Blocks learned from training
            # are sentence/window patterns; they cannot consume or delete text.
            context=text
            if 'three tokens' in rule.get('blocked_context_scope',''):
                left=[r for r in raw if r[1]<=a];right=[r for r in raw if r[0]>=b]
                context=text[left[-3][0] if len(left)>=3 else 0:right[2][1] if len(right)>=3 else len(text)]
            if any(re.search(p, context, re.I) for p in rule.get('blocked_context_regex', [])):
                continue
            if any(re.search(p,text,re.I) for p in rule.get('training_refinement_blocked_context_regex', [])):
                continue
            required = rule.get('required_context_regex')
            if required and not re.search(required, text[b:], re.I):
                continue
            replacement = match.expand(rule['replacement'])
            if match.group()[0].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            after = text[:a] + replacement + text[b:]
            if not immutable(text, after):
                continue
            if tokenize(text, expanded=True)[1] == tokenize(after, expanded=True)[1]:
                continue
            _, new_hits, new_covered = state(after, index)
            if len(new_hits) >= len(hits) or new_covered >= covered:
                continue
            result.append(dict(rule=rule['id'], before=text, after=after, start=a, end=b,
                               old=match.group(), new=replacement, gain=len(hits)-len(new_hits),
                               covered_before=covered, covered_after=new_covered, priority=priority,
                               pair_id=digest(text+'\0'+after)))
    return result


def repair(text, index, rules):
    original = text
    edits = []
    for _ in range(64):
        candidates = options(text, index, rules)
        if not candidates:
            break
        candidate = min(candidates, key=lambda c: (-c['gain'], c['covered_after'], c['priority'], c['start']))
        if not immutable(original, candidate['after']):
            break
        edits.append(candidate)
        text = candidate['after']
    return text, edits
