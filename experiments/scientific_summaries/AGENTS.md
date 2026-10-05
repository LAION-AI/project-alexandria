# Scientific summary evaluation requirements

The user requires a source-copy n-gram audit in every future evaluation.

- Check every generated summary against its own full, untruncated paper source. Use `ngram_overlap.py` and `copy_overlap_eval.py`; record the audit version and source hash.
- Report contiguous overlaps: at most five normalized words allowed, exactly six borderline, seven or more flagged. Report longest runs and covered-word fractions, including cumulative copying from multiple passages.
- The primary word definition is whitespace-delimited, matching the existing five-word quote rule. Preserve the punctuation-split diagnostic separately; do not count components of one formula/chemical term as several natural words in the primary decision.
- Audit narrative prose and explicit evidence quotes separately. Include titles, authors and bibliographic metadata as a separate diagnostic so their literal spelling is visible.
- Do not include hidden/returned reasoning, prompts, MCQs, reviewer comments or gold answers as summary prose.
- Do not drop papers, alter MCQs/gold, or regenerate a summary after inspecting QA scores to hide overlap. Preserve QA scores for all original evaluation slots; show overlap flags alongside them.
- Write `copy_overlap.json`, per-summary CSV/JSONL evidence and an English overlap section in the evaluation results. Include flags for failed/unparseable outputs rather than silently omitting them.
- Run large historical audits on a compute allocation. Preserve raw artifacts and published dataset snapshots.
