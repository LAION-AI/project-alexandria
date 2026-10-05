# Five scientific summaries: HTTPS reading page

Live page: [Five papers after backtranslation](https://projects.laion.ai/project-alexandria/summary-examples/).

This standalone static page displays five **complete saved summaries** from the Qwen-distilled Gemma 4 12B IT rank-128 adapter, merged FP8, with thinking disabled. They received the existing single guarded TranslateGemma 4B English → German → English repair round. No new model calls, rewriting or evaluation were used to make this reader.

| Subject | Frozen document ID |
| --- | --- |
| Astronomy | `arxiv-121` |
| Computer science | `arxiv-500047` |
| Chemistry | `bethgelab-125` |
| Structural biology | `bethgelab-100015` |
| Mathematics | `arxiv-149` |

These IDs were chosen for subject diversity before inspecting individual QA scores. They are examples for reading, not a representative quality estimate.

All nineteen saved summary fields are displayed, including empty-section markers. Narrative statements and their saved source excerpts are presented separately; excerpts can be expanded. All original numerical notation is HTML-escaped and preserved. Hidden reasoning, MCQs, gold answers and full paper sources are not included in this page. `examples.json` preserves each original post-repair summary object unchanged and adds provenance and previously computed audit results.

The numerical/formula guard passed for these saved guarded outputs. Residual narrative copy runs still exceed five words in every example. Automated guards and NLI do not independently certify all scientific claims; the reader retains the actual outputs, including any errors.

## Rebuild and publication

From the repository root:

```bash
python3 experiments/scientific_summaries/backtranslation_200_20261005/code/build_reader.py
```

`manifest.json` records the input snapshot, source-file hashes and generated artifact hashes. The rendered HTML and JSON are self-contained and need no external JavaScript, font, inference service or build framework.

Repository GitHub Pages already publishes the `master` branch root. Only this reader is copied to `summary-examples/` on that branch; existing landing-page files remain intact. The HTTPS domain `projects.laion.ai` is the existing Pages domain. The experiment code, generated reader and documentation are also versioned on the findings branch.

Benchmark details: [FINDINGS.md](../FINDINGS.md), [RESULTS.md](../RESULTS.md).
