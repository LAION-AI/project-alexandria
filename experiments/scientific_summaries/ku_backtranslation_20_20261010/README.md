# Knowledge Units with guarded TranslateGemma back-translation

Completed on 10 October 2026. [Live static reader](https://projects.laion.ai/project-alexandria/ku-examples/) · [Evaluation PDF](reader/evaluation.pdf) · [Complete five-paper Gemma12 examples PDF](reader/examples.pdf).

## What this experiment measures

The same 20 disjoint papers and 200 frozen four-choice questions from the KU distillation follow-up are reused. This study applies TranslateGemma 4B English → German → English paraphrasing to copied descriptive passages in ten existing KU conditions. All eighteen raw/reference contexts and ten guarded contexts are freshly answered by the same Qwen2.5-7B-Instruct judge: **28 conditions / 5,600 primary QA slots**. No training, source regeneration, question changes or QA-guided repair selection occurs.

Each fixed Gemma rank128 KU adapter was trained for one epoch on 391 Qwen27 completions from 97 separate papers, using 1,000-word source chunks. It is applied at 500 and 1,000 words without retraining. **There is one adapter per base model, with two inference profiles.**

## QA: matched raw versus guarded back-translation

| Context generator | Chunk words | Raw correct /200 | Raw accuracy | BT correct /200 | BT accuracy | Change | Paired 95% interval | Invalid BT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| qwen | 500 | 177 | 88.50% | 180 | 90.00% | +1.50 pp | +0.00 to +3.00 pp | 0 |
| gemma4_base | 500 | 151 | 75.50% | 149 | 74.50% | -1.00 pp | -3.50 to +1.50 pp | 10 |
| gemma4_r128 | 500 | 165 | 82.50% | 165 | 82.50% | +0.00 pp | -1.50 to +1.50 pp | 10 |
| gemma12_base | 500 | 150 | 75.00% | 150 | 75.00% | +0.00 pp | -2.00 to +2.00 pp | 10 |
| gemma12_r128 | 500 | 178 | 89.00% | 177 | 88.50% | -0.50 pp | -1.50 to +0.00 pp | 0 |
| qwen | 1000 | 176 | 88.00% | 175 | 87.50% | -0.50 pp | -2.50 to +1.00 pp | 0 |
| gemma4_base | 1000 | 133 | 66.50% | 133 | 66.50% | +0.00 pp | -3.00 to +3.50 pp | 0 |
| gemma4_r128 | 1000 | 159 | 79.50% | 156 | 78.00% | -1.50 pp | -3.50 to +0.50 pp | 0 |
| gemma12_base | 1000 | 129 | 64.50% | 129 | 64.50% | +0.00 pp | -3.00 to +3.00 pp | 10 |
| gemma12_r128 | 1000 | 169 | 84.50% | 168 | 84.00% | -0.50 pp | -2.00 to +1.00 pp | 0 |

Intervals use 10,000 paired paper-bootstrap draws (seed 20261010). An interval including zero does not establish a reliable change. Fresh stochastic QA can change the reported accuracy of unchanged contexts relative to the preceding report. Failed generation slots remain wrong; no text is regenerated after viewing scores.

## All reference and summary contexts

| Condition | Correct /200 | QA accuracy | 95% interval | Invalid |
| --- | ---: | ---: | --- | ---: |
| no_context | 86 | 43.00% | 38.00–48.00% | 3 |
| original | 182 | 91.00% | 87.50–94.50% | 0 |
| qwen_summary | 162 | 81.00% | 75.50–86.50% | 0 |
| gemma4_base_summary | 128 | 64.00% | 51.50–75.00% | 20 |
| gemma4_r128_summary | 136 | 68.00% | 54.50–80.00% | 20 |
| gemma12_base_summary | 147 | 73.50% | 67.00–79.50% | 0 |
| gemma12_r128_summary | 161 | 80.50% | 74.00–86.50% | 0 |
| gemma12_previous_summary_lora | 178 | 89.00% | 83.50–93.50% | 0 |

Full original paper is the reference source context, not an assumed perfect score. The summary reference texts are unchanged and do not receive KU back-translation in this study. Questions were Qwen teacher-authored and automatically source-validated, rather than independently human-curated.

## Repair and scientific fidelity

- Selected 2,517 windows; 3,971 round-trip attempts across at most two rounds.
- Inserted **1,123** accepted windows in **190/200** document versions; 0 document-level rollbacks.
- Acceptance requires unchanged conservative number/unit/formula/operator signatures, bidirectional DeBERTa NLI entailment ≥0.9, and no source-copy run above five words inside the candidate window.
- Raw scored candidates: **315** conservative critical-value flags; **164** numeric-signature and **149** formula-signature changes. Rejected changes retain the original text.
- Retained final documents: **0** global critical-value flags. Automated signatures/NLI do not independently verify all scientific claims.
- Graph names, aliases, identifiers, predicates, relation targets, bibliographic/mathematical fields and JSON keys remain unchanged. Only context summaries and descriptive attribute string values are edited.

## Full-source copying audit

| Generator | Words | Raw mean coverage ≥6 | BT mean coverage ≥6 | BT longest run | BT ≤5 pass /20 | BT ≥7 flag /20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen | 500 | 10.97% | 7.85% | 31 | 0 | 20 |
| gemma4_base | 500 | 22.51% | 19.07% | 37 | 0 | 20 |
| gemma4_r128 | 500 | 14.80% | 11.16% | 37 | 0 | 20 |
| gemma12_base | 500 | 8.16% | 6.54% | 20 | 0 | 19 |
| gemma12_r128 | 500 | 13.47% | 10.18% | 35 | 0 | 20 |
| qwen | 1000 | 10.32% | 7.56% | 22 | 0 | 20 |
| gemma4_base | 1000 | 21.27% | 18.25% | 41 | 0 | 20 |
| gemma4_r128 | 1000 | 14.19% | 9.96% | 26 | 0 | 20 |
| gemma12_base | 1000 | 8.35% | 6.30% | 15 | 1 | 16 |
| gemma12_r128 | 1000 | 11.18% | 8.55% | 37 | 0 | 20 |

Primary definitions exactly match the preceding KU benchmark: all factual strings are audited as separate narrative fragments, including technical names, relation targets and bibliographic values embedded in attributes. Document metadata is separate. Mathematical/chemical terms are not split into artificial extra words; the punctuation-split diagnostic remains in the detailed audit. Preserved names and rejected prose can leave literal overlap. This pipeline does not make every KU pass the strict five-word rule.

## Attribution, style and context

The current sequential schema records context_summary, entities, attributes, relationships and source sentence MinHash. Up to ten prior KU summaries and entity names support naming. There is **no future-source lookahead**. The current context_summary prompt describes the target in up to three factual sentences; it does not guarantee the separate preceding-context description discussed in the [Alexandria paper](https://arxiv.org/abs/2502.19413v2).

The original document title is supplied; the document-author envelope was not populated. Author names sometimes occur inside entity attributes. There is no dedicated style field. The reader exposes these actual fields and adds source-registry title/authors for display-only attribution. Its previous/next summary panels are navigation over saved KUs, not newly generated model facts or lookahead input. No metadata enrichment changes QA contexts.

## Complete examples

Five fixed papers are shown in full source chunk order. Each has six saved model/chunk views: Gemma12 KU LoRA, Gemma E4B KU LoRA and Qwen27, each at 500 and 1,000 words. Every saved KU, entity, attribute and relationship is preserved without editorial rewriting; partial failures remain explicit. Full source texts, source-containing prompts, reasoning and MCQs are excluded from the static reader.

## Measured compute

Active TranslateGemma inference: **17.498 seconds / 0.004861 GPU-hours** for 200 cached KU document versions. The optimized implementation uses length sorting, batches up to 512, BF16, prefix caching and chunked prefill. Cold load, semantic checks, QA and idle roles are excluded from this active timer.

Whole experiment: **405 allocated seconds × four GH200 GPUs = 0.4500 reserved GPU-hours**. Startup, semantic verification, all 5,600 QA slots, audits and idle roles are included. Original KU generation and earlier adapter training are excluded. This experiment alone is not a production-throughput guarantee.

## Reproduction and artifacts

- [metrics.json](metrics.json): all scores, paired intervals, fidelity, metadata and allocation metrics.
- [protocol.json](protocol.json) and [model_pins.json](model_pins.json): frozen rules, hashes and model/adapter identities.
- [qa_predictions.jsonl.gz](qa_predictions.jsonl.gz): all 560 paper-condition records and 5,600 primary QA slots, without source prompts.
- [evaluation_questions.json](evaluation_questions.json): frozen questions, options and gold labels without source evidence quotes or rationales.
- [copy_overlap.json](copy_overlap.json), [CSV](copy_overlap.csv), [detailed audit](copy_overlap_details.jsonl.gz).
- [Bibliographic copy diagnostic](bibliographic_copy_diagnostic.json): author/title/citation attributes and exact document-title strings separated as metadata using the already recorded per-fragment matches. Historical primary totals and QA stay unchanged.
- [window_quality.jsonl.gz](window_quality.jsonl.gz): every candidate, acceptance and fidelity check; [document checks](document_quality.jsonl.gz).
- [Five-paper factual outputs](public_demo_examples.jsonl.gz), [reader inventory](reader/manifest.json), and [code](code/).
- Put local source/question arrays in inputs/, configure ALEXANDRIA_KU_DISTILL_RUN, obtain pinned models in models/ and provide cached contexts under the documented sibling run folders. Source bodies and complete source-containing traces remain in experiment storage rather than GitHub.
- Run code/ku_pipeline.py --role prepare before QA; run --role run inside a four-GPU allocation; finalize.py, build_site.py and build_pdf.py create the reports. requirements.txt lists the runtime versions. Training/inference/merge code for the fixed adapters is included in their HF repositories.
- New integration and report wrappers use CC BY 4.0. Upstream Alexandria utilities/prompts and the earlier passage/critical-value helpers retain Apache 2.0 notices. Source and base-model terms remain applicable.

## Conclusions

Guarded back-translation reduces literal copying while keeping graph structure and conservative scientific-value signatures unchanged. Gemma12 at 500 words retains high QA usefulness (88.50%) with a −0.50-point matched change. This small sample does not establish a reliable accuracy gain or loss. The strict five-word copying target remains unmet for many outputs, so the repair stage is a reduction in overlap rather than complete compliance. Metadata and sequential context are inspectable, and missing author/style/lookahead annotations are described explicitly.
