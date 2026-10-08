# Qwen 27B Knowledge Units: 1,000-word chunks on 97 papers

Completed fresh evaluation on all 97 frozen papers / 970 MCQs. The extractor uses the repository's sequential few-shot KU prompt, the preceding ten generated KUs as naming context, and sentence-aware targets of at most 1,000 words. No source opening abstract, neighbor window, semantic correction or document-wide alias pass is added. This is a larger-chunk extension of the paper's 200-word full-paper protocol.

The fixed Qwen2.5-7B-Instruct answerer uses the unchanged historical ASCII-sanitized prompt/parser, temperature 0.5, top-p 0.95, 100 output tokens, frequency/presence penalties 1.05 and at most four invalid-answer retries. Every condition was freshly judged. No questions or gold labels are supplied to extraction; no paper is dropped or regenerated after QA.

| Context | Correct / 970 | QA accuracy | 95% paper-bootstrap interval | Invalid / failed slots |
| --- | ---: | ---: | ---: | ---: |
| no_context | 603 | 62.16% | 58.45–65.77% | 4 |
| original | 941 | 97.01% | 95.88–98.04% | 0 |
| knowledge_units_1000 | 879 | 90.62% | 88.66–92.47% | 0 |
| gemma_r128_fp8_summary | 912 | 94.02% | 92.27–95.67% | 0 |

## Paired differences

- knowledge_units_1000_minus_no_context: +28.45 percentage points; 95% paper-bootstrap interval [+24.95, +32.06].
- knowledge_units_1000_minus_original: -6.39 percentage points; 95% paper-bootstrap interval [-8.04, -4.85].
- knowledge_units_1000_minus_gemma_r128_fp8_summary: -3.40 percentage points; 95% paper-bootstrap interval [-5.26, -1.55].

Intervals resample 97 paper clusters with all ten MCQs retained. The fixed stochastic judge can add sampling variation; QA measures answerability under this judge, not independent human factuality.

## Extraction and compute

- 391/391 chunks generated; 0 failed papers. Failed KU papers retain ten wrong slots.
- 341,673 source words; 512,647 extractor completion tokens, including bounded format retries.
- Three independent single-GH200 extractor replicas and one single-GH200 judge share an exclusive Booster node. Documents are concurrent; each document's chunks remain sequential.
- Per-replica timing, raw usage, server startup and total allocation accounting are retained separately. Concurrent request latency sums are not GPU wall time.

## Source-copy audit

Full untruncated-source audit 2.1: 0/97 narrative outputs meet the five-word limit; 1 have only a six-word maximum; 96 contain a run of seven or more words. Maximum run: 24 words. Mean coverage in runs ≥6: 9.56%.

Factual strings are checked individually; separate graph fields are never concatenated to manufacture an overlap. Entity names and attribute/relation labels remain visible in the factual diagnostic. Document titles are bibliographic metadata. Fingerprints, source offsets, entity IDs, prompts, reasoning, questions and gold labels are excluded from KU prose. No explicit evidence quotes are emitted by this KU schema.

Five-, seven- and eleven-word Jaccard means: 5: 0.013999, 7: 0.004739, 11: 0.000674. These fragment-based diagnostics are not automatically identical to the historical paper's serialization metric. Lexical overlap alone does not establish plagiarism.

Detailed evidence: `report.json`, `qa-results.json.gz`, `copy_overlap.json`, `copy_overlap.csv`, `copy_overlap_details.jsonl.gz`, and generated KU artifacts. Raw sources and source-containing request traces remain in Scratch/durable experiment storage, outside GitHub.

## Measured performance and allocation accounting

| Measurement | Value | Scope |
| --- | ---: | --- |
| KU extraction wall time | 234.38 s (3.91 min) | Slowest of three concurrent extractor replicas; initialization excluded |
| Active extractor GPU-hours | 0.1906 | Sum of the three replica phase wall intervals |
| Mean completion tokens / paper | 5,285.0 | All KU JSON completions, not prose-only summary tokens |
| Mean completion tokens / chunk | 1,311.1 | 391 successful chunks; no format retries |
| Completion tokens / active GPU second | 747.1 | End-to-end extraction phase; includes prefill, tokenize, decode and checkpoint overhead |
| Aggregate completion tokens / wall second | 2,187.3 | Total completions divided by extraction makespan across three GPUs |
| QA phase wall time | 235.84 s | Includes waiting for complete KU papers; four fresh 970-question conditions |
| Sum of QA paper scoring intervals | 131.29 s | Judge work across 97 papers; excludes waiting for extraction |
| Successful allocation | 1.2067 GPU-hours | 18 min 06 s × four reserved GH200 GPUs, including cold startup |
| Requeued infrastructure allocation | 0.0956 GPU-hours | 86 s × four GPUs; first node failed the GPFS healthcheck before inference |
| Total including infrastructure failure | 1.3022 GPU-hours | Top-level Slurm records only; no double counting of batch/steps |

Generated graph content: 3,428 entity records, 8,686 top-level entity attribute entries and 5,344 relationship records. These are local records, not counts of globally deduplicated scientific facts. Thinking/reasoning tokens: zero.

Fresh QA reuses the frozen direct Gemma FP8 summary texts. Their 912/970 score here differs from the previous 914/970 evaluation because the historical judge is stochastic and all conditions were rerun; the summaries themselves were not regenerated.

Runtime: CPython 3.11 on aarch64, vLLM 0.30.0, PyTorch 2.13.0, Transformers 5.18.0; four GH200 120GB GPUs. FP8 weights, BF16 activations/KV, prefix caching, chunked prefill, 16 concurrent extractor documents per replica and four judge requests. Complete server arguments and code hashes are retained.

## Conclusions

- The 1000-word KU extension improves QA over no context by 28.45 percentage points, but scores 3.40 points below the direct Gemma rank-128 FP8 summaries in the same fresh evaluation.
- The paper-cluster bootstrap interval for the KU-minus-summary gap is [-5.26,-1.55] percentage points; this interval does not include zero. This is one stochastic judge run, not independent human factuality.
- The measured active extraction is fast with three batched single-GH200 replicas, but startup and the failed first prolog dominate this small benchmark allocation.
- The KU outputs do not meet the requested five-word limit: 0/97 pass, one is six-word borderline, and 96 contain at least seven consecutive source words. Graph structure does not remove the need for overlap control.
- These results evaluate a larger-chunk extension of the 200-word Alexandria setup; they do not isolate chunk size as the cause of the QA gap. No extraction was changed after QA.
