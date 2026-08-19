# Evaluation datasets

These are the canonical Parquet artifacts recovered from the Project Alexandria experiment
workspace. The project owner confirmed that they may be redistributed. They are included so that
published and future model baselines can use exactly the same source texts, questions, options,
and gold letters rather than regenerated approximations.

## Abstract MCQs

| File | Domain | Documents | Valid document shape |
|---|---|---:|---|
| `abstract/alexandria_biology_q_a_dataset.parquet` | Biology | 2,300 | normally 3 MCQs per abstract |
| `abstract/alexandria_cs_q_a_dataset.parquet` | Computer science | 2,300 | normally 3 MCQs per abstract |
| `abstract/alexandria_math_q_a_dataset.parquet` | Mathematics | 3,141 | normally 3 MCQs per abstract |
| `abstract/alexandria_physics_q_a_dataset.parquet` | Physics | 2,300 | normally 3 MCQs per abstract |

Schema: `text: string`, `question: list<string>`, `answer: list<string>`. Empty generations and
malformed non-A/B/C/D answers are retained in the source artifact but skipped by the evaluation
loader. The paper and the recovered `makeqa.py` agree that these MCQs and answer annotations were
generated with `gemini-1.5-pro-002`. A separate older helper using Meta-Llama 3.1 70B exists in the
historical tree but is not the provenance of these canonical Parquets.

## Long-document MCQs

| File | Domain | Expanded rows | Unique papers | Valid MCQs |
|---|---|---:|---:|---:|
| `long/physics_qa_pairs_expanded.parquet` | Physics | 973 | **100** | 970 |
| `long/medical_qa_pairs_expanded.parquet` | Medical | 991 | **100** | 988 |

Each source was truncated to 30,000 characters. The historical `makeqalong.py` used
`gemini-1.5-pro-002` to request ten MCQs per paper. Expansion produced one row per question. Three
Physics and one Medical records contain only an incomplete one-row group; the loader groups rows by
the SHA-256 of `text` and validates each gold letter.

## Error-detection judge study

The four files under `judge-error-detection/` are a different experiment: Mixtral 8x22B KGs were
artificially corrupted and evaluated by Meta-Llama 3.1 8B or 70B. They do **not** identify the
answering model used for the paper's MCQ accuracy tables.

## Integrity and licensing

Exact digests are in [`SHA256SUMS`](SHA256SUMS). Dataset redistribution was authorized by the
project owner on 2026-08-19. This repository's Apache-2.0 software license does not override any
attribution or source-specific obligations associated with the underlying articles.
