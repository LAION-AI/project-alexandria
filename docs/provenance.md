# Historical-code provenance

This repository is a consolidation, not a claim that the historical workspace was tidy or that
every file contributed to the paper. The source audit used three evidence layers:

1. the public `christophschuhmann/Alexandria` repository at commit `b6478e0`;
2. the later local experiment workspace (files dated through June 2025);
3. arXiv v2 and the supplied 16-page unreleased PDF.

## Implementation lineage

| Historical file | Role | Consolidated location |
|---|---|---|
| `documents_to_kgs.py` | Standalone long-document sequential extraction | `pipeline.py`, sequential mode |
| `gen_eval_long+.py` | Latest combined long-document extraction/reconstruction/evaluation | pipeline + `experiments/` |
| `prompts.py` | Few-shot KU, reconstruction, MCQ, and context-summary prompts | `prompts.py`, `experiments/mcq.py` |
| `llm.py` | HyperLab/Together HTTP client | provider-neutral `backends.py` |
| `string_into_chunks.py` and duplicated splitters | Text segmentation | deterministic `chunking.py` |
| `minhash_vector.py` | 16-permutation token MinHash | `fingerprints.py` |
| `extract_answers.py`, `multiple_choice_question.py` | MCQ generation and cloze scoring | `experiments/mcq.py` |
| `cossim/cossims.py` | Embedding controls and scrambling | `experiments/similarity.py` |
| `cossim/sherlock/sherlpy3.py` | Sherlock and 5/7/11-gram overlap | `experiments/overlap.py` |
| `makeqa.py` | Abstract MCQ dataset assembly | released Abstract Parquets |
| `makeqalong.py`, `expand.py` | Gemini 1.5 Pro 002 long MCQs and row expansion | released Long Parquets |
| `extract_answers.py`, `evaluate.py` | Fixed answerer and exact-letter scoring | `experiments/reproduce.py` |

The original prompt’s core constraints and cross-domain few-shot strategy are retained, but its
very large repeated template was edited into strict JSON examples. The historical code used
Python-literal-like output and `<kg>` tags; accepting that legacy shape is supported by the parser,
while newly generated artifacts use schema version 1.0 JSON.

## Intentional changes

- Removed HyperLab-specific URLs and all embedded credentials.
- Removed ASCII-only sanitization, which damaged names and mathematical notation.
- Removed import-time downloads, global configuration, and execution on module import.
- Replaced random chunk lengths with deterministic sentence-aware boundaries.
- Separated extraction from reconstruction and evaluation.
- Added atomic output writes, retries, validation, explicit configuration, and tests.
- Added a truly independent parallel mode and a second-pass entity resolver.
- Kept source text out of published output; only offsets and fingerprints are stored.

## Security note

The historical worktree contains a hard-coded HyperLab credential and a second local API-key file.
They are not included here and should be considered compromised and revoked. Only the ten canonical
evaluation Parquets authorized by the project owner were copied. Derived CSVs, duplicate source
extracts, raw model outputs, and files containing credentials remain excluded.

## Results provenance

- `docs/results/paper.md`: arXiv v2 Tables 2–5.
- `docs/results/unpublished-report.md`: tables and prose in the supplied unreleased PDF.
- `docs/results/embedding-similarity.md`: PDF Table 13 plus a clearly labeled later aggregate from
  `cossim/all_embedding_stats.csv` (74,411 rows).
- `results/*.csv`: machine-readable transcriptions, not regenerated measurements.
- `data/evaluation/`: canonical abstract, long-paper, and error-detection Parquets with SHA-256s.
