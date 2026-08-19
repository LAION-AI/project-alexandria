# Reproducing the MCQ evaluation

## What the historical "judge" was

The recovered code separates three roles that are easy to conflate:

1. **MCQ author:** The paper and the recovered dataset-build scripts identify
   `gemini-1.5-pro-002` as the generator and answer annotator for both abstract and long-paper
   questions. The historical tree also contains an older Meta-Llama 3.1 70B question helper, but it
   is not the provenance of the released canonical Parquets.
2. **KU extractor:** This is the row/model being compared in the paper tables.
3. **MCQ answering judge:** The published driver imported a fixed `extract_answers.py` whose active
   model alias was `gemini-1.5-flash-8b`, and used the historical temperature 0.5 prompt.

The long-result filenames and nearby experiment variants indicate that the alias most likely mapped
to `gemini-1.5-flash-8b-exp-0924`. This is strong code-level evidence, but the provider configuration
that resolved the alias was not preserved. Meta-Llama 3.1 8B/70B Parquets in the repository belong
to the separate corrupted-KG error-detection study.

The score is not a free-form LLM-as-a-Judge rating. The answerer receives the existing question and
one of three contexts, emits `;A;` through `;D;`, and is scored by exact match against the stored gold
letter. Invalid output counts as wrong after the historical five attempts. The recovered settings
are temperature 0.5, top-p 0.95, 100 output tokens, and frequency/presence penalties of 1.05. The
runner also reproduces the legacy ASCII sanitizer applied to the complete answer prompt (including
its accidental removal of semicolons from prompt examples). Therefore, a no-context rerun is the
calibration check for a candidate historical judge: on the full dataset it should approach the
paper's lower-bound value.

## Installation and pilot

```bash
python -m pip install -e '.[eval]'

alexandria evaluate data/evaluation/long/physics_qa_pairs_expanded.parquet \
  --output outputs/qwen38-physics-pilot.json \
  --limit 10 --shuffle --seed 250219413 \
  --model qwen38 --base-url http://127.0.0.1:8010/v1 \
  --judge-model gemini-1.5-flash-8b \
  --judge-base-url https://YOUR-COMPATIBLE-ENDPOINT/v1 \
  --chunk-words 500 --context-words 1000 \
  --document-batch-size 8 --concurrency 8
```

Set `ALEXANDRIA_JUDGE_API_KEY` in the environment for a protected judge endpoint. Never put it in
the command, an output artifact, or version control. The extractor and judge can use different
OpenAI-compatible endpoints. Checkpoints are atomic after each document batch; rerunning the same
command skips completed document IDs.

For a diagnostic self-judge run, set `--judge-model qwen38` and use the local base URL. Those scores
measure the full Qwen system but are **not** directly comparable to the historical fixed-judge table.

## Conditions and denominators

- `no_context`: the question plus an explicit empty-context marker;
- `original`: the question plus the original abstract or 30,000-character paper text;
- `knowledge_units`: the question plus compact KU summaries, entities, attributes, and relations.

The output contains no source text or question text: only dataset digest, document hash, source row
index, question index, gold/predicted letters, configuration, counts, scores, and elapsed time. An
invalid answer remains null and counts as incorrect. With 30 abstract-pilot questions, one answer is
3.33 percentage points; a ten-paper long pilot has 100 questions, so one answer is one point.
