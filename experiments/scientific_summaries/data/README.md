# Scientific-Summaries test set: 97 papers, 970 questions

This is the **exact frozen test set used for the summary comparison**, not the
older Alexandria paper's separate long-context benchmark. It includes 50 arXiv
and 47 Bethgelab records from [laion/Scientific-Summaries](https://huggingface.co/datasets/laion/Scientific-Summaries),
the source texts available in that dataset, existing summaries, and ten difficult
four-choice questions per paper. The questions were authored by `gpt-6-luna`
using the source text, not the tested summaries. They are synthetic, not a
human-validated benchmark. Upstream declares CC-BY-4.0; source identifiers and
attribution metadata are preserved. See the [Alexandria paper](https://arxiv.org/html/2502.19413v2).

## Files to use

| File | Purpose |
| --- | --- |
| [testset.json](testset.json) | Ready-to-use bundle: the exact 97 texts, existing summaries and 970 evaluated MCQs, with gold keys, rationales and source evidence. |
| [testset_manifest.json](testset_manifest.json) | Ordered IDs, 50/47 subset sizes, three excluded IDs, and SHA-256 hashes of the source/QA inputs. |
| [papers.json](papers.json) | Original 100 sampled records, including all upstream fields. **Do not score all 100** as the reported 97-paper cohort. |
| [qa/](qa/) | Original authored questions. Their option order is **not** the evaluated order. |
| [manifest.json](manifest.json) | Retrieval provenance and convenience-sampling procedure. |

The bundle stores the already balanced/shuffled **evaluated** options and gold
keys. Use these directly, without shuffling again. `original_paper_index` refers
to the original 100-record order; reindexing the reduced 97-record set would change
the option permutation if you reconstructed it from raw authored QA. Each question
has a ready-to-use `formatted_question`. Three excluded records are
`bethgelab-100035`, `bethgelab-200029`, and `bethgelab-200013` (no ready QA/KUs).

The dataset's available text was not independently verified as a complete
publisher full text. Some records lack bibliographies; never silently fetch missing
sections, truncate text, replace papers, or drop hard questions while claiming the
same benchmark. This is a length-filtered convenience sample, not a random sample
of all scientific papers and not a held-out general-purpose performance guarantee.

## Quick start (no GPU needed)

```bash
git clone https://github.com/LAION-AI/project-alexandria.git
cd project-alexandria
python experiments/scientific_summaries/export_testset.py --check
python - <<'PY'
import json
from pathlib import Path
test = json.loads(Path('experiments/scientific_summaries/data/testset.json').read_text())
assert len(test['papers']) == 97
paper = test['papers'][0]
print(paper['document_id'])
print(paper['questions'][0]['formatted_question'])
# Pass ONLY fulltext to a summary generator. Do not expose the QA or gold keys.
# For a student, pass ONLY the chosen context and formatted_question.
PY
```

To rebuild the bundle from the original frozen files, run the same exporter without
`--check`. It verifies each source/summary hash and exact question equality against
the completed Qwen9B evaluation. It does not download data or call a model.

## Evaluation protocol

Generate a summary from `fulltext` **without seeing questions, gold answers,
rationales or evidence annotations**. Then give the fixed student that summary
plus `formatted_question`. For controls, use `fulltext`, `existing_summary`, or
no context. `answer`, `rationale`, `evidence_quote`, and evidence offsets are for
offline scoring/audit only: never put them in the student's prompt. Do not use this
released test set for training and describe the resulting score as held out.

For direct comparison with the published scores, use the repository's
`historical_answer_prompt` and `JUDGE_SYSTEM_PROMPT` (see
[evaluate.py](../evaluate.py) and [the full prompt appendix](../summary_pipeline.html)).
The student is **Qwen/Qwen2.5-7B-Instruct**, BF16, temperature 0.5, top-p 0.95,
100 output tokens, frequency/presence penalties 1.05, no thinking, four concurrent
requests, and the historical ASCII sanitizer/case-sensitive semicolon parser.
Allow up to five formatting attempts; count a remaining invalid answer as wrong.
Accuracy is correct/970. Confidence intervals use 10,000 document-cluster bootstrap
draws (all ten questions from a paper stay together), seed 250219413. Compare
conditions using paired document-cluster bootstrap, not independent question draws.
Sampling is not deterministic despite frozen inputs; record runtime/weight hashes.

The ready-to-run evaluator expects a summary cache with `documents` containing
`document_id`, `fulltext_sha256`, and the exact student input as `judge_context`.
Its normal summary generator also saves full JSON, proof spans and raw calls.
Keep the same text hash and all 97 IDs. Use the supplied KU cache to retain the
existing KU control. On a machine with vLLM and BF16 student weights installed:

```bash
# Terminal 1: adjust Python/weights paths for your installation; no API key needed.
export ALEXANDRIA_JUDGE_PYTHON=/path/to/vllm/environment/bin/python
export ALEXANDRIA_JUDGE_WEIGHTS=/path/to/Qwen2.5-7B-Instruct
export ALEXANDRIA_JUDGE_GPU=0
export ALEXANDRIA_JUDGE_MAX_LEN=32768
export ALEXANDRIA_JUDGE_GPU_UTIL=0.90
bash experiments/scientific_summaries/serve_judge.sh

# Terminal 2, repository root. Existing controls are reused after prompt-hash checks.
python experiments/scientific_summaries/evaluate.py \
  --with-kus --with-qwen-summaries --context-limit 32768 \
  --summary-cache /path/to/your/summaries.json \
  --output /path/to/new-run/results.json \
  --baseline-checkpoint experiments/scientific_summaries/pre_qwen_summary_results.json \
  --skip-id bethgelab-100035 --skip-id bethgelab-200029 --skip-id bethgelab-200013
```

Do not reuse a baseline with different texts/questions/model/decoding settings.
Official completed model artifacts are under [summary_runs/](../summary_runs/),
including generator policy, exact student contexts, evaluated QA, raw student
responses and scores. The [comparison report](../summary_comparison.html) audits
the fixed cohort, evidence spans and unchanged control prompts before reporting
official results. Generation/repair policies differ between models; consult the
methods and run configs rather than interpreting this as a weight-only ablation.
