# Qwen3.8-27B → Gemma 4 12B IT: 865-paper generator distillation

## Frozen collection

The Qwen teacher collection was stopped at the user request on 2026-10-04.
The frozen cohort contains **865 accepted papers across ten scientific domains**.
An accepted result committed during cancellation was retained before freezing.
All source-paper identities and source/final hashes are recorded in `inputs/frozen_cohort.json`.
The external 97-paper / 970-MCQ test set is excluded by identifier, DOI, title,
source hashes and substantial shared text. The overlap audit reports zero matches.

The release is in `release/`: exact prompts, pinned checkpoints and runtime settings,
full source text, original generator summaries and actual emitted reasoning,
corrected accepted summaries, source-based quality assessments, and all preserved
raw calls/journals for each accepted paper in 865 compressed archives.
The release totals 4,299,248,730 bytes before upload metadata, with SHA256 checksums.
The dataset is **published on Hugging Face**:

[ChristophSchuhmann/scientific-summary-distillation-Qwen3.8-27B-865-20261004](https://huggingface.co/datasets/ChristophSchuhmann/scientific-summary-distillation-Qwen3.8-27B-865-20261004)

Pinned published revision: `9f572fb367865b7eaf7dca37b5cb269558b84026`.
All 930 release files were verified by remote presence and byte size;
the remote checksum index, dataset card, manifest and overlap audit match the local release exactly.
The first upload attempt on a compute node could not reach the external network.
The upload was completed from login host `jpbl-s03-01` using the standard HF credential
cache outside the dataset and repository. Credentials are excluded from release artifacts.

## Exact generator target and reasoning provenance

860 accepted final summaries came from correction calls that saw a draft and review
feedback. Only five final outputs retain a source-only generator trace.
The two requested generator LoRAs therefore train on **865 original source-only
generator outputs plus their matching actual emitted reasoning**, one per accepted paper.
They do not pair a corrected answer with mismatched raw reasoning or remove the draft
and feedback that conditioned a correction trace. Raw generator targets may contain
errors repaired later; final acceptance scores apply to corrected final summaries.
The corrected outputs and conditioned correction/review reasoning remain separate
published views. No reviewer or corrector is trained.

Teacher: `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
Native autoregressive decoding, no DFlash, full source text, thinking enabled,
medium generation/correction effort and low review effort. Generation temperature 1.0,
top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1.0;
65,536-token context and up to 24,576 output tokens including reasoning.
Original prompts and per-call payloads are authoritative.

## Parallel GPU runs

| Task | Job | Hardware | Configuration |
| --- | --- | --- | --- |
| Gemma generator LoRA rank 64 | 2171847 | 2 nodes / 8 GH200 GPUs | 1 epoch, alpha 128, dropout 0.05 |
| Gemma generator LoRA rank 128 | 2171848 | 2 nodes / 8 GH200 GPUs | 1 epoch, alpha 256, dropout 0.05 |
| Ornith same-paper generation | 2171849 | 2 nodes / 8 GH200 GPUs | Native AR BF16, no DFlash, 16 concurrent papers/GPU |
| Matched external evaluation | 2171850 | 1 node / 4 GH200 GPUs | Runs automatically after both LoRAs |

Training base: `google/gemma-4-12B-it`, pinned revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
Classic LoRA trains text-layer q/k/v/o attention and gate/up/down MLP projections;
vision/audio components are unused. Assistant-only loss supervises actual teacher
reasoning and the matching answer in Gemma's native thought-channel template.
AdamW, peak learning rate **2e-5**, weight decay 0.01, six warmup updates followed by
cosine decay, effective batch eight, 109 updates, one deterministic epoch.
BF16 base, gradient checkpointing, flex attention and cut cross entropy.
No source or target truncation: 10,430–39,172 native tokens/example,
mean 22,357.38; 19,339,133 tokens total, 11,007,336 supervised tokens.
Training file SHA256:
`3cd68ed78dfcebfb96795a38921cf45cfd8d8a0b33362aefab870f7844e44d80`.

Initial rank-64/128 jobs 2171755/2171756 used 1e-4. Rank 128 diverged
from loss approximately 0.4 to 8–10 around updates 11–12. Both attempts were
cancelled and restarted from the same base at 2e-5 for matched conditions.
Cancelled attempts and the failed initial Ornith startup remain in allocation
accounting and logs. The Ornith startup failure was a missing `ninja` executable
in PATH; the environment already contained ninja and its bin directory was added.

Ornith model: `ornith-ai/Ornith-1.5-9B`, pinned revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. The same 865 sources and exact source-only
Qwen-generation prompts are used, with thinking enabled and no semantic correction.
All failed paper identifiers remain in outputs. No Ornith outputs enter Qwen-trained LoRAs.

## Matched evaluation

Three conditions: untrained Gemma 12B, rank 64, rank 128. All use exactly the same
source-only prompt, thinking mode, temperature 1.0/top-p 0.95/top-k 20, seed policy,
full sources, output budget and bounded format-retry policy. No semantic correction.
Evaluation uses the pinned project-alexandria historical QA prompt/sanitizer/parser,
Qwen2.5-7B-Instruct BF16 revision `a09a35458c702b33eeacc393d103063234e8bc28`,
temperature 0.5, top-p 0.95, max 100 tokens, frequency/presence penalties 1.05.
970 questions per condition; 2,910 scored answers total. Failed generator papers
count as ten wrong answers. Paper-cluster bootstrap and paired differences use
10,000 draws with a fixed seed. Test questions/gold answers never enter training
or generation. No test-set checkpoint selection is performed.

The historical Gemma 12B raw QA accuracy of 86.60% used different generation
settings and is not the matched no-LoRA control. No adapter improvement is claimed
until the new evaluation finishes. QA measures answerability, not independent
human factual correctness. Native summary-token lengths, real throughput,
training compute and full Slurm allocation GPU-hours are recorded separately.

## Artifacts and monitoring

- `STATUS.json` / `MONITOR.html`: continuously refreshed progress and owned job IDs.
- `outputs/train-r64/adapter`, `outputs/train-r128/adapter`: final adapters after one epoch.
- `outputs/ornith/papers_and_summaries.jsonl`, `performance.json`: same-paper Ornith results.
- `outputs/evaluation/RESULTS.md`, `report.json`, `qa-results.json`: matched QA results and immutable answer records.
- `outputs/allocation_accounting.json`: allocated GPU-hours, including cancelled/failed attempts.
- `outputs/hf_upload.json`: publication status and dataset URL/revision when completed.
- `outputs/persistence.json`: durable second-copy progress under `/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-distillation-865-20261004`.

The workflow monitor is a local process with its exact host/PID in `monitor_process.json`.
Training and evaluation execute independently in Slurm. Hugging Face upload uses the
standard local HF login and never saves credentials in this dataset or repository.
