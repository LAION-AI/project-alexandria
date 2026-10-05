# Ornith 1.5 9B → Gemma 4 12B IT generator distillation

The user froze Ornith generation at **736 successful summaries** on 2026-10-04.
The queued rest-run 2175758 was cancelled before starting; the remaining 129 of
the original 865-paper source cohort are excluded from this frozen collection.

## Frozen targets and publication

The independent Hugging Face dataset is **published and fully verified**:

[ChristophSchuhmann/scientific-summary-distillation-Ornith-1.5-9B-736-20261004](https://huggingface.co/datasets/ChristophSchuhmann/scientific-summary-distillation-Ornith-1.5-9B-736-20261004)

Pinned revision: `1ab11230eb1928c790a9f8e0a13ec2563565dc82`. All 778 release files match their remote
content hashes (LFS SHA256 or Git blob hashes) and sizes. The release contains
736 uncompressed JSONL trace archives with original file content and member hashes.

`outputs/hf_upload.json` is authoritative for publication status, verified file
counts and the pinned published revision. The local release contains full sources,
exact source-only requests, the original emitted answers and matching actual
reasoning, per-call sampling parameters, validation flags, and preserved earlier
returned attempts for every included paper. No corrected targets are substituted.

Teacher: `ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, BF16 native autoregressive decoding,
no DFlash. Thinking is enabled: 111 retained outputs used uncapped thinking;
625 used a 16,384-token thinking budget. All use a total output budget of 24,576,
full source text and the original Qwen scientific generation prompt. Temperature
1.0, top-p 0.95, top-k 20. Actual per-call records are authoritative.

All 736 outputs have complete top-level fields, readable narratives and nonempty
reasoning. **None passes the stricter source-evidence schema validator.** Evidence
quotes exceeding five words and nested field errors are common. No semantic QC
or correction was run for these Ornith records; Qwen acceptance scores do not label
these targets. The user requested training on the retained raw collection as is.

The 97 held-out evaluation papers and their 970 MCQs are excluded from training
by identities, normalized titles, source hashes and substantial text overlap.
Their questions and gold answers are never included in the published train split.
The 736 sources cover ten domains, with frozen source and output SHA256 hashes.

Native training data is complete: **22,549,953 tokens**, including **15,333,192
supervised reasoning/answer tokens**; 11,742–49,505 tokens/example, mean 30,638.52.
The one-epoch sequence contains exactly 736 unique papers and 92 optimizer updates.
Training file SHA256: `7c946c2fea57388cdb048c09f3d84ca34057402c9089ac8b242231c40aa656f5`.
Preparation job 2175810 completed in 3 minutes 6 seconds.

## Training and automatically chained evaluation

| Task | Job | Hardware | Configuration |
| --- | --- | --- | --- |
| Release and native training data preparation | 2175810 | 1 node | Holdout audit, exact template rendering, trace archives/checksums |
| Gemma 4 12B IT rank 64 | 2175811 | 2 nodes / 8 GH200 GPUs | One epoch, alpha 128, dropout 0.05, starts after preparation |
| Gemma 4 12B IT rank 128 | 2175812 | 2 nodes / 8 GH200 GPUs | One epoch, alpha 256, dropout 0.05, starts after preparation |
| Held-out evaluation retry | 2180154 | 1 node / 4 GH200 GPUs | Running; original 2175820 failed on a Python Path/string construction and was repaired |

Both training jobs completed on 2026-10-05. Rank 64: **3,401 allocation seconds / 7.5578 billed GPU-hours**, 3,177.35 training seconds. Rank 128: **3,424 allocation seconds / 7.6089 billed GPU-hours**, 3,194.61 training seconds. Each completed all 92 optimizer steps and one epoch.

The new source-copy audit flags **734/736** Ornith raw training narratives at seven or more words (maximum 195; mean coverage in runs ≥6: 16.32%). The remaining two have a six-word longest match. The frozen dataset remains unchanged. [Complete overlap findings](../copy_overlap_audit_20261005/README.md). Final QA publication includes the mandatory audit of all 97 paper slots per condition.

Student: `google/gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Train only language-layer q/k/v/o
attention and gate/up/down MLP LoRA projections. Actual teacher thought and raw
answer are preserved in Gemma's native template; assistant-only loss, no truncation.
One deterministic epoch, seed 20261004, microbatch 1/GPU, effective batch 8,
AdamW peak LR 2e-5, weight decay 0.01, 5% warmup then cosine, BF16 base,
flex attention, gradient checkpointing and cut cross entropy. No reviewer/corrector
adapter is trained. Native training lengths and checksums are in `training/manifest.json`.

Evaluation uses precisely the previous matched source-only prompt, thinking-enabled
generation, temperature 1.0/top-p 0.95/top-k 20, seeds, 24,576-token output budget
and three format attempts, without semantic correction. Each failed generation
counts as ten wrong answers. The pinned historical Alexandria QA prompt, sanitizer,
parser and Qwen2.5-7B BF16 judge are unchanged: temperature 0.5, top-p 0.95,
max 100 tokens, presence/frequency penalties 1.05, thinking disabled.

The already audited baseline summaries and **850/970 = 87.63%** QA responses are
reused verbatim from the previous Qwen study after checking protocol hashes and
all generation/QA settings. Two LoRA generator replicas and two fixed judge
replicas share the evaluation node, scoring completed paper triples concurrently.
All 97 papers / 970 answers per condition are retained and audited; paired paper
bootstrap intervals use 10,000 draws and seed 250219413.

For reference, Qwen-distilled Gemma ranks 64/128 scored 87.94% / 92.47%. Those
adapters trained on 865 source papers, versus 736 here. Collection size, domain
mix, target length and source-evidence failures differ: comparisons between the
two studies do not isolate teacher identity. QA measures answerability, not
independent human factual superiority. The historical Ornith raw QA score of
93.81% comes from a different generation recipe, not these 736 training papers.

## Artifacts

- `job.json`, `STATUS.json`, `MONITOR.html`: owned job IDs and live progress.
- `release/`: complete published data/provenance/checksums, after preparation.
- `outputs/train-r64/adapter`, `outputs/train-r128/adapter`: final adapters.
- `outputs/evaluation/RESULTS.md`, `report.json`, `qa-results.json`: final results.
- `outputs/allocation_accounting.json`: full billed GPU-hours, including failed
  original generation allocations, clearly separate from the new training study.
- Durable backup: `/e/data1/datasets/playground/mmlaion/schuhmann1/ornith-distillation-736-20261004`.

The login monitor uploads the release with the existing Hugging Face login and
copies completed outputs into the authorized GitHub checkout. Credentials remain
outside all datasets and repository artifacts. No new QA score is claimed until
the complete evaluation report exists.
