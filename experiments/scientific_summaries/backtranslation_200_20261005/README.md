# Back-translation repair benchmark: 200 no-thinking summary versions

Status: all three arms completed on 200 summary versions; see [RESULTS.md](RESULTS.md) for measured throughput, fidelity, overlap and QA.

## Models and fixed comparison

- [WindyTranslate/translate-de-en](https://huggingface.co/WindyTranslate/translate-de-en), revision `fb6f91ba4c16c3c3e1778633d1a6c1cd6be4d0ca`, paired with [WindyTranslate/translate-en-de](https://huggingface.co/WindyTranslate/translate-en-de), revision `751ef3680a3c312229aa73a0176a228f0c233b6a`. Both greedy and beam-four round trips are measured using the same windows. Marian, FP16, PyTorch SDPA, length sorting and dynamic padding.
- [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it), revision `10042cb0e6e7fdce748996a71dc3dc432a4e0c89`. BF16 vLLM, language-only inference, context 2,048, chunked prefill, prefix caching, 512 maximum active sequences, 16,384 batched tokens. Greedy decoding; this is an explicit change from the checkpoint's sampling defaults.
- Meaning filter: [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli), revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Both directions require entailment ≥0.90. The threshold is fixed before held-out QA. This filter is not a proof of scientific equivalence.

The 200 inputs are **summary versions of 97 distinct papers**, not 200 independent papers: all 97 no-thinking tuned merged FP8 summaries, all 97 tuned merged BF16 summaries, and six live BF16 summaries selected by document-ID SHA256 ordering. They use the Qwen-distilled Gemma 4 12B rank-128 generation adapter. No original paper, MCQ, gold answer, published training record or cached baseline score is changed.

## Translation inputs and exact prompt

Narrative windows with source-copy runs of at least six normalized whitespace-delimited words are selected. A preceding sentence/chunk is included where available, with disjoint windows of at most two chunks and 75 whitespace words per chunk. Proof quotes and metadata are retained and audited separately. Full windows are tokenized without truncation; oversized or unfinished translations are flagged and rejected. One English → German → English round trip is measured, without repeated rewriting until overlap disappears.

Marian receives each English window as tokenizer input for EN→DE, then its complete German output as input to DE→EN. No textual instruction is added. `do_sample=False`, `num_beams=1` or `4`, `max_new_tokens=511`, `use_cache=True`, `renormalize_logits=True`.

TranslateGemma uses the checkpoint's exact chat template with this single message (reverse the two language codes for the return trip):

```json
[{"role":"user","content":[{"type":"text","source_lang_code":"en","target_lang_code":"de","text":"<complete selected narrative window>"}]}]
```

There is no system message and no question/gold/reasoning in translation inputs. Native vLLM sampling uses `temperature=0`, `max_tokens=768`, `seed=20261005`, with the checkpoint's EOS/end-of-turn IDs `[1,106]` as explicit stop tokens. Tokenized chat templates explicitly request `return_dict=False` for integer ID lists under Transformers 5.18.

Batch sizes 64, 128, 256 and 512 are compared using 512 copy-bearing windows from teacher training papers disjoint from all 97 evaluation papers. Selection maximizes complete round-trip native output tokens/second, never QA accuracy. Complete cohort throughput includes both directions and tokenization, excludes setup/tuning/QC/QA, and reports window and summary-version rates alongside tokenizer-specific native token rates.

## Numbers, units, formulas and meaning

Raw candidate evidence is retained even if a repair is rejected. Critical-value signatures compare numbers, signs, decimal/scientific values, associated units, detected equations, Greek variables, exponents, subscripts and inequalities. Cosmetic spacing, supported Unicode/LaTeX forms and decimal trailing zeros are normalized. General algebraic equivalence, arbitrary unit conversion and every scientific notation are not fully parsed; suspect mismatches are not automatically human-labeled errors.

A candidate is inserted only after successful generation, unchanged critical signatures, bidirectional NLI entailment ≥0.90 and candidate-window source overlap ≤5 words. Full assembled narratives are checked again and rolled back if global critical signatures differ. Rejection preserves the original passage and its copy flag. A zero-change result after guarding must be interpreted with acceptance coverage and residual copying, not presented as perfect translator fidelity.

## Evaluation and provenance

- Pinned historical test set: 97 papers, 970 MCQs; SHA256 `a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261`.
- Original, raw back-translated and guarded outputs receive full source-copy audit 2.1: ≤5 allowed, exactly six borderline, ≥7 flagged; narrative, explicit evidence and metadata remain separate.
- Guarded QA uses the fixed historical prompt/parser and Qwen2.5-7B-Instruct: temperature 0.5, top-p 0.95, max 100 tokens, frequency and presence penalties 1.05, four retries for invalid parsed answers. Unchanged contexts reuse their exact cached responses. Changed contexts use native continuous batching, max 64 sequences/16,384 batched tokens. Sampling schedule differences and judge noise are documented.
- All 200 versions retain ten questions each: 2,000 correlated version/question slots per arm. Paired bootstrap intervals resample 97 paper clusters, keeping versions of the same paper together.
- Translation, semantic QC and QA run on four separate GPUs of one owned booster node. Two Windy arms share their GPU sequentially. Active translation time and allocated node GPU hours are separate metrics.
- All model files are downloaded on a login node as [recommended by JSC](https://apps.fz-juelich.de/jsc/hps/jupiter/batchsystem.html), then all preparation, inference, audits and compression run on compute with offline mode. Failed initial preparation job 2181277 used 74 seconds of one four-GPU node (0.08222 allocated GPU hours); it failed because compute nodes have no Internet access.

Scratch: `/e/fscratch/reformo/schuhmann1/scientific-backtranslation-200-20261005`.

Durable complete source/input, raw translations, numerical/formula checks, NLI, QA and overlap evidence: `/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-backtranslation-200-20261005` (created at completion). Compact results and compressed evaluation evidence are exported here. Credentials and model weights are excluded.

Implementation: [code](code). Sixteen targeted numerical/formula/patch checks passed before execution, including compound units, leading decimals, subtraction terms and mathematical functions. Runtime compatibility is checked by the warm-up and disjoint batch sweep on compute. Batch selection prioritizes complete round trips, then speed; unfinished outputs cannot win through early termination.

Critical guard v1.1 additionally preserves proportionality and solar-unit markers. Earlier Windy guard outputs are archived and recomputed from the unchanged raw translations before QA; no QA scores were viewed or used for this change. Successful raw translation results are reused.
