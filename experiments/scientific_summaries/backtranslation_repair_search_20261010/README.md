# Back-translation search and targeted wording repair

Completed 10 October 2026: **44 comparison conditions, 20 frozen papers, 8,800 QA slots, zero invalid answers**. Saved KUs were reused; no new Knowledge Units or adapters were trained. Total allocated search compute is **2.590 GH200 GPU-hours**, including startup, QA and two failed launches.

**No tested setting reaches zero five-word-rule violations.** Guarded back-translation reduces copied wording; additional Qwen and Gemma E4B edits reduce it further. The final conservative 12B-KU chain preserves observed QA point estimates at 88.0% / 84.5%, with residual copied-word coverage of 0.413% / 0.551% for 500 / 1000 source words. Direct Gemma-only repair and its summary application are still unmeasured.

- [Recommendations and exact local-edit settings](RECOMMENDATIONS.md)
- [All 44 QA / copying conditions and conclusions](RESULTS.md)
- [38M / 60M generation and repair compute scenarios](SCALING.md)
- [Machine-readable scores](metrics.json), [protocol](protocol.json), [copy audit](copy_overlap.json), [CSV](copy_overlap.csv), [QA predictions](qa_predictions.jsonl.gz), [scaling inputs](scaling.json)
- [Earlier guarded KU repair and five-paper reader](../ku_backtranslation_20_20261010/README.md)
- [Earlier summary translation evidence](../backtranslation_200_20261005/SCALING_38M_60M.md)

## Adapter weights and both inference profiles

Both KU models below are rank128 adapters trained on 1000-word Qwen27 teacher targets for one epoch. Each supports **two evaluated inference profiles using the same weights**. There is no independently trained 500-word adapter in this release.

| KU base model | Adapter weights | 500-word profile | 1000-word profile |
| --- | --- | --- | --- |
| Gemma 4 12B IT | [Weights and model card](https://huggingface.co/laion/Alexandria-Gemma-4-12B-it-Qwen27-Knowledge-Units-LoRA-r128) | [ku500.json](https://huggingface.co/laion/Alexandria-Gemma-4-12B-it-Qwen27-Knowledge-Units-LoRA-r128/blob/main/profiles/ku500.json) | [ku1000.json](https://huggingface.co/laion/Alexandria-Gemma-4-12B-it-Qwen27-Knowledge-Units-LoRA-r128/blob/main/profiles/ku1000.json) |
| Gemma 4 E4B IT | [Weights and model card](https://huggingface.co/laion/Alexandria-Gemma-4-E4B-it-Qwen27-Knowledge-Units-LoRA-r128) | [ku500.json](https://huggingface.co/laion/Alexandria-Gemma-4-E4B-it-Qwen27-Knowledge-Units-LoRA-r128/blob/main/profiles/ku500.json) | [ku1000.json](https://huggingface.co/laion/Alexandria-Gemma-4-E4B-it-Qwen27-Knowledge-Units-LoRA-r128/blob/main/profiles/ku1000.json) |

The new extensive repair search evaluates the **12B** KU outputs. The E4B generator's previously measured BT results remain in its own model card; new 12B repair scores are not imputed to it.

## Summary adapter inventory

- [laion/Alexandria-Gemma-4-12B-it-Qwen27-Summaries-LoRA-r64](https://huggingface.co/laion/Alexandria-Gemma-4-12B-it-Qwen27-Summaries-LoRA-r64)
- [laion/Alexandria-Gemma-4-12B-it-Qwen27-Summaries-LoRA-r128](https://huggingface.co/laion/Alexandria-Gemma-4-12B-it-Qwen27-Summaries-LoRA-r128)
- [laion/Alexandria-Gemma-4-12B-it-Ornith9-Summaries-LoRA-r64](https://huggingface.co/laion/Alexandria-Gemma-4-12B-it-Ornith9-Summaries-LoRA-r64)
- [laion/Alexandria-Gemma-4-12B-it-Ornith9-Summaries-LoRA-r128](https://huggingface.co/laion/Alexandria-Gemma-4-12B-it-Ornith9-Summaries-LoRA-r128)

These adapters have their own retained training/evaluation provenance. The fastest summary scaling reference here applies specifically to the Qwen-distilled rank128 adapter, merged FP8 with thinking disabled.
