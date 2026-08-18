# Published results (arXiv v2)

Source: [Project Alexandria, arXiv:2502.19413v2](https://arxiv.org/html/2502.19413v2).
These tables transcribe the published paper; this repository does not claim to have rerun them.
Machine-readable copies are in [`results/`](../../results/).

## Abstract-level factual retention

Each cell reports `[no-context lower bound – original-text upper bound]` and KU accuracy. The study
used 1,000 abstracts per domain and three generated MCQs per abstract.

| Model | Medical | CS | Mathematics | Physics |
|---|---:|---:|---:|---:|
| Gemini 1.5 Flash 002 | [42.28–97.17], **93.37** | [58.56–97.27], **93.62** | [52.26–94.68], **91.82** | [34.19–95.30], **92.97** |
| Qwen 2.5 7B | [42.76–97.00], **92.76** | [58.59–97.27], **93.45** | [52.29–94.79], **92.87** | [36.80–95.29], **92.97** |
| Mistral Small 22B | [42.46–97.13], **92.33** | [58.79–97.37], **94.70** | [52.33–94.74], **92.91** | [34.58–95.38], **90.56** |
| Ministral 2410 3B | [42.24–97.06], **88.22** | [58.48–97.36], **91.91** | [52.21–94.80], **87.65** | [33.03–95.29], **87.14** |
| Llama 3.2 3B | [42.52–97.10], **87.08** | [58.63–97.34], **88.47** | [51.61–94.82], **86.44** | [36.89–95.33], **86.90** |
| Llama 3.1 8B | [42.69–97.13], **85.80** | [58.68–97.31], **87.75** | [52.00–94.81], **84.21** | [37.04–95.29], **85.43** |

## Full-paper factual retention

The study used 100 medical and 100 physics papers, 200-word chunks, ten previous KUs as context,
and ten MCQs per paper.

| Model | Physics | Medical |
|---|---:|---:|
| Gemini 1.5 Flash 002 | [49.48–90.72], **83.51** | [46.96–94.13], **81.76** |
| Qwen 2.5 7B | [52.23–89.69], **79.04** | [50.45–93.24], **88.29** |
| Mistral Small 22B | [50.86–89.35], **81.44** | [48.31–94.59], **90.20** |

Long documents reduce both the original-text upper bound and KU performance, but KU scores remain
substantially closer to original context than to no context.

## Text overlap

Average and top-5% overlap between original abstracts and KUs or reconstructed text:

| Model / comparison | Slice | Sherlock % | 5-gram | 7-gram | 11-gram |
|---|---|---:|---:|---:|---:|
| Gemini / KU | Overall | 2.7 | 0.009 | 0.003 | 0.001 |
| Gemini / KU | Top 5% | 14.5 | 0.023 | 0.011 | 0.003 |
| Qwen / KU | Overall | 5.9 | 0.028 | 0.015 | 0.005 |
| Qwen / KU | Top 5% | 22.9 | 0.070 | 0.047 | 0.024 |
| Gemini / reconstruction | Overall | 3.8 | 0.022 | 0.010 | 0.002 |
| Gemini / reconstruction | Top 5% | 12.1 | 0.047 | 0.030 | 0.013 |
| Qwen / reconstruction | Overall | 17.8 | 0.142 | 0.123 | 0.098 |
| Qwen / reconstruction | Top 5% | 24.0 | 0.175 | 0.157 | 0.133 |

These are empirical overlap checks, not legal tests. The paper reports that manual inspection of
the highest-overlap passages mostly found stock scientific phrases.
