# Comparative findings and final allocation accounting

200 no-thinking summary versions from 97 held-out papers were processed. These original summaries already use the Qwen-distilled Gemma 4 12B rank-128 adapter; the original QA row below is a no-repair reference, not an untuned-model score.

## Main results

| Repair arm | Complete EN→DE→EN time | Native output tokens/s/GPU | QA correct / 2,000 | QA accuracy | Mean narrative coverage by source runs ≥6 words | Strict narrative ≤5 pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original / no repair | — | — | 1,882 | 94.10% | 15.403% | 0/200 |
| windy_greedy | 73.45 s | 7149.4 | 1873 | 93.65% | 11.162% | 0/200 |
| windy_beam4 | 125.02 s | 4178.8 | 1875 | 93.75% | 11.295% | 0/200 |
| translategemma | 34.51 s | 16905.6 | 1874 | 93.70% | 9.849% | 0/200 |

TranslateGemma 4B is 2.13× faster than Windy greedy and 3.62× faster than Windy beam four on the same selected windows. It also inserts more guarded repairs and reduces copied-word coverage the most. Native token rates use different tokenizers; round-trip wall time and window rates are the direct comparison.

The QA point estimates are slightly below the original 94.10%, so this experiment shows no QA improvement from the repair stage. Paper-cluster intervals are in [RESULTS.md](RESULTS.md). The fixed historical judge samples at temperature 0.5; its scheduling differs from the cached reference, so these small differences also contain judge sampling noise.

All 200 guarded narratives still contain at least one source-copy run of seven or more words. TranslateGemma reduces mean coverage from 15.403% to 9.849%, a reduction of about 36%, but a single round trip with conservative rejection does not meet the strict whole-summary five-word requirement. These timings cannot be presented as costs for producing fully compliant summaries.

## Numerical and formula fidelity

| Arm | Suspect numeric changes / numeric windows | Suspect formula/variable/symbol changes / mathematical windows | Accepted windows inserted / 6,379 | Guarded summaries with detected critical-signature changes |
| --- | ---: | ---: | ---: | ---: |
| windy_greedy | 471/2961 (15.91%) | 521/2177 (23.93%) | 1985 | 0/200 |
| windy_beam4 | 399/2979 (13.39%) | 525/2198 (23.89%) | 1971 | 0/200 |
| translategemma | 281/2993 (9.39%) | 325/2216 (14.67%) | 2389 | 0/200 |

Denominators above include successful round trips with the corresponding original content. Failed/capped windows retain their original text and remain in the throughput/failure records. Signature flags are conservative review flags, not human-labeled errors: harmless reordering or equivalent notation can also be rejected.

Raw Windy examples flatten scientific exponents (`10⁴` → `104`), remove solar-unit markers (`L⊙` → `L`) and alter proportionality expressions. Blind back-translation is therefore unsuitable for scientific corrections. Guard v1.1 checks numeric values, units, signs, detected formulas, variables, exponents, inequalities and relevant mathematical/unit symbols, plus bidirectional NLI. Rejected windows retain their original text. Zero detected drift after guarding is produced by this rejection policy; it does not certify every possible mathematical expression or scientific claim.

Guard v1.1 was applied in a separate quality-only run to unchanged saved translations. Earlier guard outputs and QA are archived. The change was motivated by raw translation errors and code checks, without using QA scores to choose repair rules. Earlier QA responses are reused only for exact guarded contexts after verifying all historical prompt hashes; changed contexts are reevaluated. No paper, question or gold answer was removed or modified.

## Final Slurm GPU hours

| Job | Outcome | Node seconds | Allocated GPU hours |
| --- | --- | ---: | ---: |
| 2181277 | FAILED | 74 | 0.08222 |
| 2181512 | FAILED | 240 | 0.26667 |
| 2181537 | CANCELLED by 20488 | 410 | 0.45556 |
| 2181588 | COMPLETED | 346 | 0.38444 |
| 2181662 | COMPLETED | 319 | 0.35444 |
| **Total** | **Including setup, retained partial results, retries and guard recheck** | **1389** | **1.54333** |

These are allocated GPU hours on exclusive four-GPU nodes, including model loading, compilation, quality checks and QA. Active translation phase hours in RESULTS.md are components of these allocations, not additional charges. Queue waiting and login-node model downloads do not consume allocated GPU hours.

All raw candidates, rejected repairs, NLI, numerical/formula signatures, QA attempts and overlap evidence are retained in the durable evidence directory. Exact model revisions, sampling parameters, prompts, runtime versions and executed code snapshots accompany the manifest.
