# Why cosine similarity is not knowledge preservation

The original, unreleased project report tested BGE-M3 embeddings on 100 arXiv abstracts. Its
sanity check produced a striking counterexample: destroying word order barely reduced cosine
similarity, and the scrambled texts scored higher against the source than the Knowledge Units did.

| Comparison with original abstract | Cosine similarity |
|---|---:|
| Original (upper bound) | 1.0000 |
| Reconstructed KU text | 0.9130 |
| Scrambled words | **0.8903** |
| Scrambled bigrams | **0.8870** |
| Scrambled trigrams | **0.8832** |
| Knowledge Graph / Knowledge Unit | 0.8162 |
| Unrelated same-domain abstract | 0.4669 |
| Random words (reported in arXiv v2) | 0.45 |

The scrambled bag of words retains vocabulary and topical distribution but loses syntax, roles,
negation scope, chronology, and many causal relations. Its high score therefore cannot establish
that knowledge was preserved. The result supports evaluation with questions about explicit facts,
numbers, definitions, and relations instead of treating embedding proximity as a fidelity metric.

The arXiv v2 paper retained a rounded subset of this result in its Table 5. The exact values above
come from Table 13 of the unreleased PDF supplied with the historical workspace.

## Larger follow-up artifact

The workspace also contains a later 74,411-row aggregate covering BGE-M3, E5-Large-Instruct,
GTE-Large, and a raw DeBERTa baseline. Across that aggregate, the same failure mode appears:

| Embedding | Input–KU | Scrambled words | Scrambled bigrams | Scrambled trigrams |
|---|---:|---:|---:|---:|
| BGE-M3 | 0.9024 | 0.8913 | 0.8881 | 0.8851 |
| E5-Large-Instruct | 0.9435 | 0.9400 | 0.9391 | 0.9385 |
| GTE-Large | 0.9334 | 0.9460 | 0.9451 | 0.9445 |
| DeBERTaV3-base (raw pooling baseline) | 0.9843 | 0.9942 | 0.9943 | 0.9943 |

This follow-up was not reported in the paper and has not yet been independently audited. In
particular, the DeBERTa row is a raw encoder/pooling baseline rather than a recommended sentence
embedding configuration. It is preserved as historical evidence, not promoted as a benchmark.

Reproduction helpers live in
[`src/project_alexandria/experiments/similarity.py`](../../src/project_alexandria/experiments/similarity.py).
