# Local Qwen3.8 validation (2026-08-18)

This is a smoke/quality run of the consolidated implementation, not a reproduction of the paper’s
aggregate numbers. The source was the first 5,357-word physics record in the historical
`physics_qa_pairs_expanded_clean.csv` dataset (SHA-256
`f24a3eb65a39e3926a5da00a5c493ca27943ded2ee3eaefb8957d1c7807fe160`). The source is now available
inside the owner-authorized long-paper evaluation Parquet; generated KU JSON still omits source
chunks.

## Runtime

| Field | Value |
|---|---|
| Served model | `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` as `qwen38` |
| Runtime | vLLM OpenAI-compatible server |
| Server context | 32,768 tokens |
| Hardware | Existing shared server, tensor parallel across 2 × RTX 3090 |
| Thinking | Disabled with `chat_template_kwargs` |
| Temperature | 0.2 |

This was a quantized Qwen3.8-27B validation, but not the requested one-GPU GGUF configuration. The
two GPUs were occupied by another user’s existing server, and only 9.3 GB of local disk was free,
so downloading another roughly 16 GB Q4_K_M copy would have been unsafe. The one-GPU command is
documented in [running Qwen3.8](../running-qwen38.md).

## Runs

| Run | Configuration | Units | Entity mentions | Canonical IDs | Warnings |
|---|---|---:|---:|---:|---:|
| Sequential API smoke test | One 5,357-word target, 2,500 output tokens | 1 | 10 | 10 | 0 |
| Parallel full-paper test | 120-word targets, 200-word side context, concurrency 4 | 49 | 200 | 160 before semantic resolution | 0 |
| Conservative entity resolution | Names + types, relations, KU summaries, lexical candidates | — | 200 | **153** | — |

The sequential run intentionally tested the “one-go short input” code path with an oversized input;
it is not the paper’s 200-word sequential baseline. The parallel run continuously and exactly
covered source word offsets 0–5357, and every recorded SHA/MinHash was independently regenerated.

The first, verbose canonicalization prompt exhausted its 1,800-token budget and safely fell back to
exact normalization. An initial compact synonym-dictionary prompt reduced 160 IDs to 135 but
included unsafe semantic merges. The final conservative resolver uses entity types and relation
predicates plus deterministic scope/category guards. It produced 153 IDs and six nontrivial alias
groups: all six were judged safe, with no unsupported or type-divergent IDs. The higher count is an
intentional precision-over-recall tradeoff.

## Independent GPT-5.6-sol review

The requested independent reviewer checked representative KUs throughout the source without
editing the implementation or seeing this summary in advance.

| Output | Fidelity | Coverage | Relations | Naming | Style independence | Schema | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential one-shot smoke | 3.1 | 1.2 | 3.1 | 4.5 | 4.2 | 5.0 | 2.6 |
| Parallel before improved canonicalization | 3.4 | 4.3 | 3.2 | 1.8 | 3.6 | 3.1 | 3.2 |
| Parallel after conservative canonicalization | — | — | — | **3.5** | — | **5.0** | **3.4** |

The reviewer found two clear factual errors in the parallel artifact: one masked-math token was
incorrectly interpreted as a dimension, and one KU reversed conditions associated with entropy
terms. It also found several facts leaking from side context into a target KU, figure-markup noise,
overconfident relation wording, undeclared entity types, and fragmented aliases. No false entity
merge was identified in the initial exact-normalized output.

In response, the code now:

- explicitly treats `@xmath...` placeholders as opaque;
- rejects facts supported only by before/after context in the prompt’s final check;
- instructs the model to ignore figure commands and coordinate residue;
- normalizes undeclared types to `other` while retaining the model’s proposed type for diagnosis;
- assigns one stable majority type per canonical ID;
- supplies KU summaries and lexical candidate pairs to a compact synonym-only resolver;
- rejects cross-category and specific-versus-generic scope merges conservatively;
- supports rerunning canonicalization without source text or re-extraction; and
- retries malformed extraction JSON once with a compact-output instruction.

The final review found all six retained alias groups safe and all schema/type checks valid. Clear
under-merges remain (including projection/embedding-method phrasings, physical-space names, link-set
variants, and logarithmic-term variants), so entity-resolution recall is the main remaining issue.
The overall score also retains the factual and target-boundary errors from extraction; a safer
resolver cannot repair those upstream facts.

## Expression-overlap diagnostic

Jaccard overlap against the source was low:

| Artifact | 5-gram | 7-gram | 11-gram |
|---|---:|---:|---:|
| Sequential one-shot | 0.00102 | 0.00033 | 0.00000 |
| Parallel full-paper | 0.01268 | 0.00450 | 0.00098 |

These values are diagnostics only, not a legal conclusion. The generated artifacts remain local
until their factual errors and overlap are reviewed for publication.
