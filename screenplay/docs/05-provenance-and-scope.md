# Provenance and scope

## Handling of the source text

The screenplay is read on-machine and never redistributed.

| | |
|---|---|
| Published | `ku_chain.json` — structure, and a **link** to the public source |
| Published | offsets, SHA-256 digests, 16-value MinHash sketches (non-reversible) |
| Never published, never committed | the screenplay text, in any file, in any run directory |

`screenplay/.gitignore` blocks `*.txt` under any `runs/`, `**/script.normalized.txt`,
`**/matrix.txt`, and `**/*_screenplay.txt`. This is copied from the storytree repo, where a
narrower pattern **failed to cover a newly created directory and a full screenplay was
committed and had to be purged from history.** The pattern here matches by filename anywhere
in the tree for that reason. A pre-commit check additionally refuses any staged file over 20 KB
whose content matches the source digest or scores high on MinHash against it.

The extraction operates on lawfully accessible material for non-commercial scientific research,
under the text-and-data-mining research exemption LAION relies on for its other corpora.
As the repository README already states, that is a technical and legal *position*, not legal
advice, and the operator remains responsible for lawful access, temporary-copy handling, and
deletion.

## The limit that is specific to narrative

This is the part that does not carry over from the paper, and it should not be quietly
inherited.

Alexandria's original argument rests on a clean distinction: **facts are not protectable
expression.** That photosynthesis releases oxygen is not owned by the textbook that says so,
so a KU capturing it redistributes nothing protectable no matter how complete it is.

A screenplay is not like that. In a dramatic work, protection extends past the literal words
to elements such as the sequence of events, the structure of scenes, and the specific
interactions of characters. And this pipeline is designed — deliberately, because the brief
asks for it — to capture **exactly those**: every scene in order, every beat within it, every
state change, with the temporal order explicitly reconstructible.

So the honest statement of what C2 buys is narrow:

> The verbatim-overlap gate demonstrates that the artifact does not reproduce the screenplay's
> **language**. It does not, and cannot, establish that a fact-complete scene-by-scene
> temporally-ordered rendering of a film is unprotected — that object is closer to a detailed
> scene-by-scene synopsis than to a set of scientific facts, and paraphrase alone does not
> address it.

Two consequences for how this work gets written up:

1. **Do not extend the paper's headline claim to narrative without saying this.** The KU
   scores may transfer; the legal argument underneath them does not transfer unchanged.
2. **The n-gram numbers are evidence about expression only.** Reporting a low overlap figure
   next to a claim about copyright would overstate what was measured — the same shape of error
   as the eight in the storytree record, where a real number was reported as evidence for
   something it did not measure.

This is flagged, not blocking. The research purpose is legitimate, the source is public, the
artifact is structure, and the measurement is worth having. It is a caveat for the write-up
and a question for LAION's counsel, not a reason to build something different.

## The one exception: worked examples

[`07-worked-examples.md`](07-worked-examples.md) reproduces **three scenes of 225** — under 2%
of the work — alongside the units derived from them and the questions generated from them.

This is a deliberate exception to the rule above, and it is a scholarly quotation rather than
a redistribution: a reader cannot judge whether a Knowledge Unit preserves what it claims to
without seeing, for at least a few cases, what it was derived from. The scenes were chosen to
span the method's regimes — a dialogue-heavy exchange, a dense action scene, and a 23-word
scene that shows the floor — not for narrative interest.

No other file in the repository contains source prose, and the exception does not extend to
the corpus: it is three scenes for method illustration, not a sample anyone could assemble a
screenplay from.

## Scope of the published artifact

Published to the repo: the KU chain, the MCQ instrument, all check results, `protocol.json`,
and the analysis.

Not published: the screenplay, any prompt log containing source spans (per-call logs carry
full context and are gitignored), and the MCQ questions' source excerpts — questions are
published, the scene text they were generated from is not.

## Reproduction

A third party with lawful access to the same public script can reproduce the artifact:
`source_url` plus `source_sha256` plus the normalization steps in `script_map.json` pin the
exact input, and the fixed seed pins the scene sample. Anyone who cannot obtain the source can
still audit the KU chain, the checks, and the evaluation — which is the point of storing
structure by reference.
