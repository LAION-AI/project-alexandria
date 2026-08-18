# Reproducibility guide

## 1. Preserve the source boundary

Obtain source documents lawfully and keep them outside the repository. Generated JSON does not
contain source chunks, but operators should still review KUs for accidental expressive overlap
before publishing them. Record source identifiers such as DOI or arXiv ID separately when allowed.

## 2. Start a model server

Use any OpenAI-compatible server or the offline vLLM adapter. Pin the exact model revision, runtime,
quantization, context length, and sampling configuration. Qwen-specific commands are in
[running-qwen38.md](running-qwen38.md).

## 3. Run extraction

For the published baseline, use sequential mode, 200-word chunks, and ten previous units. For the
new throughput path, use parallel mode, 500-word targets, and 1,000-word side context. Store command
lines and stdout/stderr beside artifacts.

## 4. Validate artifacts

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m json.tool outputs/paper.sequential.json >/dev/null
```

Inspect every `extraction_warnings` entry. Sample chunks across the whole document, not only the
opening. Score factual fidelity, coverage, relation quality, naming, style independence, and schema
validity from 0–5 using `experiments.quality.evaluate_unit` or an independent reviewer.

## 5. Reproduce evaluation

The paper’s main protocol generates source-grounded MCQs, then compares answering accuracy under:

1. no context (lower bound);
2. original source text (upper bound);
3. Knowledge Units;
4. optionally, text reconstructed only from KUs.

Use the same question set and answering model for every condition. Repeatedly regenerate question
sets to estimate variance. The paper reported 3–5% variation across question sets.

For expression overlap, strip generated context summaries before running Sherlock and 5/7/11-gram
Jaccard comparisons. Treat these as diagnostics, not legal standards.

## 6. Record a run manifest

At minimum retain: code commit, model repository/revision, quantization, runtime version, GPU,
sampling values, chunk/context sizes, source fingerprint, start/end timestamps, retry/failure count,
and output checksum. Do not include secrets or copyrighted source text in the manifest.
