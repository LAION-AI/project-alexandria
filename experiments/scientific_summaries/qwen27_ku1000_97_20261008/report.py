"""QA counts, paper-level intervals and full-source structured-KU copy audits."""
import csv
import json
import statistics
from pathlib import Path
import time

from common import ROOT, digest, load, sources, write
from evaluate import CONDITIONS
from ngram_overlap import SourceIndex, audit_summary, tokenize, VERSION
from copy_overlap_eval import compact


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k)] + strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return [str(value)] if value is not None else []


def prose_fragments(result):
    output = []
    for unit in result["knowledge_units"]:
        output.append(unit["context_summary"])
        for entity in unit["entities"]:
            output += [entity["name"], entity["entity_type"]]
            output += strings(entity["attributes"])
            for relation in entity["relationships"]:
                output += [relation["predicate"], relation["target"]]
                output += strings(relation["attributes"])
    return output


def full_audit(papers):
    rows = []
    for paper in papers:
        ku = load(ROOT / "outputs/knowledge_units" / paper["document_id"] / "document.json")
        index = SourceIndex(paper["fulltext"])
        values = prose_fragments(ku["result"])
        # One fragment per factual string; graph fields are not evidence quotes.
        # Offsets/hashes/IDs and prompts/reasoning/MCQs are never audited as prose.
        audit = audit_summary(paper["fulltext"], {
            "narrative": values, "title": ku["result"]["title"]}, index=index)
        jac = {}
        for n in [5, 7, 11]:
            source_grams = index.grams(n)
            generated_grams = set()
            for value in values:
                tokens = tokenize(value)[1]
                generated_grams.update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
            jac[str(n)] = len(source_grams & generated_grams) / max(1, len(source_grams | generated_grams))
        rows.append(dict(document_id=paper["document_id"], source_sha256=paper["fulltext_sha256"],
            status=ku["status"], audit=audit, source_ku_jaccard=jac))
    output = ROOT / "outputs"
    with (output / "copy_overlap_details.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(audit_version=VERSION, complete=True,
        allowed_consecutive_words=5, borderline_consecutive_words=6,
        violation_from_consecutive_words=7, conditions={"knowledge_units_1000": compact(rows)},
        generation_failed_papers=sum(r["status"] != "generated" for r in rows),
        mean_jaccard={n: statistics.mean(r["source_ku_jaccard"][n] for r in rows) for n in ["5", "7", "11"]},
        fragmentation="Each KU context summary/name/attribute label/value/relation predicate/target is separate; bibliographic title is metadata.",
        no_cross_fragment_grams=True, qa_filtering=False, sources_untruncated=True)
    write(output / "copy_overlap.json", report)
    fields = ["document_id", "status", "source_sha256", "narrative_longest_match",
              "narrative_coverage_6plus", "metadata_longest_match", "jaccard_5", "jaccard_7", "jaccard_11"]
    with (output / "copy_overlap.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            n = row["audit"]["categories"]["narrative"]
            writer.writerow(dict(document_id=row["document_id"], status=row["status"], source_sha256=row["source_sha256"],
                narrative_longest_match=n["longest_contiguous_match_words"],
                narrative_coverage_6plus=n["covered_word_fraction_by_minimum_run"]["6"],
                metadata_longest_match=row["audit"]["categories"]["metadata"]["longest_contiguous_match_words"],
                **{"jaccard_"+n: row["source_ku_jaccard"][n] for n in ["5", "7", "11"]}))
    return report


def finalize():
    import numpy as np
    papers = sources()
    documents = [load(ROOT / "outputs/qa" / (p["document_id"] + ".json")) for p in papers]
    assert all(len(d["rows"]) == 10 for d in documents)
    correct = np.array([[sum(r["predictions"][c] == r["gold"] for r in d["rows"])
        for c in CONDITIONS] for d in documents])
    rng = np.random.default_rng(250219413)
    samples = correct[rng.integers(0, 97, (10000, 97))].mean(axis=1) / 10
    scores = {}
    for i, c in enumerate(CONDITIONS):
        scores[c] = dict(correct=int(correct[:, i].sum()), total=970,
            accuracy=float(correct[:, i].sum()/970),
            invalid=sum(r["predictions"][c] is None for d in documents for r in d["rows"]),
            confidence_interval_95=[float(v) for v in np.quantile(samples[:, i], [.025, .975])])
    deltas = {}
    for c in ["no_context", "original", "gemma_r128_fp8_summary"]:
        difference = samples[:, CONDITIONS.index("knowledge_units_1000")] - samples[:, CONDITIONS.index(c)]
        deltas["knowledge_units_1000_minus_"+c] = dict(
            point=scores["knowledge_units_1000"]["accuracy"]-scores[c]["accuracy"],
            confidence_interval_95=[float(v) for v in np.quantile(difference, [.025, .975])])
    overlap = full_audit(papers)
    units = [load(ROOT / "outputs/knowledge_units" / p["document_id"] / "document.json") for p in papers]
    statistics_report = dict(papers=97, source_words=sum(d["source_words"] for d in units),
        chunks_expected=sum(d["chunks_expected"] for d in units),
        chunks_generated=sum(d["chunks_generated"] for d in units),
        failed_papers=sum(d["status"] != "generated" for d in units),
        mean_chunks=statistics.mean(d["chunks_generated"] for d in units),
        prompt_tokens=sum(d["prompt_tokens"] for d in units),
        completion_tokens=sum(d["completion_tokens"] for d in units))
    report = dict(complete=True, protocol=load(ROOT / "inputs/protocol.json"), scores=scores,
        paired_paper_bootstrap=deltas, extraction=statistics_report, overlap=overlap,
        qa=load(ROOT / "outputs/qa_complete.json"), generation_metrics=[load(ROOT / f"outputs/extraction-replica-{r}.json") for r in range(3)],
        no_qa_feedback_into_extraction=True)
    write(ROOT / "outputs/report.json", report)
    write(ROOT / "outputs/qa-results.json", dict(documents=documents, scores=scores))
    lines = ["# Qwen 27B Knowledge Units: 1,000-word chunks on 97 papers", "",
        "Completed fresh evaluation on all 97 frozen papers / 970 MCQs. The extractor uses the repository's sequential few-shot KU prompt, the preceding ten generated KUs as naming context, and sentence-aware targets of at most 1,000 words. No source opening abstract, neighbor window, semantic correction or document-wide alias pass is added. This is a larger-chunk extension of the paper's 200-word full-paper protocol.", "",
        "The fixed Qwen2.5-7B-Instruct answerer uses the unchanged historical ASCII-sanitized prompt/parser, temperature 0.5, top-p 0.95, 100 output tokens, frequency/presence penalties 1.05 and at most four invalid-answer retries. Every condition was freshly judged. No questions or gold labels are supplied to extraction; no paper is dropped or regenerated after QA.", "",
        "| Context | Correct / 970 | QA accuracy | Invalid / failed slots |", "| --- | ---: | ---: | ---: |"]
    for c, s in scores.items():
        lines.append(f"| {c} | {s['correct']} | {100*s['accuracy']:.2f}% | {s['invalid']} |")
    lines += ["", "## Paired differences", ""]
    for c, d in deltas.items():
        ci = d["confidence_interval_95"]
        lines.append(f"- {c}: {100*d['point']:+.2f} percentage points; 95% paper-bootstrap interval [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}].")
    lines += ["", "Intervals resample 97 paper clusters with all ten MCQs retained. The fixed stochastic judge can add sampling variation; QA measures answerability under this judge, not independent human factuality.", "",
        "## Extraction and compute", "", f"- {statistics_report['chunks_generated']}/{statistics_report['chunks_expected']} chunks generated; {statistics_report['failed_papers']} failed papers. Failed KU papers retain ten wrong slots.",
        f"- {statistics_report['source_words']:,} source words; {statistics_report['completion_tokens']:,} extractor completion tokens, including bounded format retries.",
        "- Three independent single-GH200 extractor replicas and one single-GH200 judge share an exclusive Booster node. Documents are concurrent; each document's chunks remain sequential.",
        "- Per-replica timing, raw usage, server startup and total allocation accounting are retained separately. Concurrent request latency sums are not GPU wall time.", "", "## Source-copy audit", ""]
    n = overlap["conditions"]["knowledge_units_1000"]["narrative"]
    lines += [f"Full untruncated-source audit {VERSION}: {n['strict_five_word_pass']}/97 narrative outputs meet the five-word limit; {n['borderline_six_only']} have only a six-word maximum; {n['violation_seven_plus']} contain a run of seven or more words. Maximum run: {n['maximum_contiguous_match_words']} words. Mean coverage in runs ≥6: {100*n['mean_covered_fraction_six_plus']:.2f}%.", "",
        "Factual strings are checked individually; separate graph fields are never concatenated to manufacture an overlap. Entity names and attribute/relation labels remain visible in the factual diagnostic. Document titles are bibliographic metadata. Fingerprints, source offsets, entity IDs, prompts, reasoning, questions and gold labels are excluded from KU prose. No explicit evidence quotes are emitted by this KU schema.", "",
        "Five-, seven- and eleven-word Jaccard means: " + ", ".join(f"{k}: {v:.6f}" for k, v in overlap["mean_jaccard"].items()) + ". These fragment-based diagnostics are not automatically identical to the historical paper's serialization metric. Lexical overlap alone does not establish plagiarism.", "",
        "Detailed evidence: `report.json`, `qa-results.json`, `copy_overlap.json`, `copy_overlap.csv`, `copy_overlap_details.jsonl`, and generated KU artifacts. Raw sources and source-containing request traces remain in Scratch/durable experiment storage, outside GitHub."]
    (ROOT / "outputs/RESULTS.md").write_text("\n".join(lines)+"\n")
    return report
