"""Render five unchanged, guarded FP8 summaries as a standalone reading page."""
from pathlib import Path
import gzip
import hashlib
import html
import json


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
DEST = ROOT / "reader"
DATA = ROOT / "evidence" / "translategemma"
SNAPSHOT = "351ee28799c2127cdc005a8a93d0bd9eafb30ac3"
REPO = "https://github.com/LAION-AI/project-alexandria"
REPORT = f"{REPO}/blob/{SNAPSHOT}/experiments/scientific_summaries/backtranslation_200_20261005"
CHOICES = [
    ("arxiv-121", "Astronomy", "The dust shell around AFGL 5440"),
    ("arxiv-500047", "Computer science", "Real-time semantic segmentation"),
    ("bethgelab-125", "Chemistry", "Electrolytes for sodium-ion batteries"),
    ("bethgelab-100015", "Structural biology", "Regulating the Ycf1 transporter"),
    ("arxiv-149", "Mathematics", "Generalized entropy measures"),
]
LABELS = {
    "executive_summary": "Executive summary",
    "research_context": "Research context",
    "research_question_and_hypothesis": "Research questions & hypotheses",
    "methodological_details": "Methods",
    "procedures_and_architectures": "Procedures & architectures",
    "key_results": "Key results",
    "interpretation_and_theoretical_implications": "Interpretation & implications",
    "contradictions_and_limitations": "Limitations & contradictions",
    "claims": "Claims & supporting evidence",
    "data_and_code_availability": "Data & code availability",
    "robustness_and_ablation_notes": "Robustness & ablations",
    "ethical_considerations": "Ethical considerations",
    "key_figures_tables": "Figures & tables",
    "top_influential_citations": "Influential citations",
    "three_takeaways": "Three takeaways",
}


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def esc(value):
    return html.escape(str(value), quote=True)


def render(value):
    """Render every schema value; single-key statement/evidence pairs stay paired."""
    if value is None or value == "" or value == [] or value == {}:
        return '<p class="empty">Not specified in this saved output.</p>'
    if isinstance(value, str):
        return "".join(f"<p>{esc(part)}</p>" for part in value.split("\n") if part)
    if isinstance(value, list):
        return '<ul class="prose-list">' + "".join(f"<li>{render(item)}</li>" for item in value) + "</ul>"
    if isinstance(value, dict):
        if len(value) == 1:
            statement, evidence = next(iter(value.items()))
            if isinstance(evidence, str):
                quote = (f'<details class="evidence"><summary>Source excerpt</summary>'
                         f'<blockquote>{esc(evidence)}</blockquote></details>') if evidence else ""
                return f"<p>{esc(statement)}</p>" + quote
        parts = []
        for key, item in value.items():
            label = key.replace("_", " ").capitalize()
            parts.append(f'<div class="claim-part"><h4>{esc(label)}</h4>{render(item)}</div>')
        return "".join(parts)
    return f"<p>{esc(value)}</p>"


def main():
    rows = {r["document_id"]: r for r in read_jsonl(DATA / "summaries.jsonl.gz")
            if r["original_condition"] == "no_thinking_fp8"}
    audits = {}
    for row in read_jsonl(DATA / "copy_overlap_details.jsonl.gz"):
        audits[(row["condition"], row["document_id"])] = row["audit"]["categories"]["narrative"]
    papers = {p["document_id"]: p for p in json.loads((EXPERIMENTS / "data" / "testset.json").read_text())["papers"]}
    selected = []
    cards = []
    articles = []
    for number, (document_id, domain, short_title) in enumerate(CHOICES, 1):
        row = rows[document_id]
        assert not row["summary_rolled_back"]
        assert row["accepted_window_count"] > 0
        assert row["guarded_global_critical_values"]["passed"]
        summary = row["summary"]
        before = audits[("original", row["uid"])]
        after = audits[("guarded_backtranslation", row["uid"])]
        source = papers[document_id].get("source", {})
        source_url = ""
        if source.get("paper_id") and source.get("viewer_config") == "arxiv":
            source_url = "https://arxiv.org/abs/" + source["paper_id"]
        elif source.get("source_doi"):
            source_url = "https://doi.org/" + source["source_doi"]
        record = {
            "document_id": document_id,
            "domain": domain,
            "original_condition": row["original_condition"],
            "source_sha256": row["source_sha256"],
            "guarded_context_sha256": row["guarded_context_sha256"],
            "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "selected_windows": row["selected_window_count"],
            "accepted_windows": row["accepted_window_count"],
            "critical_signature_check_passed": row["guarded_global_critical_values"]["passed"],
            "copy_overlap": {"audit_version": "2.1", "before": before, "after": after},
            "paper_url": source_url,
            "summary": summary,
        }
        selected.append(record)
        cards.append(f'<a class="paper-link" href="#paper-{number}"><span class="paper-number">0{number}</span>'
                     f'<span><span class="paper-domain">{esc(domain)}</span><span class="paper-label">{esc(short_title)}</span></span></a>')
        authors = summary.get("authors", "")
        if not isinstance(authors, str):
            authors = json.dumps(authors, ensure_ascii=False)
        section_links = []
        sections = []
        for key, value in summary.items():
            if key in {"title", "authors", "field_subfield", "type_of_paper", "executive_summary", "three_takeaways"}:
                continue
            label = LABELS.get(key, key.replace("_", " ").capitalize())
            anchor = f"paper-{number}-{key}"
            section_links.append(f'<a href="#{anchor}">{esc(label)}</a>')
            opened = " open" if key in {"research_context", "methodological_details", "key_results"} else ""
            sections.append(f'<details class="section" id="{anchor}"{opened}><summary><h3>{esc(label)}</h3>'
                            '<span class="section-toggle" aria-hidden="true">+</span></summary>'
                            f'<div class="section-body">{render(value)}</div></details>')
        source_link = f'<a href="{esc(source_url)}" target="_blank" rel="noopener">Read the paper ↗</a>' if source_url else ""
        coverage_before = 100 * before["covered_word_fraction_by_minimum_run"]["6"]
        coverage_after = 100 * after["covered_word_fraction_by_minimum_run"]["6"]
        metadata = (f'<div class="paper-metadata"><span>{esc(summary.get("type_of_paper", ""))}</span>'
                    f'<span>{after["words"]:,} narrative words</span><span>~{max(1, round(after["words"] / 220))} min read</span></div>')
        articles.append(f'''<article class="paper" id="paper-{number}" aria-labelledby="title-{number}">
<div class="article-eyebrow"><span>0{number} / 05</span><span>{esc(domain)}</span><span class="pill">After backtranslation</span></div>
<h2 id="title-{number}">{esc(summary["title"])}</h2>
<p class="authors">{esc(authors)}</p>{metadata}
<div class="article-actions">{source_link}<a href="examples.json" download>Download saved outputs ↓</a><button class="expand" type="button">Expand all sections</button></div>
<section class="executive"><div class="eyebrow">Executive summary</div>{render(summary.get("executive_summary"))}</section>
<section class="takeaways"><h3>Three takeaways</h3>{render(summary.get("three_takeaways"))}</section>
<details class="contents"><summary>Jump to a section</summary><nav aria-label="Sections of paper {number}">{''.join(section_links)}</nav></details>
{''.join(sections)}
<details class="checks"><summary>About this repair & its checks</summary><div class="check-body"><dl>
<div><dt>Accepted repair windows</dt><dd>{row['accepted_window_count']} / {row['selected_window_count']}</dd></div>
<div><dt>Detected numerical / formula drift</dt><dd>None after guarding</dd></div>
<div><dt>Copy coverage, runs ≥6 words</dt><dd>{coverage_before:.2f}% → {coverage_after:.2f}%</dd></div>
<div><dt>Longest remaining narrative match</dt><dd>{after['longest_contiguous_match_words']} words</dd></div>
</dl><p>Suspect translations were rejected and their original passages retained. The five-word copy limit is still exceeded. These automated checks do not independently certify every scientific claim.</p>
<p class="identifiers">Saved output: {esc(row['uid'])}<br>Field: {esc(summary.get('field_subfield', ''))}</p></div></details>
<div class="paper-end"><span>End of paper 0{number}</span><a href="#top">Back to collection ↑</a></div>
</article>''')
    artifact = {
        "collection": "Five Qwen-distilled Gemma rank-128 summaries after guarded TranslateGemma backtranslation",
        "selection": "Five distinct subject areas, selected by document ID before inspecting per-paper QA; not a representative quality sample.",
        "generator": "google/gemma-4-12B-it; Qwen3.8-27B-distilled rank-128 adapter; merged FP8; no thinking",
        "repair": "google/translategemma-4b-it; BF16 greedy; one EN-DE-EN round on selected copy-bearing narrative windows; critical guard v1.1 and bidirectional NLI",
        "provenance_snapshot": SNAPSHOT,
        "notes": "Full saved post-repair summary objects are unchanged. No reasoning, questions, gold answers or full source texts are published in this reader. Source excerpts are preserved as separate collapsible evidence.",
        "examples": selected,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "examples.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stylesheet = (Path(__file__).parent / "reader.css").read_text(encoding="utf-8")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Read five complete Gemma 4 12B rank-128 scientific summaries after guarded TranslateGemma backtranslation, across five research fields.">
<meta name="theme-color" content="#174a3d"><title>Five scientific summaries · Project Alexandria</title>
<style>{stylesheet}</style></head><body id="top">
<header class="topbar"><div class="topbar-inner"><a class="brand" href="#top"><span class="brand-mark" aria-hidden="true">A</span>Project Alexandria</a><nav aria-label="Main navigation"><a href="{REPORT}/FINDINGS.md">Benchmark ↗</a><a href="{REPO}" class="repo-link">GitHub ↗</a></nav></div></header>
<main class="shell"><section class="hero"><div class="eyebrow">Scientific reading collection <span>05 October 2026</span></div>
<h1>Five papers.<br><em>After backtranslation.</em></h1>
<p class="intro">Explore complete scientific summaries across five fields, generated by Gemma and refined through a guarded English → German → English translation round.</p>
<div class="hero-meta"><span><b>Gemma 4 12B</b>Generator</span><span><b>Rank 128</b>Qwen-distilled LoRA</span><span><b>TranslateGemma 4B</b>Backtranslation</span><span><b>No thinking</b>Generation mode</span></div>
</section><div class="reading-layout"><aside class="sidebar"><div class="sidebar-inner"><div class="eyebrow">In this collection</div><nav class="paper-nav" aria-label="Five papers">{''.join(cards)}</nav>
<div class="reader-note"><span class="note-symbol" aria-hidden="true">✳</span><p>Full saved summaries, with no editorial rewriting. Repairs affect selected passages; source excerpts remain separate.</p><a href="#about">About these examples ↓</a></div></div></aside>
<div class="papers">{''.join(articles)}<section class="about" id="about"><div class="eyebrow">About the collection</div><h2>A reading view of the experiment.</h2>
<p>These five examples come from the 97-paper held-out evaluation. All use the Qwen-distilled rank-128 Gemma 4 12B adapter with merged FP8 weights and thinking disabled. Only copy-bearing narrative windows were backtranslated by TranslateGemma 4B.</p>
<p>A repair is inserted only after numerical/formula checks, bidirectional NLI meaning checks and a window-level copy check. Rejected passages keep their original wording. Residual copying remains: none of the 200 benchmark versions meets the strict whole-summary five-word limit.</p>
<p>The five papers were chosen to cover different research fields, not to rank quality or QA scores. All sections of their saved summaries are available above; empty sections are marked. The original generated text can contain errors. No new generation, correction or evaluation was performed to create this page.</p>
<div class="about-links"><a href="{REPORT}/RESULTS.md">Full methods & results ↗</a><a href="examples.json" download>Exact example data ↓</a><a href="{REPORT}/SCALING_38M_60M.md">GPU-hour estimates ↗</a></div></section></div></div>
</main><footer><span>Project Alexandria · Scientific summaries</span><a href="#top">Back to top ↑</a></footer>
<script>
document.querySelectorAll('.expand').forEach(button => button.addEventListener('click', () => {{
  const sections = [...button.closest('article').querySelectorAll('details.section')];
  const expand = sections.some(section => !section.open);
  sections.forEach(section => section.open = expand);
  button.textContent = expand ? 'Collapse detailed sections' : 'Expand all sections';
}}));
document.querySelectorAll('.contents a').forEach(link => link.addEventListener('click', () => {{
  const target = document.getElementById(link.getAttribute('href').slice(1));
  if (target) target.open = true;
}}));
if ('IntersectionObserver' in window) {{
  const links = [...document.querySelectorAll('.paper-link')];
  const observer = new IntersectionObserver(entries => {{
    entries.forEach(entry => {{if (entry.isIntersecting) {{
      links.forEach(link => {{const active = link.getAttribute('href') === '#' + entry.target.id;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
      }});
    }}}});
  }}, {{rootMargin: '-10% 0px -72% 0px', threshold: 0}});
  document.querySelectorAll('article.paper').forEach(paper => observer.observe(paper));
}}
</script></body></html>'''
    (DEST / "index.html").write_text(page, encoding="utf-8")
    manifest = {
        "artifact": "Static HTTPS reader",
        "page": "https://projects.laion.ai/project-alexandria/summary-examples/",
        "selection_document_ids": [c[0] for c in CHOICES],
        "input_snapshot": SNAPSHOT,
        "inputs": [{"path": str(path.relative_to(EXPERIMENTS)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in [DATA / "summaries.jsonl.gz", DATA / "copy_overlap_details.jsonl.gz"]],
        "outputs": [{"path": name, "sha256": hashlib.sha256((DEST / name).read_bytes()).hexdigest(), "bytes": (DEST / name).stat().st_size}
                    for name in ["index.html", "examples.json"]],
        "full_saved_summary_objects_preserved": True,
        "new_model_calls": 0,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(selected)} unchanged saved summaries in {DEST}")


if __name__ == "__main__":
    main()
