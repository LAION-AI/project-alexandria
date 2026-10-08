"""Stage the existing frozen cohort, model metadata and exact extraction code."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
ROOT = Path("/e/fscratch/reformo/schuhmann1/scientific-qwen27-ku1000-97-20261008")
REFERENCE = Path("/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97/inputs")
TEST = REFERENCE / "project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a/experiments/scientific_summaries/data/testset.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ["code", "inputs", "outputs", "logs", "cache", "vendor/src", "vendor/audit", "traces"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    for path in CODE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
        shutil.copy2(path, ROOT / "code" / path.name)
    shutil.copy2(CODE / "run.sbatch", ROOT / "code/run.sbatch")
    shutil.copy2(CODE / "README.md", ROOT / "PROTOCOL.md")
    snapshot = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    package = REPO / "src/project_alexandria"
    for path in package.rglob("*.py"):
        destination = ROOT / "vendor/src/project_alexandria" / path.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for name in ["ngram_overlap.py", "copy_overlap_eval.py"]:
        shutil.copy2(REPO / "experiments/scientific_summaries" / name, ROOT / "vendor/audit" / name)
    shutil.copy2(REFERENCE / "eval_sources_only.json", ROOT / "inputs/sources_only.json")
    shutil.copy2(TEST, ROOT / "inputs/testset.json")
    baseline_path = REPO / "experiments/scientific_summaries/gemma_r128_fp8_97/evaluation/no_thinking_fp8-summaries.jsonl"
    baseline = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    (ROOT / "inputs/gemma_fp8_baseline.json").write_text(json.dumps([
        {k: row[k] for k in ["document_id", "source_sha256", "status", "judge_context"]}
        for row in baseline], ensure_ascii=False, indent=2) + "\n")
    raw = json.loads((ROOT / "inputs/sources_only.json").read_text())
    questions = json.loads(TEST.read_text())
    assert len(raw) == len(questions["papers"]) == len(baseline) == 97
    assert sum(len(d["questions"]) for d in questions["papers"]) == 970
    assert {p["document_id"] for p in raw} == {p["document_id"] for p in questions["papers"]} == {p["document_id"] for p in baseline}
    assert sha(TEST) == "a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261"
    sys.path.insert(0, str(ROOT / "vendor/src"))
    from project_alexandria.chunking import split_text
    from project_alexandria.parsing import parse_unit_response
    chunks = []
    for paper in raw:
        target = split_text(paper["fulltext"], 1000)
        assert all(0 < c.word_count <= 1000 for c in target)
        assert " ".join(c.text for c in target).split() == paper["fulltext"].split(), paper["document_id"]
        assert hashlib.sha256(paper["fulltext"].encode()).hexdigest() == paper["fulltext_sha256"]
        chunks += target
    try:
        parse_unit_response('{"context_summary":"Incomplete", "entities":[{"name":"T"')
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("Strict parsing accepted truncated JSON")
    extractor_pin = json.loads((REFERENCE / "qwen38-teacher-manifest.json").read_text())
    judge_pin = json.loads((REFERENCE / "judge-weights-manifest.json").read_text())
    protocol = dict(created_unix=time.time(), paper="https://arxiv.org/html/2502.19413v2",
        repository_snapshot_commit=snapshot, testset_sha256=sha(TEST), paper_count=97,
        questions_per_condition=970, source_words=sum(len(p["fulltext"].split()) for p in raw),
        chunks=len(chunks), target_max_words=1000, mode="sequential", previous_units=10,
        extra_source_context_words=0, canonicalization=False,
        generation_format_retry_budgets=[8192, 16384], thinking=False,
        QA_feedback_to_extraction=False, heldout_training=False,
        extractor=dict(model=extractor_pin["model"], revision=extractor_pin["revision"],
            local_path="/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/models/Qwen3.8-27B-FP8",
            temperature=.1, top_p=.95, precision="upstream FP8 weights; BF16 activations/KV",
            replicas=3, active_documents_per_replica=16),
        judge=dict(model=judge_pin["model"], revision=judge_pin["revision"],
            local_path="/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97/models/Qwen2.5-7B-Instruct",
            temperature=.5, top_p=.95, max_output_tokens=100, frequency_penalty=1.05,
            presence_penalty=1.05, invalid_answer_retries=4, concurrency=4, context_limit=32768,
            legacy_ascii_sanitizer=True, historical_prompt_and_parser=True,
            fresh_judging_all_conditions=True),
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "inputs/sources_only.json", ROOT / "inputs/testset.json", ROOT / "inputs/gemma_fp8_baseline.json"]},
        vendor_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "vendor").rglob("*.py")},
        generation_inputs_contain_questions_or_gold=False,
        preflight=dict(python_syntax="passed", complete_frozen_cohort="passed",
            all_source_words_covered_once="passed", max_1000_words="passed",
            truncated_json_rejected="passed"))
    (ROOT / "inputs/protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (CODE / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(dict(root=str(ROOT), papers=97, questions=970, chunks=len(chunks),
        source_words=protocol["source_words"], preflight=protocol["preflight"])))


if __name__ == "__main__":
    main()
