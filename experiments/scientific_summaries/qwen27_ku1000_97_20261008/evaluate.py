"""Same fixed historical Qwen2.5 judge: no context, full source, KUs, summary."""
import concurrent.futures
import time

from common import ROOT, TracedBackend, digest, load, sources, write
from project_alexandria.experiments.mcq import historical_answer_prompt, extract_historical_choice
from project_alexandria.experiments.reproduce import JUDGE_SYSTEM_PROMPT, knowledge_unit_context
from project_alexandria.schema import DocumentResult

CONDITIONS = ["no_context", "original", "knowledge_units_1000", "gemma_r128_fp8_summary"]


def score_document(paper, question_paper, ku, baseline, backend):
    docid = paper["document_id"]
    output = ROOT / "outputs/qa" / (docid + ".json")
    if output.exists():
        return load(output)
    assert question_paper["document_id"] == docid
    assert question_paper["fulltext_sha256"] == paper["fulltext_sha256"]
    questions = question_paper["questions"]
    assert len(questions) == 10
    assert baseline["document_id"] == docid and baseline["source_sha256"] == paper["fulltext_sha256"]
    contexts = {"no_context": "", "original": paper["fulltext"],
        "knowledge_units_1000": knowledge_unit_context(DocumentResult.from_dict(ku["result"])),
        "gemma_r128_fp8_summary": baseline["judge_context"]}
    started = time.monotonic()

    def answer(question, condition):
        backend.local.document_id = docid
        prompt = historical_answer_prompt(question["formatted_question"], contexts[condition])
        record = dict(question_index=question["question_index"], condition=condition,
            prompt_sha256=digest(prompt), attempts=[], prediction=None,
            context_limit=32768, generation_failed=False)
        if condition == "knowledge_units_1000" and ku["status"] != "generated":
            record.update(generation_failed=True, error="Incomplete KU extraction; slot retained and scored wrong")
            return record
        if condition == "gemma_r128_fp8_summary" and baseline["status"] != "generated":
            record.update(generation_failed=True, error="Missing frozen baseline summary")
            return record
        for _ in range(5):
            try:
                response = backend.generate(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=100)
                trace = dict(backend.local.last)
                record["attempts"].append(trace)
                record["prompt_tokens"] = trace["prompt_tokens"]
                record["prediction"] = extract_historical_choice(response)
                if record["prediction"] is not None:
                    break
            except ValueError as error:
                record["error"] = str(error)
                break
        return record

    slots = [(q, c) for q in questions for c in CONDITIONS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(answer, q, c) for q, c in slots]
        records = [future.result() for future in futures]
    rows = [dict(question_index=q["question_index"], gold=q["answer"],
        formatted_question_sha256=digest(q["formatted_question"]),
        predictions={c: records[CONDITIONS.index(c)+i*len(CONDITIONS)]["prediction"] for c in CONDITIONS},
        responses={c: records[CONDITIONS.index(c)+i*len(CONDITIONS)] for c in CONDITIONS})
        for i, q in enumerate(questions)]
    value = dict(document_id=docid, source_sha256=paper["fulltext_sha256"],
        ku_status=ku["status"], rows=rows, elapsed_seconds=time.monotonic()-started,
        context_sha256={c: digest(text) for c, text in contexts.items()})
    write(output, value)
    print("QA_DOCUMENT_COMPLETE", docid,
        {c: sum(r["predictions"][c] == r["gold"] for r in rows) for c in CONDITIONS}, flush=True)
    return value


def run(endpoint, abort):
    papers = sources()
    test = load(ROOT / "inputs/testset.json")
    manifest = load(ROOT / "inputs/protocol.json")
    if digest((ROOT / "inputs/testset.json").read_text()) != manifest["testset_sha256"]:
        raise ValueError("Frozen test set changed")
    by_id = {p["document_id"]: p for p in test["papers"]}
    baseline = {p["document_id"]: p for p in load(ROOT / "inputs/gemma_fp8_baseline.json")}
    assert set(by_id) == {p["document_id"] for p in papers} == set(baseline)
    backend = TracedBackend(endpoint, "qwen25", "qa", max_tokens=100,
        temperature=.5, concurrency=4, penalties=1.05, context_limit=32768)
    completed = set()
    started = time.monotonic()
    while len(completed) < 97:
        if abort.is_set():
            raise RuntimeError("Generation/orchestration aborted; QA checkpoints retained")
        progressed = False
        for paper in papers:
            docid = paper["document_id"]
            if docid in completed:
                continue
            checkpoint = ROOT / "outputs/knowledge_units" / docid / "document.json"
            if not checkpoint.exists():
                continue
            score_document(paper, by_id[docid], load(checkpoint), baseline[docid], backend)
            completed.add(docid)
            progressed = True
            write(ROOT / "outputs/qa_progress.json", dict(completed_papers=len(completed),
                total_papers=97, updated_unix=time.time(), elapsed_seconds=time.monotonic()-started))
        if not progressed:
            time.sleep(2)
    report = dict(complete=True, papers=97, questions_per_condition=970,
        conditions=CONDITIONS, elapsed_seconds=time.monotonic()-started,
        fresh_judging_all_conditions=True, historical_prompt_and_parser=True,
        no_qa_feedback_into_extraction=True)
    write(ROOT / "outputs/qa_complete.json", report)
    return report
