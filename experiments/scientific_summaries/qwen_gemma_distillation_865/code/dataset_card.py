"""English Hugging Face dataset documentation from the frozen release manifest."""
import json


def render_card(manifest, config, archives):
    domains='\n'.join('| '+name+' | '+str(count)+' |' for name,count in sorted(manifest['domains'].items()))
    teacher=manifest['teacher']
    parameters={key:config.get(key) for key in ['thinking','reasoning_effort','review_reasoning_effort',
        'temperature','top_p','top_k','max_context','max_output','review_chunk_size',
        'max_structural_repairs','max_semantic_repairs']}
    parameters.update(top_k=20,min_p=0,presence_penalty=0,repetition_penalty=1.0)
    return '\n'.join([
        '---','license: cc-by-4.0','language:','- en','task_categories:','- summarization',
        'tags:','- scientific-papers','- distillation','- reasoning','- qwen','- gemma',
        'configs:','- config_name: accepted_papers','  default: true','  data_files:',
        '  - split: train','    path: data/train.jsonl',
        '- config_name: generator_reasoning','  data_files:','  - split: train',
        '    path: conversations/generation_reasoning.jsonl',
        '- config_name: final_conditioned_reasoning','  data_files:','  - split: train',
        '    path: conversations/final_conditioned_reasoning.jsonl',
        '- config_name: validated_final_summaries','  data_files:','  - split: train',
        '    path: conversations/validated_final_summary.jsonl','---','',
        '# Qwen3.8-27B scientific summary distillation: 865 accepted papers','',
        '**865 completed, accepted papers from ten scientific domains.** The original target was '
        '1,000; the user explicitly stopped collection on 2026-10-04. This repository describes '
        'the exact frozen 865-paper cohort, not a completed 1,000-paper dataset. One in-flight '
        'accepted result was committed during job cancellation and included before freezing.','',
        '## Teacher and exact generation recipe','',
        f"- Teacher: [{teacher['model']}](https://huggingface.co/{teacher['model']}).",
        f"- Pinned checkpoint revision: `{teacher['revision']}`.",
        '- Precision: official FP8 checkpoint; vLLM native autoregressive decoding, **no DFlash**.',
        '- Hardware: JUPITER GH200; 16 independent one-GPU replicas on four nodes, eight concurrent paper workflows per replica.',
        '- Full untruncated dataset source text is supplied; existing dataset summaries are not supplied to the teacher.',
        '- Thinking is enabled: medium effort for generation and correction, low effort for review. Only actual emitted reasoning is stored.',
        '- Generation/correction sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1.0.',
        '- Context budget: 65,536 tokens; generation/correction output budget up to 24,576 tokens, including reasoning.',
        '- Seeds are deterministic per source, phase and bounded draft attempt. Every actual seed and request budget is retained in the original trace.',
        '- JSON-object output; streaming SSE with usage reporting. The actual request payload, response content, emitted reasoning, finish reason and usage are archived.',
        '- Collection used a shared domain work pool and at most three independently seeded complete drafts per source after bounded failures.',
        '- The initial allocation timed out; generation resumed from completed artifacts. Collection was then explicitly cancelled at the user request.',
        '', 'Additional frozen configuration:', '', '`'*3+'json',json.dumps(parameters,indent=2),'`'*3,'',
        'The authoritative per-call sampling parameters and dynamic budgets are in the trace records; '
        'the [run configuration](provenance/run_config.json), [runtime manifest](provenance/runtime_manifest.json), '
        '[checkpoint fingerprints](provenance/teacher_manifest.json) and source code record the implementation.','',
        '## Exact prompts','',
        '- [Complete summary system prompt](provenance/summary_system_prompt.txt): the original 19-field scientific-summary template plus the frozen source-only policy.',
        '- [Summary user template](provenance/summary_user_prompt.txt): `{paper_text}` is replaced by the complete source text.',
        '- [Quality review system prompt](provenance/quality_review_system_prompt.txt): statement-wise source review and fixed central-fact coverage.',
        '- [Original reference prompts and hashes](provenance/reference_prompts.json); [reference provenance](provenance/reference_manifest.json).',
        '- Structural and semantic correction prompts, draft inputs and feedback are retained verbatim in each original call and the frozen teacher code.',
        '',
        'Initial format retries can append the prior format error to the source-only user prompt. '
        'No draft, quality assessment or final accepted summary is added to generator-training inputs. '
        'The 19 fields cover title/authors, discipline and paper type, executive summary, research context, '
        'methods, results, claims, uncertainties/limitations, citation roles and takeaways. Literal proof '
        'fragments are constrained to at most five words. See the exact prompt for every field and rule.','',
        '## Acceptance, correction and source evidence','',
        'Every accepted final summary passes the original complete-schema, field-shape and literal '
        'source-evidence checks and the same-teacher source review. Reviews cover every enumerated '
        'summary statement in chunks, including executive-summary claims, quantitative claims and '
        'citation claims. Five to eight central source facts are frozen at the first review and '
        'reused across correction rounds. Acceptance is recomputed from validated scores, statement '
        'verdicts, issues and coverage; the teacher-proposed boolean alone is not accepted.',
        '',
        'The workflow permits up to five structural repairs and three semantic repairs per draft. '
        'Semantic correction returns the complete 19-field object and propagates changes across '
        'fields. Actual requests, proposed edits, failed attempts, review rounds, source evidence '
        'ledgers and emitted reasoning remain available.',
        '',
        '**These quality labels are same-teacher judgments, not independent scientific ground truth '
        'or human factual verification.** A literal quotation does not by itself establish entailment.',
        '',
        'Mean final teacher scores:', '', '`'*3+'json',json.dumps(manifest['quality_scores_mean'],indent=2),'`'*3,'',
        '## Generator reasoning versus corrected final reasoning','',
        f"**{manifest['final_reasoning_conditioned_papers']} of 865 final summaries were reached through "
        f"correction; only {manifest['final_reasoning_source_only_papers']} have an unchanged source-only final target.**",
        '',
        'The repository keeps three distinct, explicitly described training views:',
        '',
        '| View | Target and input | Used for the requested generator LoRAs? |',
        '| --- | --- | --- |',
        '| generator_reasoning | Actual original source-only generator output plus its matching emitted reasoning, from the successful accepted draft attempt | Yes |',
        '| final_conditioned_reasoning | Actual final summary and matching reasoning with the original draft/feedback input retained | No |',
        '| validated_final_summaries | Corrected, accepted final summary under a source-only input, without invented reasoning | No |',
        '',
        '**The generator targets are the original uncorrected outputs.** Acceptance and the final '
        'quality labels apply to the repaired final summaries, not automatically to those raw targets. '
        'Raw targets can contain errors later corrected by the pipeline. Original raw reasoning is '
        'never paired with a different corrected answer, and correction-dependent reasoning is never '
        'relabeled as source-only reasoning. No reasoning is fabricated or inferred from hidden states.',
        '',
        'Requested experiment: Gemma 4 12B IT, classic LoRA ranks 64 and 128, one epoch each, '
        'summary generator only. Both supervise the original generator reasoning and final response '
        'using the native Gemma thought-channel template and assistant-only token masking. '
        'No reviewer or corrector is trained. No generation outputs from Ornith enter these Qwen-trained LoRAs.','',
        '## Scientific domains','', '| Domain | Accepted papers |','| --- | ---: |',domains,'',
        'The early stop leaves an approximately balanced cohort; it is not exactly 100 papers per domain.','',
        '## Evaluation separation','',
        'All 865 source papers are excluded from the frozen external **97-paper / 970-MCQ** test set. '
        'Checks cover paper identifiers/DOIs, normalized and near-duplicate titles, exact and normalized '
        'source hashes and substantial shared source-text spans. The [overlap audit](evaluation_overlap_audit.json) '
        'reports zero matching papers. Evaluation questions, answer options, gold answers, rationales '
        'and QA evidence are never used as teacher or training inputs. The release contains no evaluation questions.',
        '',
        'All 865 papers are in this release\'s training split. There is no internal validation/test split '
        'in this user-stopped release; adapter ranks and one epoch were specified before external '
        'evaluation. The external test remains separate and is not used for checkpoint selection. '
        'The later LoRA evaluation includes an untrained Gemma baseline with exactly the same '
        'prompt, reasoning mode and sampling settings to measure adapter gains fairly.','',
        '## Files and reproducibility','',
        '- `data/train.jsonl`: dataset-viewer friendly paper records, raw and accepted summaries, emitted reasoning and final quality assessments.',
        '- `papers_and_traces.jsonl`: detailed source records, full generator/final calls, accepted summaries, correction events and review rounds.',
        '- `conversations/`: the three precisely labeled views above, with the original input/target conversations.',
        '- `traces/<document_id>.tar.gz`: all preserved raw calls, interrupted streams, failed attempts and journals belonging to each accepted source paper.',
        '- `trace_archives.jsonl`: per-archive SHA256 hashes and byte counts.',
        '- `provenance/`: exact prompts, configuration, model/runtime/source manifests and frozen teacher/experiment code.',
        '- `manifest.json`, `evaluation_overlap_audit.json` and `SHA256SUMS`: cohort identity, separation checks and file integrity.',
        '',
        f"The {len(archives)} per-paper trace archives contain {sum(a['bytes'] for a in archives):,} compressed bytes. "
        'Models, token caches, access credentials and unrelated incomplete source papers are excluded.',
        '',
        '## Source attribution and limitations','',
        '[LAION Scientific-Summaries](https://huggingface.co/datasets/laion/Scientific-Summaries), '
        'pinned revision `232547b7b938a6588ce5d9d7faa96384860b7a16`. Full source selection '
        'and attribution are recorded in [selection_manifest.json](provenance/selection_manifest.json). '
        'Dataset annotations follow CC-BY-4.0 attribution; underlying source-paper rights remain '
        'with the original authors and publishers. The dataset text may contain OCR artifacts or '
        'incomplete sections and is not claimed to match every publisher PDF in full.',
        '',
        'Qwen outputs and source review remain fallible. Reasoning traces can contain errors, '
        'speculation, repetition or intermediate conclusions not present in the final summary. '
        'No independently established LoRA improvement is claimed in this release; that is the '
        'purpose of the separate matched evaluation.',''
    ])
