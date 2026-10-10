# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
"""Standalone scientific report and complete five-paper KU examples PDF."""
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

from common import ROOT, load
from ku_pipeline import KU_CONDITIONS
from build_site import SITE, LIVE, label, slug, HF12, HF4, TEACHER

def register_fonts():
    fonts=list(Path('/usr/share/fonts').rglob('*.ttf'))
    for name,filename in [('DV','DejaVuSans.ttf'),('DV-Bold','DejaVuSans-Bold.ttf'),('DV-Mono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(next(p for p in fonts if p.name==filename))))
    pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')

def main():
    register_fonts();metrics=load(ROOT/'outputs/metrics.json');copy=load(ROOT/'outputs/copy_overlap.json')
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText','Title','Heading1','Heading2','Heading3']:
        styles[name].fontName='DV-Bold' if name in ['Title','Heading1','Heading2','Heading3'] else 'DV'
    styles.add(ParagraphStyle('small',fontName='DV',fontSize=8,leading=11,spaceAfter=5,textColor=colors.HexColor('#526473')))
    styles.add(ParagraphStyle('cell',fontName='DV',fontSize=7.2,leading=10,spaceAfter=0))
    styles.add(ParagraphStyle('value',fontName='DV-Mono',fontSize=8,leading=11,spaceAfter=4,splitLongWords=True))
    styles['BodyText'].fontSize=9;styles['BodyText'].leading=13
    def p(text,style='BodyText',markup=False):
        return Paragraph(text if markup else html.escape(str(text)).replace('\n','<br/>'),styles[style])
    def table(headers,rows,widths):
        data=[[p(x,'cell') for x in headers]]+[[p(x,'cell') for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f1f5')),
                             ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),
                             ('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.35,colors.HexColor('#dce5eb'))]))
        return t
    def footer(canvas,doc):
        canvas.setFont('DV',7);canvas.setFillColor(colors.HexColor('#667888'))
        canvas.drawString(14*mm,9*mm,'Project Alexandria · KU back-translation · 20 papers · 10 October 2026')
        canvas.drawRightString(doc.pagesize[0]-14*mm,9*mm,str(doc.page))
    story=[p('Project Alexandria: Knowledge Units with guarded back-translation','Title'),
           p('20 disjoint papers · 200 frozen multiple-choice questions per condition · 28 conditions / 5,600 QA slots','small'),
           p('The same Qwen2.5-7B answerer uses each context. Generation has thinking disabled. Original failed generations remain wrong. English → German → English repair targets copied descriptive wording; graph identifiers, names, targets and metadata remain intact. No retraining or QA-guided repair selection.'),Spacer(1,8),
           p('QA comparison: unchanged raw KUs versus guarded paraphrasing','Heading2')]
    rows=[]
    for c in KU_CONDITIONS:
        a,b=metrics['scores'][c],metrics['scores'][c+'_bt'];d=metrics['paired_backtranslation_minus_raw'][c];lo,hi=d['confidence_interval_95'];name,words=label(c)
        rows.append([name,str(words),f'{100*a["accuracy"]:.1f}%\n{a["correct"]}/200',f'{100*b["accuracy"]:.1f}%\n{b["correct"]}/200',f'{100*d["point"]:+.1f} pp',f'{100*lo:+.1f} to {100*hi:+.1f} pp',str(b['invalid'])])
    story += [table(['KU generator','Source words','Raw QA','Guarded BT QA','Change','Paired 95% interval','Invalid'],rows,[230,60,70,85,65,100,50]),
              Spacer(1,8),p('Paired intervals resample whole papers in 10,000 bootstrap draws. An interval including zero does not establish a reliable accuracy change. These questions were teacher-authored and automatically source-validated, not independently human-curated.','small'),
              p('Full-paper, no-context and summary controls','Heading2')]
    names={'original':'Full original paper / ground-truth context','no_context':'No paper context','qwen_summary':'Qwen27 summary',
           'gemma4_base_summary':'Gemma E4B base summary','gemma4_r128_summary':'Gemma E4B KU LoRA used for summaries',
           'gemma12_base_summary':'Gemma12 base summary','gemma12_r128_summary':'Gemma12 KU LoRA used for summaries',
           'gemma12_previous_summary_lora':'Gemma12 earlier summary-trained LoRA rank128'}
    rows=[]
    for c,name in names.items():
        s=metrics['scores'][c];lo,hi=s['confidence_interval_95'];rows.append([name,f'{100*s["accuracy"]:.1f}%',f'{s["correct"]}/200',f'{100*lo:.1f}–{100*hi:.1f}%',str(s['invalid'])])
    story += [table(['Context','QA accuracy','Correct','95% interval','Invalid'],rows,[330,75,65,115,75]),PageBreak(),
              p('Copying, fidelity and measured compute','Heading1'),
              p('The primary audit counts all factual KU strings separately, preserving the previous benchmark’s definition. Document title/author metadata are separate. The whole source is used: ≤5 copied normalized whitespace words passes, exactly six is borderline, ≥7 is flagged. Reasoning, prompts and questions are excluded.'),Spacer(1,8)]
    rows=[]
    for c in KU_CONDITIONS:
        a,b=copy['conditions'][c]['narrative'],copy['conditions'][c+'_bt']['narrative'];name,words=label(c)
        rows.append([name,str(words),f'{100*a["mean_covered_fraction_six_plus"]:.2f}%',f'{100*b["mean_covered_fraction_six_plus"]:.2f}%',str(b['maximum_contiguous_match_words']),f'{b["strict_five_word_pass"]}/20',f'{b["violation_seven_plus"]}/20'])
    story += [table(['Generator','Words','Raw coverage ≥6','BT coverage ≥6','BT max run','BT ≤5 pass','BT ≥7 flag'],rows,[230,55,90,90,65,65,65]),Spacer(1,10)]
    q=metrics['quality'];t=metrics['translation'];a=metrics['allocation']
    story += [p(f"Repair inserted {q['inserted_windows']:,} windows in {q['changed_documents']} / 200 KU document versions. Maximum two round-trips. Rejected edits retain the original field. Acceptance requires bidirectional NLI ≥0.9, unchanged number/unit/formula signatures and no copy run >5 inside the candidate window."),
              p(f"Raw candidates produced {q['raw_attempts_with_critical_flags']} conservative critical-value flags; guarded final documents have {q['guarded_global_critical_failures']} retained critical-value flags. This is an automated signature/NLI result rather than independent human verification of every claim."),
              p(f"Active translation: {t['active_translation_seconds']:.2f} seconds / {t['active_translation_gpu_hours']:.5f} GPU-hours. Full experiment: {a.get('reserved_gpu_hours',0):.4f} reserved GPU-hours on four GH200 GPUs, including startup, semantic checks, QA and idle roles. These costs cover the 200 cached KU document versions, not the original extraction or training."),
              Spacer(1,10),p('Models, profiles and complete examples','Heading2')]
    for repo,name in [(HF12,'Gemma12 KU adapter'),(HF4,'Gemma E4B KU adapter'),(TEACHER,'Qwen teacher')]:
        url='https://huggingface.co/'+repo;story.append(p(f'<link href="{url}">{name}: {url}</link>','small',True))
    story.append(p('Each KU adapter was trained at rank128, one epoch, on 391 Qwen27 completions from 97 papers using 1,000-word chunks. 500 and 1,000 words are two inference profiles for the same fixed weights, not separately trained adapters.'))
    story.append(p(f'<link href="{LIVE}">Interactive static reader: {LIVE}</link>','small',True))
    story += [PageBreak(),p('Attribution, style and sequential context','Heading1'),
              p('The paper defines entities, attributes, relationships, contextual notes and sentence MinHash. The current repository sequential prompt supplies up to ten earlier KUs to stabilize naming, but supplies no future text. Its context_summary describes the target under the current schema; it does not guarantee a separate retelling of preceding paragraphs.'),
              p('Source-registry title and authors are shown for attribution in the reader. The original document author field was not populated, although some entities carry author attributes. There is no dedicated style field. Reader links and adjacent summaries show what comes before/after without adding future facts to model outputs or the QA context.'),Spacer(1,10)]
    rows=[]
    for c in KU_CONDITIONS:
        m=metrics['metadata'][c];name,words=label(c)
        rows.append([f'{name} · {words}',str(m['exact_document_title_matches']),str(m['nonempty_document_authors']),str(m['author_attributes_present']),str(m['explicit_style_attributes_present']),f'{m["context_summary_present_units"]}/{m["units"]}',f'{m["sentence_minhash_present_units"]}/{m["units"]}'])
    story += [table(['Condition','Title /20','Author field /20','Author attrs /20','Style attrs /20','Context / KU','MinHash / KU'],rows,[265,60,70,70,70,60,65]),Spacer(1,10),
              p('Five example papers','Heading2')]
    examples=[json.loads(x) for x in (ROOT/'outputs/public_demo_examples.jsonl').read_text().splitlines()]
    defaults=[e for e in examples if e['condition']=='gemma12_r128_ku500_bt']
    for e in defaults:
        url=LIVE+e['document_id']+'/'+slug(e['condition'])+'.html'
        story.append(p(f'<link href="{url}">{html.escape(e["source_metadata"]["title"])}</link>','BodyText',True))
    SITE.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(SITE/'evaluation.pdf'),pagesize=landscape(A4),rightMargin=14*mm,leftMargin=14*mm,
                      topMargin=12*mm,bottomMargin=15*mm,title='Alexandria KU back-translation QA evaluation',author='LAION / Project Alexandria').build(story,onFirstPage=footer,onLaterPages=footer)
    # The second PDF includes all saved Gemma12 KUs for both chunk sizes,
    # rather than selecting a few attractive units from each paper.
    story=[p('Five complete scientific-paper Knowledge Unit extractions','Title'),
           p('Gemma 4 12B IT KU LoRA rank128, at 500 and 1,000 source words per chunk, with guarded TranslateGemma back-translation. All saved entities, attributes, relationships, context summaries, IDs and source references are retained. Paper bodies are not included. Source order is not a publication-date chronology.'),
           p('Attribution is source-registry metadata for display only; it was not added to QA. The copying diagnostic can remain flagged. See the separate evaluation PDF for scores and acceptance rules.'),PageBreak()]
    for default in defaults:
        for words in [500,1000]:
            e=next(e for e in examples if e['document_id']==default['document_id'] and e['condition']==f'gemma12_r128_ku{words}_bt')
            story += [p(e['source_metadata']['title'],'Heading1'),p(f'{words:,}-word chunks · '+e['status'],'small'),
                      p('Source authors: '+'; '.join(e['source_metadata']['authors'])),
                      p('KU document title: '+e['result']['title']),p('KU document author field: '+(e['result'].get('author') or 'not populated'),'small')]
            for u in e['result']['knowledge_units']:
                story += [Spacer(1,10),p(f'KU {u["chunk_index"]+1}: source words {u["source"]["start_word"]+1}–{u["source"]["end_word"]}','Heading2'),p(u['context_summary'])]
                for entity in u['entities']:
                    story += [Spacer(1,7),p(entity['name'],'Heading3'),p('Type: '+entity['entity_type']+' · ID: '+entity['entity_id'],'small')]
                    if entity['aliases']:story.append(p('Aliases: '+', '.join(entity['aliases']),'small'))
                    for k,v in entity['attributes'].items():
                        value=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)
                        story.append(p(str(k)+': '+value,'value'))
                    for rel in entity['relationships']:
                        story.append(p(rel['predicate']+' → '+rel['target'],'value'))
                        if rel['target_id']:story.append(p('Target ID: '+rel['target_id'],'small'))
                        for k,v in rel['attributes'].items():story.append(p(str(k)+': '+(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)),'value'))
                story.append(p('Source reference: '+json.dumps(u['source'],ensure_ascii=False),'small'))
                if u['extraction_warnings']:story.append(p('Extraction warnings: '+json.dumps(u['extraction_warnings'],ensure_ascii=False),'small'))
            story.append(PageBreak())
    if isinstance(story[-1],PageBreak):story.pop()
    SimpleDocTemplate(str(SITE/'examples.pdf'),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=15*mm,
                      bottomMargin=17*mm,title='Five complete Alexandria KU examples',author='LAION / Project Alexandria').build(story,onFirstPage=footer,onLaterPages=footer)
    print('BUILT_PDFS',str(SITE/'evaluation.pdf'),str(SITE/'examples.pdf'),flush=True)

if __name__=='__main__':main()
