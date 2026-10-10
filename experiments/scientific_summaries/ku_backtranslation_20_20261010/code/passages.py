"""Select source-copy sentence windows and patch structured narrative only."""
import copy
import re
from ngram_overlap import SourceIndex, fragments, STRUCTURAL, EVIDENCE, METADATA


def sentence_spans(text, max_words=75):
    ends = [m.end() for m in re.finditer(r'[.!?]\s+(?=[A-Z\u0391-\u03a9(\[“"\'])|\n\s*\n', text)]
    spans=[];start=0
    for end in ends+[len(text)]:
        words=list(re.finditer(r'\S+',text[start:end]))
        for i in range(0,len(words),max_words):
            a=start if i==0 else start+words[i].start()
            b=end if i+max_words>=len(words) else start+words[i+max_words].start()
            if text[a:b].strip():spans.append((a,b))
        start=end
    return spans


def select(summary,source):
    index=SourceIndex(source);selected=[]
    for text,path,category in fragments(summary):
        if category!='narrative':continue
        whole=index.fragment(text,path,category)
        if whole['longest_contiguous_match_words']<6:continue
        spans=sentence_spans(text)
        flagged=set()
        for i,(start,end) in enumerate(spans):
            own=index.fragment(text[start:end],path,category)
            if own['longest_contiguous_match_words']>=6:flagged.add(i)
            if i:
                combined=index.fragment(text[spans[i-1][0]:end],path,category)
                if combined['longest_contiguous_match_words']>=6 and own['longest_contiguous_match_words']<6:
                    flagged.update([i-1,i])
        expanded=sorted(flagged|{i-1 for i in flagged if i})
        groups=[]
        for i in expanded:
            if groups and len(groups[-1])<2 and i==groups[-1][-1]+1:groups[-1].append(i)
            else:groups.append([i])
        for group in groups:
            i=group[0];start=spans[i][0];end=spans[group[-1]][1]
            previous=text[spans[i-1][0]:spans[i-1][1]] if i else ''
            selected.append(dict(path=path,fragment=text,start=start,end=end,text=text[start:end],
                                 preceding_context=previous,source_match=index.fragment(text[start:end],path,category)))
    return selected


def patch(summary, replacements):
    """Replacement keys are (audit path, original string); proof/meta unchanged."""
    def replace(text,path,category):
        return replacements.get((path,text),text) if category=='narrative' else text
    def walk(value,path,category):
        if isinstance(value,str):return replace(value,path,category)
        if isinstance(value,list):return [walk(v,f'{path}[{i}]',category) for i,v in enumerate(value)]
        if isinstance(value,dict):
            output={}
            for key,item in value.items():
                if key in EVIDENCE:newkey=key;newvalue=walk(item,path+'.'+key,'evidence')
                elif key in METADATA:newkey=key;newvalue=walk(item,path+'.'+key,'metadata')
                elif key in STRUCTURAL:newkey=key;newvalue=walk(item,path+'.'+key,category)
                else:
                    newkey=replace(key,path+'.<statement>',category)
                    newvalue=walk(item,path+'.<evidence>','evidence' if category!='metadata' else 'metadata')
                if newkey in output:raise ValueError('Replacement would collide with an existing JSON key')
                output[newkey]=newvalue
            return output
        return copy.deepcopy(value)
    output={}
    for key,value in summary.items():
        if key=='top_influential_citations' and isinstance(value,list):
            citations=[]
            for i,citation in enumerate(value):
                if not isinstance(citation,dict):citations.append(copy.deepcopy(citation));continue
                item={}
                for k,v in citation.items():
                    if k=='quotes':item[k]=walk(v,f'{key}[{i}].quotes','evidence')
                    elif k in STRUCTURAL:item[k]=walk(v,f'{key}[{i}].{k}','metadata' if k in METADATA else 'narrative')
                    else:item[k]=walk(v,f'{key}[{i}].<justification>','narrative')
                citations.append(item)
            output[key]=citations
        else:output[key]=walk(value,key,'metadata' if key in METADATA else 'narrative')
    return output


def assemble(summary,windows,texts,accepted=None):
    groups={}
    for i,w in enumerate(windows):
        if accepted is not None and not accepted[i]:continue
        groups.setdefault((w['path'],w['fragment']),[]).append((w['start'],w['end'],texts[i]))
    replacements={}
    for key,changes in groups.items():
        value=key[1];last=len(value)
        for start,end,text in sorted(set(changes),reverse=True):
            if end>last:raise ValueError('Overlapping replacement windows')
            value=value[:start]+text.rstrip()+' '+value[end:];last=start
        replacements[key]=value
    return patch(summary,replacements)
