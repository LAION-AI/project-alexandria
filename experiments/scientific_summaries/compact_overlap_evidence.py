"""Keep a small review artifact in GitHub; retain complete spans in the Data1 audit."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);args=parser.parse_args()
    with (args.audit/'per_summary.jsonl').open() as source,gzip.open(args.audit/'longest_examples.jsonl.gz','wt',compresslevel=6) as out:
        for line in source:
            row=json.loads(line);a=row['audit'];examples={}
            for category in ['narrative','evidence','metadata']:
                matches=[dict(path=f['path'],**s) for f in a['fragments'] if f['category']==category for s in f['longest_examples']]
                examples[category]=sorted(matches,key=lambda x:-x['words'])[:3]
            value={k:row[k] for k in ['condition','document_id','source_sha256','summary_sha256','representation','locations'] if k in row}
            if 'locations' not in value:value['locations']=[row['location']]
            value.setdefault('representation','schema_object')
            value.update(audit_version=a['audit_version'],categories=a['categories'],longest_examples=examples,
                expanded_word_diagnostic=a.get('expanded_word_diagnostic'))
            out.write(json.dumps(value,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
