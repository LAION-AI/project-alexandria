"""Pinned, resumable 200-summary back-translation experiment."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/e/fscratch/reformo/schuhmann1/scientific-backtranslation-200-20261005')
BASE = ROOT.parent
PYTHON = BASE / 'scientific-distillation-1000/env/bin/python'
PINNED = BASE / 'scientific-ornith-dflash-eval-97/inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a/experiments/scientific_summaries'
REFERENCE = BASE / 'scientific-gemma-r128-fp8-20261005'
TEST_SHA = 'a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261'
ARMS = ['windy_greedy', 'windy_beam4', 'translategemma']
NAMES = {'windy_greedy': 'WindyTranslate EN→DE→EN / greedy',
         'windy_beam4': 'WindyTranslate EN→DE→EN / beam 4',
         'translategemma': 'TranslateGemma 4B EN→DE→EN / greedy'}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    temporary.replace(path)
