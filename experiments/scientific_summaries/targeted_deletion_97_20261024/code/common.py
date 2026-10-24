"""Portable scientific-value guards shared with the measured runtime."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor'))


def load(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def named_values(text):
    return set(re.findall(r'\b[A-Z][A-Z0-9]{1,}(?:[-_][A-Za-z0-9]+)*\b|\b[A-Z][a-z]?(?:\+[A-Z][a-z]?)+\b',text))


def negations(text):
    value=re.sub(r"n['’]t\b",' not',text.casefold())
    return len(re.findall(r'\b(?:not|no|never|without|neither|nor)\b',value))
