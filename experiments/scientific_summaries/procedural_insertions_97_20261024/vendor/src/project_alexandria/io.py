"""Document readers and atomic JSON artifact writing."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def read_documents(path: str, text_column: str = "text") -> List[Dict[str, str]]:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix in (".txt", ".md", ".rst"):
        return [{"text": source.read_text(encoding="utf-8"), "title": source.stem, "abstract": ""}]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF input requires: pip install project-alexandria[pdf]") from error
        reader = PdfReader(str(source))
        return [{
            "text": "\n\n".join(page.extract_text() or "" for page in reader.pages),
            "title": str(reader.metadata.title or source.stem) if reader.metadata else source.stem,
            "abstract": "",
        }]
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else [value]
        return [_normalize_record(record, text_column) for record in values]
    if suffix == ".jsonl":
        return [
            _normalize_record(json.loads(line), text_column)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        with source.open(encoding="utf-8", newline="") as handle:
            return [_normalize_record(row, text_column) for row in csv.DictReader(handle)]
    raise ValueError("unsupported input type: {}".format(suffix or "no extension"))


def _normalize_record(record: Dict[str, Any], text_column: str) -> Dict[str, str]:
    if text_column not in record:
        raise ValueError("input record is missing text column {!r}".format(text_column))
    return {
        "text": str(record[text_column] or ""),
        "title": str(record.get("title") or ""),
        "abstract": str(record.get("abstract") or ""),
    }


def write_json_atomic(path: str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
