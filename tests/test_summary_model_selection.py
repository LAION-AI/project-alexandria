"""Offline regression tests for selecting a summary model without protocol changes."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries'
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('summary_selection_runner', ROOT / 'run_summary_comparison.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_default_order_and_explicit_qwen_selection():
    assert [m['name'] for m in runner.selected_models()] == ['qwen27b', 'ornith15_9b', 'qwen35_9b']
    selected = runner.selected_models(['qwen35_9b'])
    assert [m['name'] for m in selected] == ['qwen35_9b']
    assert selected[0]['concurrency'] == 4
    assert selected[0]['runtime'] == 'llama.cpp'
    assert len(runner.MODELS) == 3


def test_repeated_selections_are_deduplicated_in_original_order():
    selected = runner.selected_models(['qwen35_9b', 'qwen27b', 'qwen35_9b'])
    assert [m['name'] for m in selected] == ['qwen27b', 'qwen35_9b']
    with pytest.raises(ValueError, match='Unknown summary model'):
        runner.selected_models(['unknown'])
