"""Offline checks for the paired DFlash pilot; no model downloads or GPU writes."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries'
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('dflash_pilot', ROOT / 'benchmark_ornith_dflash.py')
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_selection_is_balanced_deterministic_and_ignores_input_order():
    papers = [dict(document_id='%s_%s' % (source, i), viewer_config=source)
              for source in ('arxiv', 'bethgelab') for i in range(15)]
    eligible = {p['document_id'] for p in papers}
    chosen = pilot.select_papers(papers, eligible)
    assert chosen == pilot.select_papers(list(reversed(papers)), eligible)
    assert len({p['document_id'] for p in chosen}) == 20
    assert [p['viewer_config'] for p in chosen] == ['arxiv', 'bethgelab'] * 10
    with pytest.raises(ValueError, match='Insufficient'):
        pilot.select_papers(papers, {'arxiv_0'})


def test_server_commands_differ_only_by_dflash_configuration():
    args = argparse.Namespace(server_python='python', cache_dir='/tmp/example', concurrency=4)
    base = pilot.server_command(args, 'baseline')
    spec = pilot.server_command(args, 'dflash')
    assert spec[:-2] == base
    config = json.loads(spec[-1])
    assert config['method'] == 'dflash'
    assert config['revision'] == pilot.DRAFT_REV
    assert config['attention_backend'] == 'FLASH_ATTN'
    assert config['num_speculative_tokens'] == 8
    assert base[base.index('--tensor-parallel-size') + 1] == '2'


def test_speculative_metrics_sum_engines_and_ignore_metadata():
    raw = '# HELP x\nvllm:spec_decode_num_draft_tokens_total{engine="0"} 20\n' \
          'vllm:spec_decode_num_draft_tokens_total{engine="1"} 30\n' \
          'vllm:spec_decode_num_draft_tokens_created 12345678\n'
    assert pilot.counters(raw) == {'vllm:spec_decode_num_draft_tokens_total': 50}


def test_sharded_journal_hydration_detects_tampering(tmp_path):
    record = dict(document_id='arxiv_1', attempts=[dict(response='full response', tasks=['proof'])])
    lean = pilot.save_record(record, tmp_path)
    assert 'response' not in lean['attempts'][0]
    assert pilot.hydrate(lean, tmp_path) == record
    path = tmp_path / lean['raw_journal_file']
    path.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        pilot.hydrate(lean, tmp_path)
