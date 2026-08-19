# Pilot result artifacts

`qwen38/` contains source-free prediction records from the 2026-08-19 diagnostic pilot. Each row
stores only a document SHA-256, original dataset row index, question index, gold letter, and the
three predicted letters. Source texts and question wording remain in the released Parquets.

These artifacts used Qwen3.8 as both extractor and answerer and are not a historical fixed-judge
reproduction. Exact configuration, results, caveats, and walltimes are documented in
[`docs/results/qwen38-mcq-pilot.md`](../../docs/results/qwen38-mcq-pilot.md).
