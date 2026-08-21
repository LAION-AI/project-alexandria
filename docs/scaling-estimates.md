# Throughput and JUPITER planning estimates

These are capacity-planning estimates, not GH200 benchmarks. They are anchored to the completed
Qwen3.8-27B parallel extraction on this host and deliberately reported as ranges. Benchmark the
exact vLLM/TensorRT-LLM build, quantization, prompt lengths, and output cap on a JUPITER allocation
before reserving a large run.

## Measured local baseline

`Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` ran with vLLM on two RTX 3090 GPUs. Physics and Medical
were submitted concurrently to the same continuously batching server. The 97 Physics documents
recorded 15,026 seconds and the 99 Medical documents 37,986 seconds. Since the jobs overlapped, the
capacity-relevant total is approximately 10.55 wall-hours on two GPUs: **21.10 RTX-3090 GPU-hours
for 196 papers, or 0.108 GPU-hour per paper**. The observed service throughput was about 18.6
papers/hour. Paper length and generated KU length varied substantially, so domain-specific averages
should not be treated as constants.

The fixed Qwen2.5-7B answer pass is small by comparison: approximately 15--20 wall-minutes for the
same 196 papers on the two GPUs. The extraction projection below excludes that optional evaluation.

## GH200 estimate

One GH200 can hold either proposed model and avoids tensor parallelism. In the absence of an exact
Alexandria GH200 measurement, planning assumes one GH200 sustains **3--7 times** the throughput of
the measured two-3090 service for the 27B MixedInt4 workload. This wide factor is intentional:
long-prefill/short-decode mixtures, quantized kernels, canonicalization calls, and scheduler settings
can move the result materially. For a 9B model, the table assumes a further **2--3 times** throughput
gain at similar output lengths.

| Model | One average paper | 1 million papers | 100 million papers |
|---|---:|---:|---:|
| Qwen3.8 27B MixedInt4 | 0.015--0.036 GH200-hours (0.9--2.2 min) | 15,000--36,000 GH200-hours | 1.5--3.6 million GH200-hours |
| Qwen3.8 9B MixedInt4 estimate | 0.005--0.018 GH200-hours (0.3--1.1 min) | 5,000--18,000 GH200-hours | 0.5--1.8 million GH200-hours |

JUPITER's Booster has roughly 6,000 nodes with four GH200 Superchips per node, or about 24,000
accelerators ([ECMWF overview](https://www.ecmwf.int/en/about/media-centre/news/2025/reaching-jupiter-ecmwf-celebrates-first-european-exascale),
[JSC configuration](https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html)). At impossible
100% whole-machine utilization and perfect linear scaling, the ranges above correspond to:

| Model | 1 million papers on 24,000 GH200s | 100 million papers on 24,000 GH200s |
|---|---:|---:|
| 27B | 0.6--1.5 hours | 64--150 hours (2.7--6.3 days) |
| 9B | 0.2--0.8 hours | 21--75 hours (0.9--3.1 days) |

Real wall time will be higher. At 60--80% end-to-end utilization, multiply those whole-machine
figures by about 1.25--1.67. Storage reads, JSON writes, preprocessing, stragglers, failed jobs,
scheduler allocation, and canonicalization all reduce scaling efficiency. A million-paper pilot
should therefore begin with several hundred to several thousand GH200s and report distributions of
input tokens, output tokens, chunks per paper, and retry counts rather than only papers/hour.

## How to replace the estimate with a benchmark

1. Freeze the model revision, quantization, vLLM version, prompt version, output limits, and seed.
2. Sample at least 1,000 papers stratified by domain and source length.
3. Run enough independent replicas to saturate each GH200 without tensor parallelism.
4. Record successful papers/hour, prompt and generation tokens/second, retry rate, and p50/p95
   paper latency.
5. Recompute GPU-hours from total allocated accelerator time, including failed and retried work.

