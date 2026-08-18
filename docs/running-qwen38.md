# Running Qwen3.8-27B on one GPU

The requested model is
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). A Q4_K_M file is
roughly 16 GB, so weights fit on a 24 GB RTX 3090, but KV cache and runtime overhead constrain long
contexts. Begin with a 32K context and reduce `--gpu-memory-utilization` or concurrency if another
display/process uses the card.

Current [vLLM GGUF documentation](https://docs.vllm.ai/en/latest/features/quantization/gguf.html)
describes support as experimental and moved to the external `vllm-gguf-plugin`. It recommends
using the tokenizer from the base model rather than converting the GGUF tokenizer.

```bash
python -m pip install 'vllm>=0.8' vllm-gguf-plugin

CUDA_VISIBLE_DEVICES=0 vllm serve \
  'unsloth/Qwen3.8-27B-GGUF:Q4_K_M' \
  --tokenizer Qwen/Qwen3.8-27B \
  --served-model-name qwen38 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

Then use `--backend openai --base-url http://127.0.0.1:8000/v1 --model qwen38`. Concurrent client
requests are continuously batched by vLLM. Alternatively, `--backend vllm` sends all parallel
prompts to vLLM’s offline `LLM.chat` batch API.

Alexandria disables Qwen thinking by default through `chat_template_kwargs` so that extraction
budgets are spent on schema output. Pass `--thinking` only for quality experiments with a larger
`--max-tokens` budget.

## Memory levers

- Prefer Q4_K_M (or IQ4_XS if validated) before lowering model context.
- Keep extraction prompts bounded; side context is 1,000 words, not the whole paper.
- Prefix caching benefits repeated few-shot instructions and document metadata.
- Lower `--max-model-len` when KV-cache allocation prevents startup.
- Lower extraction concurrency when the server reports preemption or out-of-memory failures.
- Speculative decoding can reduce latency but does not solve weight/KV capacity; validate output
  quality before using it for result runs.

The 27B test described in this repository used an already-running MixedInt4 AutoRound derivative,
not the linked GGUF file, because the two available 3090s were occupied by another user’s server
and the filesystem lacked enough free space for another model copy. This distinction must remain in
the run manifest.
