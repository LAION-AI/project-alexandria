"""One-epoch, generator-only Gemma12B LoRA with real teacher reasoning supervision."""
import argparse
import datetime
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import LoraConfig, get_peft_model
from cut_cross_entropy import linear_cross_entropy

from common import ROOT, TARGET_MODULES, load, write, digest


class TextLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, labels):
        base = self.model.get_base_model()
        output = base.model(input_ids=ids, use_cache=False, return_dict=True)
        softcap = getattr(base.config.get_text_config(), 'final_logit_softcapping', None)
        return linear_cross_entropy(output.last_hidden_state[:, :-1, :], base.lm_head.weight,
                                    labels[:, 1:], ignore_index=-100, softcap=softcap,
                                    reduction='mean', filter_eps=None)


def max_rank(value):
    tensor = torch.tensor(float(value), device='cuda')
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def row_at(path, offset):
    with open(path, 'rb') as handle:
        handle.seek(offset)
        return json.loads(handle.readline())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lora-rank', type=int, choices=[64, 128], required=True)
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 8
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    seed = 20261004
    torch.manual_seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = load(ROOT / 'training/manifest.json')
    cohort = load(ROOT / 'inputs/frozen_cohort.json')
    ready = load(ROOT / 'outputs/core_ready.json')
    assert ready['training_manifest_sha256'] == digest(ROOT / 'training/manifest.json')
    path = ROOT / 'training/gemma12_generation_reasoning.jsonl'
    assert digest(path) == manifest['sha256']
    approved = {p['document_id'] for p in cohort['papers']}
    offsets, identifiers = [], []
    with path.open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            assert row['document_id'] in approved and row['reasoning_in_target']
            assert row['teacher_call_id'].startswith('generation')
            assert len(row['input_ids']) == len(row['labels']) <= 65536
            offsets.append(offset)
            identifiers.append(row['document_id'])
    assert len(offsets) == len(set(identifiers)) == cohort['accepted_papers'] == 865
    order = list(range(len(offsets)))
    random.Random(seed).shuffle(order)
    steps = math.ceil(len(order) / world)
    target = ROOT / 'outputs' / ('train-r' + str(args.lora_rank))
    target.mkdir(exist_ok=True)
    start = time.monotonic()
    base = AutoModelForImageTextToText.from_pretrained(
        ROOT / 'models/gemma-4-12b-it', local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='flex_attention')
    for name in ['visual', 'vision_tower', 'audio_tower', 'embed_vision', 'embed_audio']:
        if hasattr(base.model, name):
            setattr(base.model, name, None)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                         lora_dropout=.05, bias='none', target_modules=TARGET_MODULES,
                                         task_type=None))
    model.cuda()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert parameters and all('lora_A.' in name or 'lora_B.' in name for name in names)
    ddp = DistributedDataParallel(TextLoss(model), device_ids=[0], broadcast_buffers=False)
    peak_lr = 2e-5
    optimizer = torch.optim.AdamW(parameters, lr=peak_lr, weight_decay=.01)
    warmup = max(1, math.ceil(steps * .05))
    result = dict(base_model='google/gemma-4-12B-it', model_revision=manifest['tokenizer_revision'],
                  lora_rank=args.lora_rank, lora_alpha=2 * args.lora_rank, dropout=.05,
                  epochs=1, task='source-only generator reasoning and answer',
                  reviewer_training=False, corrector_training=False, source_papers=865,
                  precision='BF16', nodes=2, gpus=world, microbatch=1, effective_batch=world,
                  optimizer='AdamW', learning_rate=peak_lr, weight_decay=.01,
                  schedule='5% warmup then cosine decay', optimizer_steps=steps, warmup_steps=warmup,
                  seed=seed, gradient_checkpointing=True, attention='flex_attention',
                  loss='assistant-only actual reasoning plus original full summary; cut_cross_entropy filter_eps=None',
                  max_length=65536, truncation=False, trainable_parameters=sum(p.numel() for p in parameters),
                  training_manifest_sha256=digest(ROOT / 'training/manifest.json'),
                  training_file_sha256=digest(path), job_id=os.environ['SLURM_JOB_ID'],
                  trainable_module_names=names, status='training')
    dist.barrier()
    result['load_and_setup_seconds'] = max_rank(time.monotonic() - start)
    if rank == 0:
        write(target / 'config.json', result)
    checkpoint_step = 0
    checkpoint = target / 'latest_checkpoint.json'
    if checkpoint.exists():
        saved = load(checkpoint)
        assert saved['training_file_sha256'] == digest(path)
        weights = torch.load(saved['optimizer_state'], map_location='cpu', weights_only=False)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(saved['adapter_file']))
        optimizer.load_state_dict(weights)
        checkpoint_step = saved['completed_steps']
    timings = []
    seen = []
    for step in range(checkpoint_step, steps):
        index = step * world + rank
        real = index < len(order)
        source_index = order[index] if real else order[-1]
        row = row_at(path, offsets[source_index])
        ids = torch.tensor([row['input_ids']], device='cuda', dtype=torch.long)
        labels = torch.tensor([row['labels']], device='cuda', dtype=torch.long)
        ntarget = int((labels[:, 1:] != -100).sum()) if real else 0
        total_targets = torch.tensor(ntarget, device='cuda', dtype=torch.long)
        dist.all_reduce(total_targets)
        if step < warmup:
            lr = peak_lr * (step + 1) / warmup
        else:
            lr = peak_lr * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = ddp(ids, labels)
            weighted = loss * (ntarget * world / total_targets.item())
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite generator loss')
        weighted.backward()
        if real and not any(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().max() > 0 for p in parameters):
            raise ValueError('No finite nonzero adapter gradient')
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = max_rank(time.monotonic() - tick)
        peak = max_rank(torch.cuda.max_memory_allocated() / 2**30)
        reduced = weighted.detach().clone()
        dist.all_reduce(reduced)
        reduced /= world
        if real:
            seen.append(row['document_id'])
        timings.append(seconds)
        if rank == 0:
            observation = dict(step=step + 1, steps=steps, loss=float(reduced), learning_rate=lr,
                               seconds_max_rank=seconds, peak_allocated_gib_max_rank=peak,
                               epoch=(step + 1) / steps, remaining_compute_seconds=(steps - step - 1) * sum(timings[-5:]) / len(timings[-5:]))
            with (target / 'timings.jsonl').open('a') as handle:
                handle.write(json.dumps(observation) + '\n')
            print(json.dumps(observation), flush=True)
        if (step + 1) % 20 == 0 or step + 1 == steps:
            dist.barrier()
            if rank == 0:
                folder = target / ('checkpoint-step' + str(step + 1))
                model.save_pretrained(folder, safe_serialization=True)
                torch.save(optimizer.state_dict(), folder / 'optimizer.pt')
                write(checkpoint, dict(completed_steps=step + 1, training_file_sha256=digest(path),
                                       adapter_file=str(folder / 'adapter_model.safetensors'),
                                       optimizer_state=str(folder / 'optimizer.pt')))
            dist.barrier()
        del ids, labels, loss, weighted, row
    seen_all = [None] * world
    dist.all_gather_object(seen_all, seen)
    if checkpoint_step == 0:
        consumed = [name for lane in seen_all for name in lane]
        assert len(consumed) == len(set(consumed)) == 865 and set(consumed) == approved
    dist.barrier()
    if rank == 0:
        model.save_pretrained(target / 'adapter', safe_serialization=True)
        AutoTokenizer.from_pretrained(ROOT / 'models/gemma-4-12b-it', local_files_only=True).save_pretrained(target / 'adapter')
        result.update(status='complete', completed_steps=steps, completed_epochs=1,
                      compute_seconds=sum(timings), total_worker_seconds=time.monotonic() - start,
                      adapter=str(target / 'adapter'), source_papers_used_once=True)
        write(target / 'result.json', result)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
