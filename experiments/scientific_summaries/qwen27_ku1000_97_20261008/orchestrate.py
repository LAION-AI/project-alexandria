"""One owned allocation: three Qwen27 extractor services and one fixed QA judge."""
import concurrent.futures
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from common import ROOT, digest, load, sources, write
from extract import run as extract_run
from evaluate import run as qa_run
from report import finalize

OWNED = []
OWNED_LOCK = threading.Lock()
ABORT = threading.Event()
START = time.monotonic()


def status(phase, **fields):
    write(ROOT / "outputs/status.json", dict(phase=phase, elapsed_seconds=time.monotonic()-START,
        updated_unix=time.time(), job_id=os.environ.get("SLURM_JOB_ID"), **fields))
    print("STATUS", phase, fields, flush=True)


def start_server(gpu, kind):
    protocol = load(ROOT / "inputs/protocol.json")
    model = protocol["extractor"]["local_path"] if kind == "extractor" else protocol["judge"]["local_path"]
    alias = "qwen27-ku" if kind == "extractor" else "qwen25"
    port = 20380 + gpu
    cache = ROOT / "cache" / f"gpu-{gpu}"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HOME=str(ROOT / "cache/hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8", PYTHONUNBUFFERED="1",
        XDG_CACHE_HOME=str(cache), TRITON_CACHE_DIR=str(cache / "triton"),
        VLLM_CACHE_ROOT=str(cache / "vllm"), TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"))
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
        "--served-model-name", alias, "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", "1", "--max-model-len", "32768",
        "--max-num-seqs", "16" if kind == "extractor" else "4",
        "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", ".90",
        "--generation-config", "vllm", "--enable-prefix-caching", "--enable-chunked-prefill"]
    if kind == "extractor":
        argv += ["--reasoning-parser", "qwen3", "--language-model-only"]
    log = (ROOT / "logs" / f"server-{kind}-{gpu}-{time.time_ns()}.log").open("w")
    start = time.monotonic()
    process = subprocess.Popen(argv, env=environment, stdout=log, stderr=log, start_new_session=True)
    item = dict(process=process, log=log, argv=argv, gpu=gpu, kind=kind,
                start=start, endpoint=f"http://127.0.0.1:{port}")
    with OWNED_LOCK:
        OWNED.append(item)
    write(ROOT / "outputs/server_commands" / f"{kind}-{gpu}.json", dict(command=argv,
        gpu=gpu, kind=kind, pid=process.pid, job_id=os.environ.get("SLURM_JOB_ID")))
    for _ in range(360):
        if process.poll() is not None:
            raise RuntimeError(f"Owned {kind} server on GPU {gpu} exited; see {log.name}")
        try:
            with urllib.request.urlopen(item["endpoint"] + "/health", timeout=5) as response:
                if response.status == 200:
                    item["load_seconds"] = time.monotonic()-start
                    return item
        except (OSError, ValueError):
            pass
        time.sleep(5)
    raise RuntimeError(f"Owned {kind} server health timeout")


def stop(item):
    process = item["process"]
    if process.poll() is None:
        if os.getpgid(process.pid) != process.pid:
            raise RuntimeError("Owned server process group changed")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    item["log"].close()
    write(ROOT / "outputs/server_accounting" / f"{item['kind']}-{item['gpu']}.json",
        dict(kind=item["kind"], gpu=item["gpu"], seconds=time.monotonic()-item["start"],
             load_seconds=item.get("load_seconds"), exit_code=process.returncode,
             command=item["argv"], log_file=item["log"].name))
    with OWNED_LOCK:
        OWNED.remove(item)


def signals(signum, frame):
    ABORT.set()
    raise SystemExit(f"Signal {signum}; atomic experiment checkpoints retained")


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run GPU/large-audit work in the owned compute allocation")
    protocol = load(ROOT / "inputs/protocol.json")
    for relative, expected in protocol["input_sha256"].items():
        if digest((ROOT / relative).read_text()) != expected:
            raise ValueError("Pinned input changed: " + relative)
    for relative, expected in protocol["vendor_sha256"].items():
        if digest((ROOT / relative).read_text()) != expected:
            raise ValueError("Pinned repository code changed: " + relative)
    sources()
    write(ROOT / "outputs/code_snapshot.json", {p.name: digest(p.read_text())
        for p in (ROOT / "code").glob("*.py")})
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, signals)
    status("loading_servers", extractor_replicas=3, judge_replicas=1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(start_server, gpu, "extractor" if gpu < 3 else "judge") for gpu in range(4)]
        servers = [f.result() for f in pending]
    status("extracting_and_evaluating", papers=97, new_source_words_per_call=1000)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        extraction = [pool.submit(extract_run, s["endpoint"], s["gpu"]) for s in servers[:3]]
        qa = pool.submit(qa_run, servers[3]["endpoint"], ABORT)
        try:
            for future in concurrent.futures.as_completed(extraction):
                report = future.result()
                stop(servers[report["replica"]])
            qa.result()
            stop(servers[3])
        except BaseException:
            ABORT.set()
            raise
    status("source_copy_audit_and_report")
    report = finalize()
    elapsed = time.monotonic()-START
    allocation = dict(job_id=os.environ["SLURM_JOB_ID"], nodes=1, allocated_gpus=4,
        orchestrator_elapsed_seconds=elapsed, orchestrator_allocated_gpu_hours=elapsed*4/3600,
        slurm_final_elapsed_pending=True,
        note="Actual sacct allocation includes module/srun setup and teardown; reconcile after job completion.")
    write(ROOT / "outputs/allocation_accounting.json", allocation)
    status("complete", scores=report["scores"])
    write(ROOT / "outputs/complete.json", dict(complete=True, papers=97, questions_per_condition=970,
        conditions=4, job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=elapsed))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        ABORT.set()
        status("failed", error=repr(error))
        raise
    finally:
        for item in list(reversed(OWNED)):
            stop(item)
