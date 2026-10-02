"""Local queue locking and identity-checked adoption of an orphaned model server."""
import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def exclusive_lock(path):
    # Keep the inode: unlinking a lock permits another process to lock a new inode.
    with Path(path).open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AlreadyRunning('An existing process owns this queue lock') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def process_identity(pid):
    # comm may contain spaces or parentheses, so split after its final closing paren.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return fields[0], fields[19]  # state, Linux starttime (field 22)


class AdoptedServer:
    """Popen-compatible lifecycle for one explicitly requested, validated PID.

    No arbitrary GPU process is adopted or stopped. Validate the Unix owner,
    session/group identity, exact serving module, model, revision, alias and port.
    Preserve starttime to reject PID reuse before signalling or waiting.
    """

    def __init__(self, pid, model):
        self.pid = pid
        directory = Path(f'/proc/{pid}')
        if directory.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
            raise ValueError('Requested server is not a same-user isolated process group')
        args = directory.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        expected = {'--model': model['model'], '--revision': model['revision'],
                    '--served-model-name': model['alias'], '--port': '8010', '--host': '127.0.0.1'}
        if 'vllm.entrypoints.openai.api_server' not in args:
            raise ValueError('Requested PID is not the expected vLLM serving module')
        for flag, value in expected.items():
            if args.count(flag) != 1 or args.index(flag) + 1 >= len(args) or args[args.index(flag) + 1] != value:
                raise ValueError('Requested server settings do not match the pinned extractor')
        state, self.starttime = process_identity(pid)
        if state == 'Z':
            raise ValueError('Requested server has already exited')

    def poll(self):
        try:
            state, starttime = process_identity(self.pid)
        except FileNotFoundError:
            return 0
        return 0 if state == 'Z' or starttime != self.starttime else None

    def assert_identity(self):
        if self.poll() is not None or os.getpgid(self.pid) != self.pid:
            raise RuntimeError('Adopted server identity changed; refusing to signal it')

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('adopted model server', timeout)
            time.sleep(0.2)
        return 0
