"""OS-process wrapper used by the subprocess-based worker context."""
import os
import signal
import subprocess


def _get_default_signal() -> signal.Signals:
    return signal.SIGTERM


class SubprocessHandler:
    """Owns one worker ``Popen`` and its lifecycle."""

    def __init__(
        self,
        args: tuple | str,
        env: dict[str, str],
        stdout: str | None = None,
        stderr: str | None = None,
        local_rank_id: int = -1,
        numa_options=None,
    ) -> None:
        if stdout and stderr and os.path.realpath(os.path.normpath(stdout)) == os.path.realpath(
            os.path.normpath(stderr)
        ):
            raise ValueError(
                f"local_rank_id {local_rank_id}: stdout and stderr files must not be the same"
            )
        self.local_rank_id = local_rank_id
        self.stdout = stdout
        self.stderr = stderr
        self._numa_options = numa_options
        self._stdout_file = None
        self._stderr_file = None
        self.proc = self._popen(args, env)

    def _popen(self, args: tuple, env: dict[str, str]) -> subprocess.Popen:
        # Streams default to inheriting the agent's stdout/stderr; files are
        # used only when redirection was requested for this rank.
        env_vars = os.environ.copy()
        env_vars.update(env)
        self._stdout_file = open(self.stdout, "w", buffering=1) if self.stdout else None
        self._stderr_file = open(self.stderr, "w", buffering=1) if self.stderr else None
        return subprocess.Popen(
            args=args,
            env=env_vars,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
            start_new_session=(os.name != "nt"),
        )

    def close(self, death_sig: signal.Signals | None = None, timeout: int = 30) -> None:
        """Terminate the process, escalating to kill after a grace period."""
        if not death_sig:
            death_sig = _get_default_signal()
        try:
            if self.proc.poll() is None:
                try:
                    if os.name != "nt":
                        os.killpg(self.proc.pid, death_sig)
                    else:
                        self.proc.send_signal(death_sig)
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name != "nt":
                            os.killpg(self.proc.pid, signal.SIGKILL)
                        else:
                            self.proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            for stream in (self._stdout_file, self._stderr_file):
                if stream is not None:
                    stream.close()
            self._stdout_file = None
            self._stderr_file = None

    def poll(self) -> int | None:
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
