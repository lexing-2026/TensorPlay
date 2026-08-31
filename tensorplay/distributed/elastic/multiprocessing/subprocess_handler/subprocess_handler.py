"""OS-process wrapper used by the subprocess-based worker context."""
import os
import signal
import subprocess
from types import FrameType


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
        self.proc = self._popen(args, env)

    def _popen(self, args: tuple, env: dict[str, str]) -> subprocess.Popen:
        # Streams default to inheriting the agent's stdout/stderr; files are
        # used only when redirection was requested for this rank.
        return subprocess.Popen(
            args=args,
            env=env,
            stdout=open(self.stdout, "w") if self.stdout else None,
            stderr=open(self.stderr, "w") if self.stderr else None,
        )

    def close(self, death_sig: signal.Signals | None = None) -> None:
        """Terminate the process, escalating to kill after a grace period."""
        if not death_sig:
            death_sig = _get_default_signal()
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(death_sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=5)

    def poll(self) -> int | None:
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
