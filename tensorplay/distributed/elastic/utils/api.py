"""Process launch helpers shared by the elastic agent and its tooling."""
import os
import socket


def get_env_variable_or_raise(env_name: str) -> str:
    """Return the value of ``env_name`` or raise if it is unset/empty."""
    value = os.environ.get(env_name)
    if value is None:
        raise ValueError(f"Environment variable '{env_name}' is required but has no value")
    return value


def get_socket_with_port() -> socket.socket:
    """Create a socket bound to an ephemeral local port and return it.

    The caller owns the socket; close it once the port has been consumed.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0))
    sock.listen(1)
    return sock


class macros:
    """Substitution variables usable in worker argument templates.

    Any ``${field}`` token inside the worker argument list is replaced with
    the concrete value for the worker being launched. Supported fields are
    the entries of :attr:`fields`.
    """

    fields = [
        "record_id",
        "role_name",
        "role_rank",
        "role_world_size",
        "local_rank",
        "local_world_size",
    ]

    @classmethod
    def substitute(cls, args: list[str], local_rank: str | None = None) -> list[str]:
        """Replace ``${field}`` tokens in ``args``; unknown tokens are kept."""
        if local_rank is not None:
            from string import Template

            return [
                Template(value).safe_substitute(local_rank=local_rank)
                if isinstance(value, str)
                else value
                for value in args
            ]
        if not any("${" in str(arg) for arg in args):
            return args
        import re

        pattern = re.compile(r"\$\{(" + "|".join(cls.fields) + r")\}")

        def _replace(match: "re.Match[str]") -> str:
            field = match.group(1)
            return str(macros.to_map().get(field, match.group(0)))

        return [pattern.sub(_replace, str(arg)) for arg in args]

    @classmethod
    def to_map(cls) -> dict[str, str]:
        """Current macro values from the running process' environment."""
        env = os.environ
        return {
            "record_id": env.get("TORCHELASTIC_RUN_ID", ""),
            "role_name": env.get("ROLE_NAME", ""),
            "role_rank": env.get("ROLE_RANK", ""),
            "role_world_size": env.get("ROLE_WORLD_SIZE", ""),
            "local_rank": env.get("LOCAL_RANK", ""),
            "local_world_size": env.get("LOCAL_WORLD_SIZE", ""),
        }
