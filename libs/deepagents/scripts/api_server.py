"""Run the Deep Agents FastAPI server."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000


def main() -> None:
    """Start the API server with `uvicorn`."""
    _load_runtime_env()
    try:
        import uvicorn  # noqa: PLC0415

        from deepagents.server.app import create_app  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `deepagents[api]` to run the API server."
        raise ImportError(msg) from exc

    host = os.environ.get("DEEPAGENTS_API_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("DEEPAGENTS_API_PORT", str(_DEFAULT_PORT)))
    uvicorn.run(create_app(), host=host, port=port)


def _load_runtime_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / "deepagents" / "runtime" / "runtime.env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


if __name__ == "__main__":
    main()
