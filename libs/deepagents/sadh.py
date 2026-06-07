from __future__ import annotations

import os
from pathlib import Path

from deepagents.runtime import AgentRuntimeConfig
from deepagents.websearch import create_web_agent_tools


def load_runtime_env() -> None:
    env_path = Path("deepagents/runtime/runtime.env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    load_runtime_env()

    config = AgentRuntimeConfig.from_env()
    tools = {tool.name: tool for tool in create_web_agent_tools(config)}

    url = "https://www.bilibili.com/"

    print("\n========== STATIC FETCH ==========\n")
    static_result = tools["web_fetch_static"].invoke({"url": url})
    print(static_result[:5000])

    print("\n========== RENDER FETCH ==========\n")
    render_result = tools["web_render_page"].invoke({"url": url})
    print(render_result[:5000])


if __name__ == "__main__":
    main()