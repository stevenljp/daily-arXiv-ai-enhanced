"""Runtime configuration shared by the paper-enhancement job."""

import json
import os
from urllib.parse import urlparse


def build_chat_openai_kwargs(model_name: str, base_url: str, api_key: str) -> dict:
    """Build provider-safe ChatOpenAI settings from workflow configuration."""
    model_name = model_name.strip()
    base_url = base_url.strip().rstrip("/")
    api_key = api_key.strip()

    missing = [
        name
        for name, value in (
            ("MODEL_NAME", model_name),
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")

    kwargs = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
    }
    hostname = (urlparse(base_url).hostname or "").lower()

    # 这些服务商默认开启 thinking 模式,而 thinking 模式不支持 tool_choice。
    # with_structured_output(method="function_calling") 会带上 tool_choice,
    # 于是每篇论文都 400:
    #   Thinking mode does not support this tool_choice
    # 结果是整批 enhance 挂掉 -> exit 1 -> 当天数据一条都提交不了。
    # 所以对已知厂商默认关掉 thinking。
    thinking_off_hosts = ("volces.com", "minimax", "minimaxi", "moonshot", "dashscope")
    if any(h in hostname for h in thinking_off_hosts):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # 也允许用 LLM_EXTRA_BODY 显式传(优先级最高),用于其他厂商
    raw_extra = (os.environ.get("LLM_EXTRA_BODY") or "").strip()
    if raw_extra:
        try:
            extra = json.loads(raw_extra)
        except json.JSONDecodeError as exc:
            print(f"LLM_EXTRA_BODY 不是合法 JSON,已忽略: {exc}", file=sys.stderr)
        else:
            if isinstance(extra, dict):
                merged = dict(kwargs.get("extra_body") or {})
                merged.update(extra)
                kwargs["extra_body"] = merged
    return kwargs


def raise_if_processing_failed(errors: list[str]) -> None:
    """Make a failed AI batch fail the workflow instead of publishing placeholders."""
    if errors:
        raise RuntimeError(f"{len(errors)} paper(s) failed AI enhancement: {errors[0]}")
