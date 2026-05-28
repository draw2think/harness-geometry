"""
Environment variable loader for API keys.

Loads from .env file in project root.
"""

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai as _genai
    from openai import OpenAI as _OpenAI


def load_env_file():
    """Load environment variables from .env file."""
    # Find .env file (in project root)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        print(f"Warning: .env file not found at {env_file}")
        return

    # Load .env file
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key] = value


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a provider.

    Args:
        provider: "google", "openai", "anthropic", or "proxy"

    Returns:
        API key string or None
    """
    # Ensure .env is loaded
    load_env_file()

    key_map = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "proxy": "PROXY_API_KEY",
    }

    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    api_key = os.getenv(env_var)
    if not api_key:
        print(f"Warning: {env_var} not found in environment")

    return api_key


def make_genai_client(provider: str = "auto") -> "_genai.Client":
    """
    Create a google.genai Client for either the official Google API or a proxy.

    provider:
        "auto":   read USE_PROXY; use proxy if it is "1", otherwise use Google
        "google": call the official Google API with GOOGLE_API_KEY
        "proxy":  call a proxy with PROXY_API_KEY + PROXY_BASE_URL

    Switching:
        export USE_PROXY=1        # temporarily route through proxy
        export USE_PROXY=         # return to the official endpoint
    You can also set USE_PROXY=1 in .env.
    """
    load_env_file()
    from google import genai
    from google.genai.types import HttpOptions

    if provider == "auto":
        provider = "proxy" if os.getenv("USE_PROXY") == "1" else "google"

    if provider == "proxy":
        api_key = os.getenv("GOOGLE_PROXY_KEY") or os.getenv("PROXY_API_KEY", "")
        base_url = os.getenv("PROXY_BASE_URL", "")
        if not api_key or api_key.startswith("sk-<YourKey>"):
            raise ValueError("GOOGLE_PROXY_KEY or PROXY_API_KEY not set in .env")
        print(f"[client] Using proxy {base_url}")
        return genai.Client(
            api_key=api_key,
            http_options=HttpOptions(baseUrl=base_url),
        )
    else:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        print("[client] Using Google official API")
        return genai.Client(api_key=api_key)


def make_dashscope_client(api_type: str = "responses") -> "_OpenAI":
    """
    Create an OpenAI-compatible client for Alibaba Cloud DashScope (Singapore).

    api_type:
        "responses":  Responses API, recommended for text models and
                        previous_response_id multi-turn state. Use it for
                        qwen3-max, qwen3.5-plus/flash, qwen3.5-397b-a17b,
                        qwen3.5-122b-a10b, qwen3.5-27b, etc.
        "chat":       Chat Completions API for Qwen3-VL vision models such
                        as qwen3-vl-plus and qwen3-vl-flash.

    Model examples:
        qwen3-max            qwen3-max-2026-01-23
        qwen3.5-plus         qwen3.5-plus-2026-02-15
        qwen3.5-397b-a17b    qwen3.5-122b-a10b
        qwen3-vl-plus        qwen3-vl-flash       (requires api_type="chat")

    Enable thinking for Responses API:
        extra_body={"enable_thinking": True}
    Enable thinking for Chat Completions API:
        extra_body={"enable_thinking": True}
    Reasoning content fields:
        Responses API → output item type="reasoning"
        Chat API      → choice.message.reasoning_content
    """
    load_env_file()
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not set in .env")

    if api_type == "responses":
        base_url = "https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
        print(f"[client] DashScope Responses API (Singapore)")
    else:
        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        print(f"[client] DashScope Chat Completions API (Singapore, for VL)")

    return OpenAI(api_key=api_key, base_url=base_url)


def verify_all_keys() -> dict:
    """
    Verify all API keys are present.

    Returns:
        Dict with provider: bool mapping
    """
    load_env_file()

    return {
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY"))
    }
