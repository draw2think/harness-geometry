"""
Model registry for geometry eval model selection and client routing.

Usage:
    from symbolic.utils.model_registry import get_model, make_client, list_models

    # List models with vision, tool calling, and reasoning support.
    for m in list_models(vision=True, tool_calling=True, thinking=True):
        print(m.id, m.provider, m.price)

    # Build a client from a registry id.
    client, card = make_client("kimi-k2.5")
    client, card = make_client("gemini-3.1-pro-preview")
    client, card = make_client("claude-sonnet-4-6")

SDK routing:
    google-genai  →  google.genai.Client        (Gemini family)
    openai        →  openai.OpenAI              (Kimi/Step/GLM/DeepSeek/MiniMax/Qwen/OpenAI)
    anthropic     →  anthropic.Anthropic         (Claude family)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple


@dataclass
class ModelCard:
    """Complete metadata for one model."""

    # ── Identity ──
    id: str                     # Unique registry key, e.g. "kimi-k2.5"
    provider: str               # Provider: google, openai, anthropic, moonshot, zhipu, stepfun, ...
    model_name: str             # API model parameter, e.g. "kimi-k2.5", "gemini-3.1-pro-preview"

    # ── SDK routing ──
    sdk: str                    # "google-genai" | "openai" | "anthropic"
    base_url: Optional[str]     # OpenAI-compatible endpoint; None = provider default
    api_key_env: str            # Environment variable name, e.g. "MOONSHOT_API_KEY"

    # ── Capabilities ──
    vision: bool = False        # Image input
    thinking: bool = False      # Deep reasoning / thinking mode
    tool_calling: bool = False  # function calling
    context_k: int = 128        # Context window (K tokens)
    thinking_level: str = ""    # Gemini: minimal/low/medium/high; OpenAI: low/medium/high; "" = unset
    use_responses_api: bool = False  # OpenAI Responses API (v1/responses) for reasoning+tools
    fixed_temperature: Optional[float] = None  # Model-enforced temperature, e.g. Kimi K2.5 = 1.0

    # ── Metadata ──
    price: str = ""             # Short pricing string, e.g. "$0.5/M"
    open_source: bool = False
    release: str = ""           # Release date, e.g. "2026-02"
    notes: str = ""             # Notes


# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry: 2026-03 SOTA
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRY: List[ModelCard] = [

    # ── Google Gemini ────────────────────────────────────────────────────────
    # ID format: model@thinking_level. Different thinking levels are separate cards.
    ModelCard(
        id="gemini-3.1-pro-preview@high",
        provider="google", model_name="gemini-3.1-pro-preview",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="high",
        price="$2/$12 M", release="2026-02",
        notes="AA Index #1; thought parts must be filtered from history",
    ),
    ModelCard(
        id="gemini-3.1-pro-preview@medium",
        provider="google", model_name="gemini-3.1-pro-preview",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="medium",
        price="$2/$12 M", release="2026-02",
        notes="medium thinking lowers token usage",
    ),
    ModelCard(
        id="gemini-3.1-pro-preview-customtools@high",
        provider="google", model_name="gemini-3.1-pro-preview-customtools",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="high",
        price="$2/$12 M", release="2026-02",
        notes="customtools variant; prefers registered tools over bash; preferred for construct mode",
    ),
    # Proxy routing is selected by USE_PROXY=1, so a separate card is usually unnecessary.
    ModelCard(
        id="gemini-3.1-pro-preview-high",
        provider="google-proxy", model_name="gemini-3-pro-preview-thinking-high",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_PROXY_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="",
        price="$1.86/M", release="2026-03",
        notes="proxy endpoint includes thinking=high; model name has -thinking-high suffix",
    ),
    ModelCard(
        id="gemini-3-flash-preview@high",
        provider="google", model_name="gemini-3-flash-preview",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="high",
        price="$0.5/$3 M", release="2025-12",
        notes="Flash flagship; 1M context; near-Pro performance at one-quarter cost",
    ),
    ModelCard(
        id="gemini-3-flash-preview@medium",
        provider="google", model_name="gemini-3-flash-preview",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="medium",
        price="$0.5/$3 M", release="2025-12",
        notes="Flash flagship; medium thinking reduces circle-problem timeouts",
    ),
    ModelCard(
        id="gemini-3-flash-preview@low",
        provider="google", model_name="gemini-3-flash-preview",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="low",
        price="$0.5/$3 M", release="2025-12",
        notes="Flash flagship; low thinking gives the fastest response",
    ),
    ModelCard(
        id="gemini-3.5-flash@high",
        provider="google", model_name="gemini-3.5-flash",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="high",
        price="$0.5/$3 M", release="2026-05",
        notes="Gemini 3.5 Flash; 1M ctx; 64K output; thinking level=high",
    ),
    ModelCard(
        id="gemini-3.5-flash@medium",
        provider="google", model_name="gemini-3.5-flash",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="medium",
        price="$0.5/$3 M", release="2026-05",
        notes="Gemini 3.5 Flash; balanced quality/cost/latency",
    ),
    ModelCard(
        id="gemini-3.5-flash@low",
        provider="google", model_name="gemini-3.5-flash",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="low",
        price="$0.5/$3 M", release="2026-05",
        notes="Gemini 3.5 Flash; low-latency reasoning setting",
    ),
    ModelCard(
        id="gemini-2.5-flash@high",
        provider="google", model_name="gemini-2.5-flash",
        sdk="google-genai", base_url=None, api_key_env="GOOGLE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="high",
        price="very low", release="2025",
        notes="bulk low-cost default; controllable thinking budget",
    ),

    # ── Anthropic Claude ─────────────────────────────────────────────────────
    # Thinking blocks must be retained in history, unlike Gemini.
    # thinking + tool_choice supports only "auto", not "any" or "tool".
    # Claude @adaptive: thinking={"type":"adaptive"} + output_config={"effort": thinking_level}
    # Effort levels: low/medium/high (all models), max (Opus 4.6 only).
    # No @suffix means ordinary fast mode without thinking; @off is not registered separately.
    ModelCard(
        id="claude-opus-4-7@high",
        provider="anthropic", model_name="claude-opus-4-7",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="high",
        price="$5/$25 M", release="2026-05",
        notes="Latest Opus; adaptive thinking; 1M ctx; high effort",
    ),
    ModelCard(
        id="claude-opus-4-7@medium",
        provider="anthropic", model_name="claude-opus-4-7",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1000,
        thinking_level="medium",
        price="$5/$25 M", release="2026-05",
        notes="Latest Opus; adaptive thinking; 1M ctx; medium effort",
    ),
    ModelCard(
        id="claude-opus-4-6@high",
        provider="anthropic", model_name="claude-opus-4-6",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="high",
        price="$5/$25 M", release="2026-02",
        notes="latest flagship; adaptive thinking; effort controls depth; Batch $2.5/$12.5",
    ),
    ModelCard(
        id="claude-opus-4-6@medium",
        provider="anthropic", model_name="claude-opus-4-6",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="medium",
        price="$5/$25 M", release="2026-02",
        notes="flagship medium effort; baseline comparison setting; lower thinking cost",
    ),
    ModelCard(
        id="claude-opus-4-5@high",
        provider="anthropic", model_name="claude-opus-4-5",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="high",
        price="$5/$25 M", release="2025",
        notes="previous flagship; adaptive thinking; compare against 4.6 upgrade",
    ),
    ModelCard(
        id="claude-opus-4-5@medium",
        provider="anthropic", model_name="claude-opus-4-5",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="medium",
        price="$5/$25 M", release="2025",
        notes="previous flagship; medium effort; baseline comparison setting",
    ),
    ModelCard(
        id="claude-sonnet-4-6@high",
        provider="anthropic", model_name="claude-sonnet-4-6",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="high",
        price="$3/$15 M", release="2026-02",
        notes="latest value model; adaptive thinking; Batch $1.5/$7.5",
    ),
    ModelCard(
        id="claude-sonnet-4-6@medium",
        provider="anthropic", model_name="claude-sonnet-4-6",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="medium",
        price="$3/$15 M", release="2026-02",
        notes="value model with medium effort; baseline comparison setting; lower thinking cost",
    ),
    ModelCard(
        id="claude-sonnet-4-5@high",
        provider="anthropic", model_name="claude-sonnet-4-5",
        sdk="anthropic", base_url=None, api_key_env="ANTHROPIC_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="high",
        price="$3/$15 M", release="2025",
        notes="previous value model; adaptive thinking; compare upgrade effect",
    ),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    # reasoning_effort: none/low/medium/high/xhigh (latest GPT-5 family)
    # gpt-5.2-pro supports only Responses API; Chat Completions is unsupported.
    ModelCard(
        id="gpt-5.2",
        provider="openai", model_name="gpt-5.2",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        thinking_level="high",
        price="$1.75/$14 M", release="2025-12",
        notes="reasoning flagship; 400K ctx; 128K output; τ²-Bench 98.7%; GPQA 92.4%",
    ),
    ModelCard(
        id="gpt-5.2-2025-12-11",
        provider="openai", model_name="gpt-5.2-2025-12-11",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        thinking_level="high",
        price="$1.75/$14 M", release="2025-12-11",
        notes="gpt-5.2 pinned snapshot; reproducible eval target",
    ),
    ModelCard(
        id="gpt-5.3-chat-latest",
        provider="openai", model_name="gpt-5.3-chat-latest",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=False, tool_calling=True, context_k=128,
        thinking_level="",
        price="$1/$4 M", release="2026-01",
        notes="instant snapshot; without reasoning; 128K ctx / 16K output; baseline only",
    ),
    ModelCard(
        id="o4-mini",
        provider="openai", model_name="o4-mini",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=128,
        price="medium", release="2025",
        notes="GeoEval PCS=78.5 / ACS=92.8; strongest geometry score",
    ),

    # ── GPT-5.1 (proxy) ─────────────────────────────────────────────────────
    ModelCard(
        id="gpt-5.1",
        provider="openai-proxy", model_name="gpt-5.1",
        sdk="openai", base_url=None, api_key_env="OPENAI_PROXY_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="",
        price="proxy", release="2026-03",
        notes="GPT-5.1 via proxy; without reasoning",
    ),
    ModelCard(
        id="gpt-5.1@low",
        provider="openai-proxy", model_name="gpt-5.1",
        sdk="openai", base_url=None, api_key_env="OPENAI_PROXY_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="low",
        price="proxy", release="2026-03",
        notes="GPT-5.1 via proxy; reasoning_effort=low",
    ),
    ModelCard(
        id="gpt-5.1@medium",
        provider="openai-proxy", model_name="gpt-5.1",
        sdk="openai", base_url=None, api_key_env="OPENAI_PROXY_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="medium",
        price="proxy", release="2026-03",
        notes="GPT-5.1 via proxy; reasoning_effort=medium",
    ),
    ModelCard(
        id="gpt-5.1@high",
        provider="openai-proxy", model_name="gpt-5.1",
        sdk="openai", base_url=None, api_key_env="OPENAI_PROXY_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        thinking_level="high",
        price="proxy", release="2026-03",
        notes="GPT-5.1 via proxy; reasoning_effort=high",
    ),

    # ── GPT-5.5 ──────────────────────────────────────────────────────────────
    ModelCard(
        id="gpt-5.5",
        provider="openai", model_name="gpt-5.5",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1050,
        thinking_level="",
        price="$5/$30 M", release="2026-04-23",
        notes="GPT-5.5; 1.05M ctx; 128K output; default reasoning effort",
    ),
    ModelCard(
        id="gpt-5.5@medium",
        provider="openai", model_name="gpt-5.5",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1050,
        thinking_level="medium", use_responses_api=True,
        price="$5/$30 M", release="2026-04-23",
        notes="GPT-5.5; reasoning_effort=medium; via Responses API",
    ),
    ModelCard(
        id="gpt-5.5@high",
        provider="openai", model_name="gpt-5.5",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1050,
        thinking_level="high", use_responses_api=True,
        price="$5/$30 M", release="2026-04-23",
        notes="GPT-5.5; reasoning_effort=high; via Responses API",
    ),
    ModelCard(
        id="gpt-5.5@xhigh",
        provider="openai", model_name="gpt-5.5",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=1050,
        thinking_level="xhigh", use_responses_api=True,
        price="$5/$30 M", release="2026-04-23",
        notes="GPT-5.5; reasoning_effort=xhigh; via Responses API",
    ),

    # ── GPT-5.4 ──────────────────────────────────────────────────────────────
    ModelCard(
        id="gpt-5.4",
        provider="openai", model_name="gpt-5.4",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        thinking_level="",
        price="$2.50/$15.00 per M", release="2026-03",
        notes="GPT-5.4; 400K ctx; official endpoint; default reasoning_effort from config",
    ),
    ModelCard(
        id="gpt-5.4@medium",
        provider="openai", model_name="gpt-5.4",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        thinking_level="medium", use_responses_api=True,
        price="$5.00/$30.00 per M", release="2026-03",
        notes="GPT-5.4; reasoning_effort=medium",
    ),
    ModelCard(
        id="gpt-5.4@high",
        provider="openai", model_name="gpt-5.4",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        thinking_level="high", use_responses_api=True,
        price="$5.00/$30.00 per M", release="2026-03",
        notes="GPT-5.4; reasoning_effort=high; tools+reasoning requires Responses API",
    ),
    ModelCard(
        id="gpt-5.4-mini",
        provider="openai", model_name="gpt-5.4-mini",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        price="$0.75/$4.50 per M", release="2026-03",
        notes="GPT-5.4 mini; 400K ctx; 128K max output; official endpoint; default reasoning_effort from config",
    ),
    # gpt-5.4-mini: reasoning_effort + tools requires Responses API (v1/responses),
    # not supported in Chat Completions (v1/chat/completions).
    # @medium/@high variants need Responses API adapter to actually use reasoning.
    # Without it, reasoning_effort is auto-skipped and logged as think=0.
    ModelCard(
        id="gpt-5.4-mini@medium",
        provider="openai", model_name="gpt-5.4-mini",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        thinking_level="medium", use_responses_api=True,
        price="$0.75/$4.50 per M", release="2026-03",
        notes="GPT-5.4 mini; reasoning_effort=medium; via Responses API (v1/responses)",
    ),
    ModelCard(
        id="gpt-5.4-mini@high",
        provider="openai", model_name="gpt-5.4-mini",
        sdk="openai", base_url=None, api_key_env="OPENAI_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=400,
        thinking_level="high", use_responses_api=True,
        price="$0.75/$4.50 per M", release="2026-03",
        notes="GPT-5.4 mini; reasoning_effort=high; via Responses API (v1/responses)",
    ),

    # ── Moonshot (Kimi) ──────────────────────────────────────────────────────
    ModelCard(
        id="kimi-k2.5",
        provider="moonshot", model_name="kimi-k2.5",
        sdk="openai", base_url="https://api.moonshot.ai/v1",
        api_key_env="MOONSHOT_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=256,
        fixed_temperature=1.0,
        price="$0.5/M", open_source=True, release="2026-01",
        notes="native multimodal; default thinking; base64 images; retain reasoning_content in history",
    ),

    # ── Zhipu (GLM) ─────────────────────────────────────────────────────────
    ModelCard(
        id="glm-4.6v",
        provider="zhipu", model_name="glm-4.6v",
        sdk="openai", base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=128,
        price="$0.3/$0.9 M", open_source=True, release="2026-01",
        notes="native multimodal FC; 128K context; arbitrary-resolution images; OpenRouter $0.3/$0.9",
    ),
    ModelCard(
        id="glm-5",
        provider="zhipu", model_name="glm-5",
        sdk="openai", base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        vision=False, thinking=True, tool_calling=True, context_k=200,
        price="$1/$3.2 M", release="2026-02",
        notes="flagship text model; coding approaches Claude Opus; three thinking modes",
    ),
    ModelCard(
        id="glm-5v-turbo",            # CN endpoint (default)
        provider="zhipu", model_name="glm-5v-turbo",
        sdk="openai", base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        release="2026-04",
        notes="multimodal coding base; 200K ctx / 128K out; vision understanding and agent adaptation",
    ),
    ModelCard(
        id="glm-5v-turbo-intl",       # INTL endpoint (api.z.ai)
        provider="zhipu", model_name="glm-5v-turbo",
        sdk="openai", base_url="https://api.z.ai/api/paas/v4",
        api_key_env="ZHIPU_API_KEY_INTL",
        vision=True, thinking=True, tool_calling=True, context_k=200,
        release="2026-04",
        notes="Same as glm-5v-turbo but via intl endpoint api.z.ai; needs separate ZHIPU_API_KEY_INTL",
    ),

    # ── StepFun (Step) ──────────────────────────────────────────────────────
    ModelCard(
        id="step-3",
        provider="stepfun", model_name="step-3",
        sdk="openai", base_url="https://api.stepfun.com/v1",
        api_key_env="STEPFUN_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=128,
        price="¥1.5/¥4 M", open_source=True, release="2025-07",
        notes="321B MoE/38B active; native multimodal reasoning; 300% of R1 reasoning efficiency",
    ),
    ModelCard(
        id="step-3.5-flash",
        provider="stepfun", model_name="step-3.5-flash",
        sdk="openai", base_url="https://api.stepfun.com/v1",
        api_key_env="STEPFUN_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=128,
        price="¥1.5/¥4 M", open_source=True, release="2026-02",
        notes="1960B/110B active; 350 tok/s; agent workflow scenarios",
    ),

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    ModelCard(
        id="deepseek-v3.2",
        provider="deepseek", model_name="deepseek-chat",
        sdk="openai", base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        vision=False, thinking=True, tool_calling=True, context_k=128,
        price="$0.27/$1.1 M", open_source=True, release="2025-12",
        notes="Thinking-in-Tool-Use; text-only; AIME 96%",
    ),
    ModelCard(
        id="deepseek-reasoner",
        provider="deepseek", model_name="deepseek-reasoner",
        sdk="openai", base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        vision=False, thinking=True, tool_calling=True, context_k=128,
        price="$0.55/$2.19 M", open_source=True, release="2025-12",
        notes="reasoning_content field; exclude from history",
    ),
    # DeepSeek V4: pending release (2026-03)
    # ModelCard(id="deepseek-v4", ..., vision=True, ...)

    # ── MiniMax ──────────────────────────────────────────────────────────────
    ModelCard(
        id="minimax-m2.5",
        provider="minimax", model_name="MiniMax-M2.5",
        sdk="openai", base_url="https://api.minimax.chat/v1",
        api_key_env="MINIMAX_API_KEY",
        vision=False, thinking=True, tool_calling=True, context_k=1000,
        price="$0.3/$1.2 M", open_source=True, release="2026-02",
        notes="SWE-Bench 80.2%; text-only; among the cheapest options",
    ),

    # ── Qwen (DashScope) ────────────────────────────────────────────────────
    ModelCard(
        id="qwen3-vl-plus",
        provider="dashscope", model_name="qwen3-vl-plus",
        sdk="openai",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=128,
        price="medium", open_source=True, release="2026",
        notes="<think> tags; Chat Completions API; among the strongest open-source vision geometry models",
    ),
    ModelCard(
        id="qwen3.5-plus",
        provider="dashscope", model_name="qwen3.5-plus",
        sdk="openai",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=262,
        price="$0.4/$1.2 M", open_source=True, release="2026-02",
        notes="397B MoE/17B active; native multimodal early fusion; AA Index 45; scalable to 1M context",
    ),
    ModelCard(
        id="qwen3.5-plus-2026-02-15",
        provider="dashscope", model_name="qwen3.5-plus-2026-02-15",
        sdk="openai",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=262,
        price="$0.4/$1.2 M", open_source=True, release="2026-02-15",
        notes="qwen3.5-plus pinned version snapshot (2026-02-15)",
    ),

    # ── InternVL (third-party hosted) ───────────────────────────────────────
    ModelCard(
        id="internvl3-latest",
        provider="openrouter", model_name="OpenGVLab/InternVL3-latest",
        sdk="openai", base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=128,
        price="medium", open_source=True, release="2026",
        notes="NeSyGeo-Test 68.7%; MMMU 72.2; strongest open-source vision model",
    ),

    # ── xAI Grok (OpenAI-compatible proxy) ─────────────────────────────────
    ModelCard(
        id="grok-4.20-reasoning",
        provider="xai-proxy", model_name="grok-4.20-beta-0309-reasoning",
        sdk="openai", base_url=None,
        api_key_env="PROXY_API_KEY",
        vision=True, thinking=True, tool_calling=True, context_k=2000,
        price="$1.23/$3.70 M (proxy)", release="2026-03",
        notes="Grok 4.20 flagship reasoning; 2M context; vision+FC+reasoning; set PROXY_BASE_URL",
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Lookup & Filtering
# ═══════════════════════════════════════════════════════════════════════════════

_INDEX = {m.id: m for m in REGISTRY}


def get_model(model_id: str) -> ModelCard:
    """Look up a model by registry id."""
    if model_id not in _INDEX:
        available = ", ".join(sorted(_INDEX.keys()))
        raise KeyError(f"Unknown model '{model_id}'. Available: {available}")
    return _INDEX[model_id]


def list_models(
    *,
    vision: Optional[bool] = None,
    thinking: Optional[bool] = None,
    tool_calling: Optional[bool] = None,
    open_source: Optional[bool] = None,
    sdk: Optional[str] = None,
    provider: Optional[str] = None,
) -> List[ModelCard]:
    """Filter models by capabilities. None leaves a field unfiltered."""
    result = []
    for m in REGISTRY:
        if vision is not None and m.vision != vision:
            continue
        if thinking is not None and m.thinking != thinking:
            continue
        if tool_calling is not None and m.tool_calling != tool_calling:
            continue
        if open_source is not None and m.open_source != open_source:
            continue
        if sdk is not None and m.sdk != sdk:
            continue
        if provider is not None and m.provider != provider:
            continue
        result.append(m)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Client Construction
# ═══════════════════════════════════════════════════════════════════════════════

def _load_env():
    """Ensure the .env file has been loaded."""
    from symbolic.utils.env_loader import load_env_file
    load_env_file()


def make_client(model_id: str) -> Tuple[Any, ModelCard]:
    """
    Build an API client from a registry id.

    Returns:
        (client, model_card)

    Usage:
        client, card = make_client("kimi-k2.5")
        # Client type depends on card.sdk:
        #   "google-genai" → google.genai.Client
        #   "openai"       → openai.OpenAI
        #   "anthropic"    → anthropic.Anthropic
    """
    _load_env()
    card = get_model(model_id)
    api_key = os.getenv(card.api_key_env, "")

    # USE_PROXY=1 allows the official key to be empty because proxy keys are used later.
    if not api_key and os.getenv("USE_PROXY") != "1":
        raise ValueError(
            f"API key not set: export {card.api_key_env}=<your-key> (or add to .env)"
        )

    # ── google-genai SDK ──
    if card.sdk == "google-genai":
        from google import genai
        from google.genai.types import HttpOptions

        # Proxy routing: card.provider == "google-proxy" or USE_PROXY=1.
        use_proxy = (card.provider == "google-proxy"
                     or os.getenv("USE_PROXY") == "1")
        if use_proxy:
            proxy_key = (os.getenv("GOOGLE_PROXY_KEY")
                         or os.getenv("PROXY_API_KEY", ""))
            base_url = os.getenv("PROXY_BASE_URL", "")
            if not proxy_key:
                raise ValueError(
                    "Proxy enabled but neither GOOGLE_PROXY_KEY nor "
                    "PROXY_API_KEY is set")
            client = genai.Client(
                api_key=proxy_key,
                http_options=HttpOptions(baseUrl=base_url),
            )
        else:
            client = genai.Client(api_key=api_key)
        return client, card

    # ── OpenAI-compatible SDK ──
    if card.sdk == "openai":
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if card.base_url:
            kwargs["base_url"] = card.base_url
        elif card.provider.endswith("-proxy") or (
                not card.base_url and os.getenv("USE_PROXY") == "1"
                and card.provider.startswith("openai")):
            # proxy route: PROXY_BASE_URL + /v1
            proxy_base = os.getenv("PROXY_BASE_URL", "")
            if proxy_base:
                kwargs["base_url"] = proxy_base.rstrip("/") + "/v1"
            # When routing through proxy, swap to proxy key regardless of
            # what the card's api_key_env says; the proxy does not accept
            # the official OpenAI key.
            proxy_key = (os.getenv("OPENAI_PROXY_KEY")
                         or os.getenv("PROXY_API_KEY", ""))
            if proxy_key:
                kwargs["api_key"] = proxy_key
        client = OpenAI(**kwargs)
        return client, card

    # ── anthropic SDK ──
    if card.sdk == "anthropic":
        from anthropic import Anthropic

        if os.getenv("USE_PROXY") == "1":
            proxy_key = os.getenv("PROXY_API_KEY", "")
            base_url = os.getenv("PROXY_BASE_URL", "")
            if not proxy_key:
                raise ValueError("USE_PROXY=1 but PROXY_API_KEY not set")
            client = Anthropic(api_key=proxy_key, base_url=base_url)
        else:
            client = Anthropic(api_key=api_key)
        return client, card

    raise ValueError(f"Unknown SDK type: {card.sdk}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI: python -m symbolic.utils.model_registry
# ═══════════════════════════════════════════════════════════════════════════════

def print_registry(models: Optional[List[ModelCard]] = None):
    """Print the model registry."""
    models = models or REGISTRY
    v = lambda b: "✅" if b else "-"

    print()
    print(f"{'ID':<40} {'Provider':<12} {'SDK':<12} {'V':>2} {'T':>2} {'FC':>2} {'TL':<7} {'Price':<14}")
    print("─" * 114)
    for m in models:
        tl = m.thinking_level or "-"
        price = {"very low": "very low", "medium": "medium"}.get(m.price, m.price)
        print(
            f"{m.id:<40} {m.provider:<12} {m.sdk:<12} "
            f"{v(m.vision):>2} {v(m.thinking):>2} {v(m.tool_calling):>2} "
            f"{tl:<7} {price:<14}"
        )
    print("─" * 114)
    print(f"Total: {len(models)} models")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Registry")
    parser.add_argument("--vision", action="store_true", default=False)
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--tool-calling", action="store_true", default=False)
    parser.add_argument("--open-source", action="store_true", default=False)
    parser.add_argument("--sdk", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Show all models and ignore filters")
    args = parser.parse_args()

    if args.all or not any([args.vision, args.thinking, args.tool_calling,
                            args.open_source, args.sdk, args.provider]):
        print_registry()
    else:
        filtered = list_models(
            vision=True if args.vision else None,
            thinking=True if args.thinking else None,
            tool_calling=True if args.tool_calling else None,
            open_source=True if args.open_source else None,
            sdk=args.sdk,
            provider=args.provider,
        )
        print_registry(filtered)
