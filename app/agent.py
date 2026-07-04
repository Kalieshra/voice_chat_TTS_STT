"""
Conversational agent backed by OpenAI or Anthropic models (user-selectable).

API keys are read from `.env` at the project root:
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
The agent replies in Egyptian Arabic by default so the STT → agent → TTS loop
stays in-dialect.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE, ".env"))

SYSTEM_PROMPT = (
    "انت مساعد ذكي مصري محترف، بتتكلم باللهجة المصرية بس بأسلوب رسمي ومهذب "
    "زي موظف خدمة عملاء محترم، مش عامية الشارع. "
    "خاطب المستخدم بـ (حضرتك) دايمًا، مش (انت). "
    "استخدم كلمات مؤدبة ورسمية زي: لو سمحت، من فضلك، تحت أمر حضرتك، يشرفني، "
    "اتفضل، ممكن، أقدر أساعد حضرتك، حاضر، طبعًا، بكل سرور، في خدمة حضرتك. "
    "ابعد خالص عن الكلمات السوقية زي: يلا، بص، جامد، زفت، اهو، يا صاحبي. "
    "مهم جدًا: رسمي مش معناه فصحى. ممنوع تمامًا الكلمات الفصحى دي، "
    "استبدلها بالمصري: (ماذا/ما) قول (إيه)، (كيف) قول (إزاي)، "
    "(يمكنني/أستطيع) قول (ممكن/أقدر)، (تريد/ترغب) قول (عايز/محتاج)، "
    "(أيضًا) قول (كمان أو بردو)، (الآن) قول (دلوقتي)، (يوجد) قول (فيه)، "
    "(لا يوجد) قول (مفيش)، (هذا) قول (ده)، (هل تفضل) قول (تحب)، "
    "وماتبدأش سؤال بـ (هل) خالص. "
    "أمثلة على الأسلوب الصح: (ممكن أساعد حضرتك في إيه النهاردة؟) — "
    "(حاضر يا فندم، تحت أمر حضرتك) — (يشرفني إني أساعد حضرتك) — "
    "(حضرتك محتاج إيه بالظبط؟). "
    "ومثال غلط ممنوع: (كيف يمكنني مساعدتك؟) و(ماذا تريد؟) و(أتمنى أن تكون بخير). "
    "خلي الرد مختصر وواضح ومحترم. "
    "وماتكتبش أي تشكيل حركات على الكلام. "
    "مهم جدًا: اكتب كلام منطوق عادي بس، وماتستخدمش أي رموز أو علامات "
    "زي @ # * _ ~ ^ ( ) [ ] { } < > / \\ | = + - ولا نجوم ولا شرط ولا "
    "قوايم بنقط ولا عناوين ولا رموز تعبيرية ولا أي تنسيق. "
    "استخدم بس نقطة وفاصلة وعلامة استفهام وتعجب لو محتاج. "
    "لو المستخدم كتب بالإنجليزي، رد بالإنجليزي بأسلوب رسمي مهذب بنفس القواعد دي."
)

# --- Model catalog ----------------------------------------------------------
# provider: "openai" | "anthropic".  reasoning=True => o-series decode rules.
MODELS = [
    # OpenAI
    {"id": "gpt-4o-mini", "label": "GPT-4o mini", "provider": "openai"},
    {"id": "gpt-4o", "label": "GPT-4o", "provider": "openai"},
    {"id": "gpt-4.1", "label": "GPT-4.1", "provider": "openai"},
    {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini", "provider": "openai"},
    {"id": "gpt-4.1-nano", "label": "GPT-4.1 nano", "provider": "openai"},
    {"id": "o4-mini", "label": "o4-mini (reasoning)", "provider": "openai",
     "reasoning": True},
    {"id": "o3-mini", "label": "o3-mini (reasoning)", "provider": "openai",
     "reasoning": True},
    # Anthropic
    {"id": "claude-opus-4-8", "label": "Claude Opus 4.8", "provider": "anthropic"},
    {"id": "claude-sonnet-5", "label": "Claude Sonnet 5", "provider": "anthropic"},
    {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5",
     "provider": "anthropic"},
    {"id": "claude-fable-5", "label": "Claude Fable 5", "provider": "anthropic"},
    {"id": "claude-3-5-sonnet-latest", "label": "Claude 3.5 Sonnet",
     "provider": "anthropic"},
    {"id": "claude-3-5-haiku-latest", "label": "Claude 3.5 Haiku",
     "provider": "anthropic"},
]

_BY_ID = {m["id"]: m for m in MODELS}
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class AgentError(RuntimeError):
    pass


import re

# Backstop: even with a strong prompt, GPT occasionally drops an MSA word into an
# otherwise-Egyptian reply. Swap the clearest, safe offenders to formal Egyptian.
_MSA_TO_EGY = {
    "كيف": "إزاي",
    "ماذا": "إيه",
    "يمكنني": "أقدر",
    "أيضًا": "كمان",
    "أيضاً": "كمان",
    "الآن": "دلوقتي",
}
_MSA_RE = [(re.compile(r"\b" + re.escape(k) + r"\b"), v) for k, v in _MSA_TO_EGY.items()]


def _egyptianize(text: str) -> str:
    for pat, repl in _MSA_RE:
        text = pat.sub(repl, text)
    return text


def _reload_env() -> None:
    load_dotenv(os.path.join(_BASE, ".env"), override=True)


def _key(provider: str) -> str:
    env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    return os.environ.get(env, "").strip()


def _key_ok(provider: str) -> bool:
    k = _key(provider)
    return bool(k) and not k.startswith("sk-your")


def provider_configured() -> dict:
    _reload_env()
    return {"openai": _key_ok("openai"), "anthropic": _key_ok("anthropic")}


def available_models() -> dict:
    cfg = provider_configured()
    models = [
        {**m, "configured": cfg.get(m["provider"], False)} for m in MODELS
    ]
    # Default to a model whose provider has a key, else the nominal default.
    default = DEFAULT_MODEL
    if default not in _BY_ID or not cfg.get(_BY_ID[default]["provider"]):
        for m in models:
            if m["configured"]:
                default = m["id"]
                break
    return {"models": models, "configured": cfg, "default": default}


def agent_available() -> bool:
    cfg = provider_configured()
    return cfg["openai"] or cfg["anthropic"]


def _build_messages(user_text: str, history: list[dict] | None):
    msgs = []
    if history:
        for turn in history[-8:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text.strip()})
    return msgs


def _ask_openai(model: str, messages: list, reasoning: bool, max_tokens: int,
                temperature: float) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=_key("openai"))
    full = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    kwargs = {"model": model, "messages": full}
    if reasoning:
        # o-series: no custom temperature, uses max_completion_tokens.
        kwargs["max_completion_tokens"] = max(max_tokens, 2000)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    reply = (resp.choices[0].message.content or "").strip()
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
             "completion_tokens": getattr(u, "completion_tokens", None)} if u else None
    return {"reply": reply, "usage": usage}


def _ask_anthropic(model: str, messages: list, max_tokens: int,
                   temperature: float) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=_key("anthropic"))
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,
        temperature=temperature,
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    reply = "".join(parts).strip()
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens": getattr(u, "input_tokens", None),
             "completion_tokens": getattr(u, "output_tokens", None)} if u else None
    return {"reply": reply, "usage": usage}


def ask_agent(
    user_text: str,
    history: list[dict] | None = None,
    model: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.5,
) -> dict:
    if not user_text or not user_text.strip():
        raise AgentError("Empty message.")
    _reload_env()

    model = model or DEFAULT_MODEL
    meta = _BY_ID.get(model)
    if meta is None:
        raise AgentError(f"Unknown model '{model}'.")
    provider = meta["provider"]

    if not _key_ok(provider):
        env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        raise AgentError(
            f"{provider.title()} not configured. Add {env}=... to the .env file."
        )

    messages = _build_messages(user_text, history)
    try:
        if provider == "openai":
            out = _ask_openai(
                model, messages, meta.get("reasoning", False), max_tokens, temperature
            )
        else:
            out = _ask_anthropic(model, messages, max_tokens, temperature)
    except AgentError:
        raise
    except Exception as e:  # noqa: BLE001
        raise AgentError(f"{provider.title()} request failed: {e}") from e

    reply = out["reply"]
    # Only Egyptianize Arabic replies (leave English answers untouched).
    if any("؀" <= ch <= "ۿ" for ch in reply):
        reply = _egyptianize(reply)
    return {"reply": reply, "model": model, "provider": provider,
            "usage": out.get("usage")}
