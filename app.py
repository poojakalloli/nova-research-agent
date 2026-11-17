"""
NOVA - Research & Calculator Agent (Streamlit + LangChain)

Run from VS Code terminal (Windows):
    streamlit run app.py

.env keys (add as many as you like; missing ones are skipped automatically):
    AI models, tried in this order:  GEMINI_API_KEY -> GROQ_API_KEY -> OPENROUTER_API_KEY -> HF_TOKEN
    Web search, tried in this order: TAVILY_API_KEY -> SERPER_API_KEY -> DuckDuckGo (no key needed)
Optional:
    GEMINI_MODEL, GEMINI_FALLBACK_MODEL, GROQ_MODEL, OPENROUTER_MODEL, HF_MODEL
    LLM_PROVIDER_ORDER=gemini,gemini_backup,groq,openrouter,huggingface
"""
import ast
import html
import inspect
import json
import logging
import operator
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=True)   # .env next to app.py (re-read on every rerun)
load_dotenv(override=True)             # also a .env in the folder you launched from

# Full-width buttons that work on both old and new Streamlit versions.
try:
    STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}
except (TypeError, ValueError):
    STRETCH = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nova")

# -------------------------------------------------
# 1. CONSTANTS
# -------------------------------------------------
# AI model providers. If one fails (busy, rate-limited, key problem) the next one is used.
# A provider without an API key in .env is skipped automatically.
LLM_PROVIDERS = {
    "gemini": {
        "label": "Gemini", "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "model_env": "GEMINI_MODEL", "model": "gemini-3.8-flash",
    },
    "gemini_backup": {
        "label": "Gemini (backup model)", "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "model_env": "GEMINI_FALLBACK_MODEL", "model": "gemini-2.5-flash",
    },
    "groq": {
        "label": "Groq", "keys": ("GROQ_API_KEY", "GROQ_KEY"),
        "model_env": "GROQ_MODEL", "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "label": "OpenRouter", "keys": ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPENROUTER_KEY"),
        "model_env": "OPENROUTER_MODEL", "model": "openai/gpt-oss-20b",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "huggingface": {
        "label": "Hugging Face", "keys": ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_KEY", "HF_API_KEY"),
        "model_env": "HF_MODEL", "model": "meta-llama/Llama-3.3-70B-Instruct",
        "base_url": "https://router.huggingface.co/v1",
    },
}
DEFAULT_PROVIDER_ORDER = "gemini,gemini_backup,groq,openrouter,huggingface"
MAX_INPUT_CHARS = 2000          # protects cost + prompt size
MAX_HISTORY_MESSAGES = 10       # how much past chat the agent can see
STRUCTURED_TOOL_NAME = "AgentResponse"  # internal tool LangChain uses for structured output
MAX_EXPONENT = 100
MAX_ABS_VALUE = 10 ** 100

SYSTEM_PROMPT = """You are NOVA, a helpful research and calculator assistant.

Rules:
- Use the web search tool for current events, news, prices, recent facts, or anything you are not sure about.
  Do not search for stable general knowledge, greetings or simple chat.
- Use the arithmetic tools (add, subtract, multiply, divide, power, modulus, square_root, percentage_of)
  or the calculator tool for ALL arithmetic. Never calculate in your head.
- Use the current_datetime tool for questions about today's date or the current time.
- Use the get_weather tool for current weather and forecasts of a city.
- If the search tool reports that search is unavailable, say so honestly and answer only from general knowledge.
- Only state what the search results support. If they do not answer the question, say so clearly.
- Never invent URLs. In `sources`, list only URLs that came from search results (empty list if you did not search).
- `answer`: clear, concise, well-formatted markdown.  `summary`: one or two plain sentences.
- You cannot play videos yourself. If asked to play something, tell the user to type: Play <song or video name>.
"""

TOOL_LABELS = {
    "web_search": "Web Search",
    "tavily_search": "Web Search",
    "calculator": "Calculator",
    "add": "Addition",
    "subtract": "Subtraction",
    "multiply": "Multiplication",
    "divide": "Division",
    "power": "Power",
    "modulus": "Modulus",
    "square_root": "Square root",
    "percentage_of": "Percentage",
    "current_datetime": "Date & Time",
    "get_weather": "Weather",
}
MATH_TOOL_NAMES = {
    "calculator", "add", "subtract", "multiply", "divide",
    "power", "modulus", "square_root", "percentage_of",
}

# The toolbar: one panel is shown at a time. Switching tools clears the previous result.
TOOLBAR = [
    "⌕ Tavily Search",
    "🧮 Arithmetic",
    "▶ YouTube Search",
    "🌦 Weather",
    "✉ Gmail",
    "📅 Calendar",
]
ARITH_OPS = ["Add (A + B)", "Multiply (A × B)", "Subtract (A − B)", "Divide (A ÷ B)"]


# -------------------------------------------------
# 2. PURE HELPERS (no Streamlit / no network)
# -------------------------------------------------
# ---- Safe arithmetic (no eval) ----
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node, depth=0):
    if depth > 50:
        raise ValueError("The expression is too complex.")
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, depth + 1))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"The exponent is too large (maximum {MAX_EXPONENT}).")
        result = _BIN_OPS[type(node.op)](left, right)
        if isinstance(result, complex):
            raise ValueError("The result is not a real number.")
        if abs(result) > MAX_ABS_VALUE:
            raise ValueError("The result is too large.")
        return result
    raise ValueError("Only numbers and + - * / // % ** ( ) are allowed.")


def safe_calculate(expression):
    """Evaluate a plain arithmetic expression safely. Raises ValueError on bad input."""
    expr = (
        str(expression)
        .strip()
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace(",", "")
    )
    if not expr or len(expr) > 200:
        raise ValueError("Please provide a short arithmetic expression.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError("That is not a valid arithmetic expression.")
    try:
        return _eval_node(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero is not possible.")
    except OverflowError:
        raise ValueError("The result is too large.")


def format_number(value):
    if isinstance(value, int):
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("The result is not a finite number.")
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


_NUM = r"-?\d+(?:\.\d+)?"
_SEP = r"(?:\s+(?:and|to|by)\s+|\s+)"
_NUM_LIST = rf"({_NUM}(?:{_SEP}{_NUM})+)"   # two or more numbers


def _paren(n):
    return f"({n})" if n.startswith("-") else n


def _nums(text):
    return re.findall(_NUM, text)


def _g(m, i):
    return _paren(m.group(i))


# (pattern, builder -> (expression, pretty_or_None), name used in the summary)
_WORD_FORMS = [
    (rf"(?:add|sum of|addition of|total of)\s+{_NUM_LIST}",
     lambda m: (" + ".join(_paren(n) for n in _nums(m.group(1))), None), "sum"),
    (rf"(?:multiply|multiplication of|product of)\s+{_NUM_LIST}",
     lambda m: (" * ".join(_paren(n) for n in _nums(m.group(1))), None), "product"),
    (rf"subtract\s+({_NUM})\s+from\s+({_NUM})",
     lambda m: (f"{_g(m, 2)} - {_g(m, 1)}", None), "difference"),
    (rf"difference between\s+({_NUM})\s+and\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} - {_g(m, 2)}", None), "difference"),
    (rf"divide\s+({_NUM})\s+by\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} / {_g(m, 2)}", None), "quotient"),
    (rf"remainder (?:of|when)\s+({_NUM})\s+(?:is\s+)?(?:divided\s+by|/|by)\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} % {_g(m, 2)}", None), "remainder"),
    (rf"({_NUM})\s*(?:%|percent)\s+of\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} / 100 * {_g(m, 2)}", f"{m.group(1)}% of {m.group(2)}"), "percentage"),
    (rf"(?:square root of|sqrt of|sqrt)\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} ** 0.5", f"√{m.group(1)}"), "square root"),
    (rf"square of\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} ** 2", f"{m.group(1)}²"), "square"),
    (rf"({_NUM})\s+squared",
     lambda m: (f"{_g(m, 1)} ** 2", f"{m.group(1)}²"), "square"),
    (rf"cube of\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} ** 3", f"{m.group(1)}³"), "cube"),
    (rf"({_NUM})\s+cubed",
     lambda m: (f"{_g(m, 1)} ** 3", f"{m.group(1)}³"), "cube"),
    (rf"(?:average|mean) of\s+{_NUM_LIST}",
     lambda m: (f"({' + '.join(_paren(n) for n in _nums(m.group(1)))}) / {len(_nums(m.group(1)))}",
                f"average of {', '.join(_nums(m.group(1)))}"), "average"),
    (rf"({_NUM})\s+(?:raised to(?: the power of)?|to the power of|power)\s+({_NUM})",
     lambda m: (f"{_g(m, 1)} ** {_g(m, 2)}", f"{m.group(1)}^{m.group(2)}"), "power"),
]
_WORD_SYMBOLS = [
    ("multiplied by", "*"),
    ("divided by", "/"),
    ("to the power of", "**"),
    ("times", "*"),
    ("plus", "+"),
    ("minus", "-"),
    ("modulo", "%"),
    ("mod", "%"),
]


def calculate_locally(question):
    """
    Handle clear arithmetic requests locally: exact, instant, free (no API call).
    Supports add, subtract, multiply, divide, power, remainder (mod), percentage,
    square root, square, cube, average and full expressions like (10+5)^2.
    Returns (answer, summary) or None when the question needs the agent.
    """
    q = question.lower().strip().replace(",", "")
    q = re.sub(
        r"^(?:please\s+)?(?:what(?:'s| is)|calculate|compute|solve|find|evaluate)\s+(?:the\s+)?",
        "",
        q,
    )
    q = q.strip(" ?.!=")
    if not q:
        return None

    expr = pretty = kind = None
    for pattern, builder, form_kind in _WORD_FORMS:
        match = re.fullmatch(pattern, q)
        if match:
            expr, pretty = builder(match)
            kind = form_kind
            break

    if expr is None:
        candidate = q
        for word, symbol in _WORD_SYMBOLS:
            candidate = re.sub(rf"\b{word}\b", symbol, candidate)
        candidate = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", "*", candidate)
        if (
            re.fullmatch(r"[\d\s.+\-*/^%()÷×]+", candidate)
            and re.search(r"\d", candidate)
            and re.search(r"[+*/^%÷×]|[\d)]\s*-", candidate)
        ):
            expr = candidate.strip()

    if expr is None:
        return None

    if pretty is None:
        pretty = expr.replace("**", "^").replace("*", "×").replace("/", "÷")
    try:
        result = format_number(safe_calculate(expr))
    except ValueError as exc:
        return f"⚠️ {exc}", "The expression could not be evaluated."

    answer = f"{pretty} = **{result}**"
    if kind:
        summary = f"The {kind} is {result}."
    else:
        summary = "Calculated instantly with a safe arithmetic evaluator."
    return answer, summary


def arithmetic_question(op, a, b, numbers_text=""):
    """Turn the toolbar inputs into a sentence calculate_locally() understands."""
    a_s, b_s = f"{a:.12g}", f"{b:.12g}"
    if op.startswith("Add"):
        return f"Add {a_s} and {b_s}"
    if op.startswith("Subtract"):
        return f"Subtract {b_s} from {a_s}"
    if op.startswith("Multiply"):
        return f"Multiply {a_s} by {b_s}"
    if op.startswith("Divide"):
        return f"Divide {a_s} by {b_s}"
    if op.startswith("Power"):
        return f"{a_s} to the power of {b_s}"
    if op.startswith("Remainder"):
        return f"remainder of {a_s} divided by {b_s}"
    if op.startswith("Percentage"):
        return f"{a_s}% of {b_s}"
    if op.startswith("Square root"):
        return f"square root of {a_s}"
    if op.startswith("Average"):
        nums = re.findall(_NUM, numbers_text.replace(",", " "))
        return ("average of " + " ".join(nums)) if len(nums) >= 2 else None
    return None


# ---- YouTube ----
_MEDIA_WORDS = (
    r"(?:song|songs|video|videos|bhajan|bhajans|aarti|chalis[a]?|trailer|music|"
    r"ringtone|remix|mashup|karaoke)"
)
_QUESTION_START = re.compile(
    r"^(?:what|who|why|how|when|where|which|explain|tell|define|describe|is|are|can|"
    r"does|do|write|give|summari[sz]e)\b",
    re.IGNORECASE,
)


def extract_youtube_query(question):
    """Detect requests to play/search a video, incl. short ones like 'dil kush song'."""
    text = question.strip()
    patterns = [
        r"^(?:please\s+)?play\s+(.+?)[.!?]*$",
        r"^(?:please\s+)?(?:listen to|watch)\s+(.+?)[.!?]*$",
        r"^(?:search|find)\s+(?:on\s+)?youtube\s+(?:for\s+)?(.+?)[.!?]*$",
        r"^youtube\s+(?:search|play)\s+(.+?)[.!?]*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            query = re.sub(r"\s+on youtube$", "", match.group(1), flags=re.IGNORECASE).strip()
            return query or None

    # Bare media request such as "Hanuman chalisa" or "dil kush song": no LLM needed.
    words = text.split()
    if (
        2 <= len(words) <= 8
        and not _QUESTION_START.match(text)
        and re.search(rf"\b{_MEDIA_WORDS}\s*[.!?]*$", text, flags=re.IGNORECASE)
    ):
        return text.rstrip(".!? ").strip()
    return None


def create_youtube_search_url(query):
    """Create a YouTube search link without requiring a YouTube API key."""
    return "https://www.youtube.com/results?search_query=" + quote_plus(query)


# ---- Messages, tools, sources ----
def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def content_to_text(content):
    """Gemini can return a string or a list of content blocks; always return text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(p for p in parts if p)
    return str(content or "")


def pretty_tool(name):
    if "search" in name.lower():
        return TOOL_LABELS["web_search"]
    return TOOL_LABELS.get(name, name.replace("_", " ").title())


def tool_status_text(name, args):
    args = args if isinstance(args, dict) else {}
    if "search" in name.lower():
        return f"🔎 Searching the web: {args.get('query', '')}"
    if name == "get_weather":
        return f"🌦 Checking the weather: {args.get('city', '')}"
    if name in MATH_TOOL_NAMES:
        values = ", ".join(str(v) for v in args.values())
        return f"🧮 {pretty_tool(name)}: {values}"
    if name == "current_datetime":
        return "🕒 Checking the current date and time"
    return f"🔧 Using {pretty_tool(name)}"


def _call_name(call):
    return call.get("name") if isinstance(call, dict) else getattr(call, "name", None)


def _call_args(call):
    return call.get("args") if isinstance(call, dict) else getattr(call, "args", {})


def get_tool_names(messages):
    """Names of the real tools the agent called (the internal structured-output tool is hidden)."""
    names = []
    search_provider = None
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = _call_name(call)
            if name and name != STRUCTURED_TOOL_NAME:
                label = pretty_tool(name)
                if label not in names:
                    names.append(label)
        if getattr(msg, "type", "") == "tool" and "search" in (getattr(msg, "name", "") or "").lower():
            text = str(getattr(msg, "content", ""))
            match = re.match(r"\s*\[via ([^\]]+)\]", text)
            if match:
                search_provider = match.group(1)
            elif "temporarily unavailable" in text[:100].lower():
                search_provider = "unavailable"
    if search_provider:
        names = [f"{n} ({search_provider})" if n == "Web Search" else n for n in names]
    return names


_URL_RE = re.compile(r"""https?://[^\s<>"'\]\)]+""")


def extract_search_urls(messages, limit=8):
    """Collect URLs that the search tool REALLY returned (so we never show invented links)."""
    urls = []
    for msg in messages:
        if getattr(msg, "type", "") != "tool":
            continue
        if "search" not in (getattr(msg, "name", "") or "").lower():
            continue
        for url in _URL_RE.findall(str(getattr(msg, "content", ""))):
            url = url.rstrip(".,;:!?)")
            if url not in urls:
                urls.append(url)
    return urls[:limit]


def merge_sources(model_sources, verified_urls):
    """Verified search URLs first; keep model-written names, drop model-written URLs that were not verified."""
    verified_keys = {u.lower().rstrip("/") for u in verified_urls}
    merged = list(verified_urls)
    seen = set(verified_keys)
    for source in model_sources or []:
        source = str(source).strip()
        if not source:
            continue
        key = source.lower().rstrip("/")
        if key in seen:
            continue
        if source.lower().startswith(("http://", "https://")):
            continue  # unverified URL, possibly hallucinated
        seen.add(key)
        merged.append(source)
    return merged


def source_markdown(source):
    if source.lower().startswith(("http://", "https://")):
        parsed = urlparse(source)
        label = parsed.netloc.removeprefix("www.") + (parsed.path.rstrip("/")[:40] if parsed.path else "")
        return f"- [{label}]({source})"
    return f"- {source}"


def build_agent_messages(chat, max_messages=MAX_HISTORY_MESSAGES):
    """Turn the visible chat into the message list the agent sees (gives it follow-up memory)."""
    out = []
    for item in chat:
        if item["role"] == "user":
            out.append({"role": "user", "content": item["content"]})
        elif item.get("error"):
            if out and out[-1]["role"] == "user":
                out.pop()  # drop the question whose answer failed
        else:
            out.append({"role": "assistant", "content": item.get("answer", "")})
    out = out[-max_messages:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


class ConfigError(Exception):
    """Raised when required configuration (API keys) is missing."""


class AllProvidersFailed(Exception):
    """Every configured AI provider failed. `errors` has one line per provider."""

    def __init__(self, errors):
        super().__init__("\n".join(errors))
        self.errors = errors
        self.count = len(errors)


def friendly_error(exc):
    text = str(exc)
    low = text.lower()
    if isinstance(exc, ConfigError):
        return text
    if isinstance(exc, AllProvidersFailed):
        if exc.count <= 1:
            return (
                "Only one AI provider is active, so there was nothing to switch to. Add another key "
                "(GROQ_API_KEY, OPENROUTER_API_KEY or HF_TOKEN) to your .env file. "
                "The sidebar shows which keys were found."
            )
        return f"All {exc.count} AI providers failed. Open 'Technical details' to see why each one failed."
    if any(k in low for k in ("429", "quota", "resource_exhausted", "rate limit")):
        return "The AI service is rate-limited right now. Please wait a minute and try again."
    if any(k in low for k in ("api key", "api_key", "401", "403", "permission", "unauthenticated")):
        return "An API key was rejected. Check the keys in your .env file (the sidebar shows which keys were found)."
    if any(k in low for k in ("503", "overloaded", "unavailable")):
        return "The AI service is busy at the moment. Please try again shortly."
    if any(k in low for k in ("timeout", "timed out")):
        return "The request took too long. Please try again or ask a simpler question."
    return "Something went wrong while processing your request."


def is_quota_error(exc):
    """Quota / rate-limit errors: retrying the same provider will not help, switch instead."""
    low = str(exc).lower()
    return any(k in low for k in ("429", "resource_exhausted", "quota", "rate limit"))


def is_retryable(exc):
    """Temporary problems (busy / rate-limited / timeout) where retry or a backup model can help."""
    if isinstance(exc, ConfigError):
        return False
    low = str(exc).lower()
    keys = ("503", "unavailable", "overloaded", "429", "resource_exhausted",
            "quota", "rate limit", "timeout", "timed out")
    return any(k in low for k in keys)


def redact_secrets(text):
    """Hide every configured API key before showing an error to the user."""
    names = {n for cfg in LLM_PROVIDERS.values() for n in cfg["keys"]}
    names |= {"TAVILY_API_KEY", "SERPER_API_KEY"}
    for name in names:
        value = os.getenv(name)
        if value and len(value) > 6:
            text = text.replace(value, "***")
    return text


def chat_to_markdown(chat):
    lines = ["# NOVA conversation", f"_Exported {datetime.now():%d %b %Y, %I:%M %p}_", ""]
    for item in chat:
        if item["role"] == "user":
            lines += [f"**You:** {item['content']}", ""]
        else:
            lines += [f"**NOVA:** {item.get('answer', '')}", ""]
            if item.get("summary"):
                lines += [f"> {item['summary']}", ""]
            for source in item.get("sources", []):
                lines.append(f"- {source}")
            if item.get("sources"):
                lines.append("")
    return "\n".join(lines)


# -------------------------------------------------
# 2b. AI MODEL PROVIDERS + WEB SEARCH PROVIDERS (automatic fallback)
# -------------------------------------------------
def get_provider_key(cfg):
    for name in cfg["keys"]:
        value = (os.getenv(name) or "").strip().strip("\"'")
        if value:
            return value
    return None


def llm_chain():
    """Providers that have a key, in the order they will be tried."""
    order = os.getenv("LLM_PROVIDER_ORDER", DEFAULT_PROVIDER_ORDER).split(",")
    chain = []
    for provider_id in (x.strip() for x in order):
        cfg = LLM_PROVIDERS.get(provider_id)
        if cfg and get_provider_key(cfg):
            chain.append({
                "id": provider_id,
                "label": cfg["label"],
                "model": os.getenv(cfg["model_env"], cfg["model"]),
            })
    return chain


def build_llm(provider_id, model_name, temperature):
    cfg = LLM_PROVIDERS[provider_id]
    key = get_provider_key(cfg)
    if not key:
        raise ConfigError(f"{cfg['label']} API key is missing in your .env file.")
    if provider_id.startswith("gemini"):
        os.environ["GOOGLE_API_KEY"] = key  # the Gemini integration reads this name
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, max_retries=1, timeout=60
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ConfigError("To use Groq / OpenRouter / Hugging Face run: pip install langchain-openai")
    # Groq, OpenRouter and Hugging Face all speak the OpenAI-compatible API.
    return ChatOpenAI(
        model=model_name, temperature=temperature, api_key=key,
        base_url=cfg["base_url"], max_retries=1, timeout=60,
    )


def _search_tavily(query, n):
    """Call the Tavily Search API directly and normalize its results."""
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip().strip("\"'")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add TAVILY_API_KEY=your_key to the .env "
            "file located next to app.py, then restart Streamlit."
        )

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max(1, min(int(n), 10)),
            "include_answer": True,
            "include_raw_content": False,
        },
        timeout=25,
    )

    if not response.ok:
        # Do not include the request URL or API key in the displayed error.
        detail = response.text[:500]
        raise RuntimeError(
            f"Tavily API returned HTTP {response.status_code}. "
            f"Check that your Tavily key is valid and has available credits. "
            f"API message: {detail}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Tavily returned a response that was not valid JSON.") from exc

    items = data.get("results", []) if isinstance(data, dict) else []
    tavily_answer = data.get("answer", "") if isinstance(data, dict) else ""
    normalized = []
    for result in items:
        if not isinstance(result, dict):
            continue
        normalized.append({
            "title": result.get("title") or "Untitled result",
            "url": result.get("url") or "",
            "snippet": result.get("content") or result.get("snippet") or "",
        })
    if normalized and tavily_answer:
        normalized[0]["tavily_answer"] = tavily_answer
    return normalized


def _search_serper(query, n):
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY not set")
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": n},
        timeout=15,
    )
    response.raise_for_status()
    items = response.json().get("organic", [])[:n]
    return [{"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")} for r in items]


def _search_duckduckgo(query, n):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    items = DDGS().text(query, max_results=n)
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in items]


SEARCH_PROVIDERS = [
    ("Tavily", _search_tavily),
    ("Serper", _search_serper),
    ("DuckDuckGo", _search_duckduckgo),
]


def format_search_results(label, results):
    lines = [f"[via {label}]"]
    for i, r in enumerate(results, 1):
        snippet = (r.get("snippet") or "").strip().replace("\n", " ")[:500]
        lines.append(f"{i}. {(r.get('title') or '').strip()}\n   URL: {r.get('url', '')}\n   {snippet}")
    return "\n".join(lines)


def collect_search_results(query, max_results, providers=None):
    """Try each search provider in turn. Returns (provider_label, results) or (None, [])."""
    for label, search_fn in providers or SEARCH_PROVIDERS:
        try:
            results = search_fn(query, max_results)
            if results:
                return label, results
            logger.warning("Search provider %s returned no results", label)
        except Exception as exc:
            logger.warning("Search provider %s failed: %s", label, exc)
    return None, []


def run_web_search(query, max_results, providers=None):
    """Search tool for the agent; never crashes the agent if all providers fail."""
    label, results = collect_search_results(query, max_results, providers)
    if results:
        return format_search_results(label, results)
    return (
        "Web search is temporarily unavailable. Tell the user you could not search the web "
        "right now and answer only from general knowledge."
    )


def search_only_answer(question, results):
    """Markdown list of web results, used when every AI model is down."""
    lines = [
        f"The AI models are busy or rate-limited right now, so here are web results for **{question}**:",
        "",
    ]
    for r in results:
        title = (r.get("title") or r.get("url") or "Result").strip()
        snippet = (r.get("snippet") or "").strip().replace("\n", " ")[:220]
        lines.append(f"- [{title}]({r.get('url', '')}) — {snippet}")
    return "\n".join(lines)


def search_only_item(question, max_results, started):
    """Last-resort answer without any AI model: raw web results (+ YouTube link for short queries)."""
    label, results = collect_search_results(question, max_results)
    if not results:
        return None
    youtube_search = None
    if len(question.split()) <= 4 and not _QUESTION_START.match(question.strip()):
        youtube_search = {"query": question, "url": create_youtube_search_url(question)}
    return make_item(
        search_only_answer(question, results),
        "AI answer unavailable, showing raw web results instead. Try again in a minute.",
        sources=[r["url"] for r in results if r.get("url")],
        tools=[f"Web Search ({label})"],
        elapsed=time.perf_counter() - started,
        youtube_search=youtube_search,
    )


def make_search_tool(max_results):
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information. Returns titles, URLs and snippets."""
        return run_web_search(query, max_results)

    return web_search


# -------------------------------------------------
# 2c. WEATHER (Open-Meteo, free, no API key)
# -------------------------------------------------
WEATHER_CODES = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"), 48: ("Fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy drizzle", "🌧️"),
    56: ("Freezing drizzle", "🌧️"), 57: ("Freezing drizzle", "🌧️"),
    61: ("Light rain", "🌦️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"), 77: ("Snow grains", "🌨️"),
    80: ("Light showers", "🌦️"), 81: ("Showers", "🌧️"), 82: ("Violent showers", "⛈️"),
    85: ("Snow showers", "🌨️"), 86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm with hail", "⛈️"), 99: ("Thunderstorm with hail", "⛈️"),
}
_WEATHER_PATTERNS = [
    r"^(?:please\s+)?(?:what(?:'s| is)\s+the\s+|how(?:'s| is)\s+the\s+|tell me the\s+|show me the\s+|check the\s+|get the\s+)?"
    r"(?:current\s+|today'?s\s+)?(?:weather|temperature|forecast)(?:\s+(?:report|update))?\s+(?:in|at|of|for|near)\s+(.+?)[?.!]*$",
    r"^(?:please\s+)?(?:weather|temperature|forecast)\s+(.+?)[?.!]*$",
    r"^(.+?)\s+(?:weather|temperature|forecast)(?:\s+today)?[?.!]*$",
    r"^(?:is it|will it)\s+(?:rain|raining|going to rain|sunny|hot|cold)\s+(?:in|at)\s+(.+?)[?.!]*$",
]


def weather_text(code):
    return WEATHER_CODES.get(code, ("Unknown", "🌡️"))


def extract_weather_city(question):
    """'weather in Pune', 'Delhi weather', 'temperature in Goa today' -> city, else None."""
    text = question.strip()
    for pattern in _WEATHER_PATTERNS:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        city = re.sub(r"\s+(?:today|now|tomorrow|tonight|right now|this week)$", "", match.group(1).strip(), flags=re.IGNORECASE)
        words = city.split()
        if (
            city
            and len(words) <= 4
            and not _QUESTION_START.match(city)
            and city.lower() not in {"the", "today", "now", "current", "my area", "here", "me"}
            and re.fullmatch(r"[A-Za-z .,'\-]+", city)
        ):
            return city
    return None


def fetch_weather(city, days=3):
    """Look up a city and its forecast. Raises LookupError if the city is not found."""
    name = city.split(",")[0].strip()
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise LookupError(f"Could not find a place called '{city}'.")
    place = results[0]
    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=10,
    )
    forecast.raise_for_status()
    return {"place": place, "forecast": forecast.json()}


def _num(value):
    return "–" if value is None else f"{value:.0f}"


def _at(values, index):
    return values[index] if values and index < len(values) else None


def weather_place_name(data):
    place = data["place"]
    return ", ".join(x for x in (place.get("name"), place.get("admin1"), place.get("country")) if x)


def format_weather(data):
    """Markdown version of the weather (used by the AI tool and the chat download)."""
    current = data["forecast"].get("current", {})
    daily = data["forecast"].get("daily", {})
    desc, icon = weather_text(current.get("weather_code"))
    lines = [
        f"### {icon} {weather_place_name(data)}",
        f"**{_num(current.get('temperature_2m'))}°C** · {desc} · feels like {_num(current.get('apparent_temperature'))}°C",
        f"💧 Humidity {_num(current.get('relative_humidity_2m'))}% · 💨 Wind {_num(current.get('wind_speed_10m'))} km/h"
        f" · 🌧️ Rain now {current.get('precipitation') or 0} mm",
        "",
        "| Day | Weather | Min / Max | Rain chance |",
        "|---|---|---|---|",
    ]
    for i, day in enumerate(daily.get("time", [])):
        d_desc, d_icon = weather_text(_at(daily.get("weather_code"), i))
        try:
            label = datetime.strptime(day, "%Y-%m-%d").strftime("%a %d %b")
        except ValueError:
            label = day
        lines.append(
            f"| {label} | {d_icon} {d_desc} | {_num(_at(daily.get('temperature_2m_min'), i))}° / "
            f"{_num(_at(daily.get('temperature_2m_max'), i))}° | {_num(_at(daily.get('precipitation_probability_max'), i))}% |"
        )
    return "\n".join(lines)


def weather_item(data, started):
    current = data["forecast"].get("current", {})
    desc, _ = weather_text(current.get("weather_code"))
    item = make_item(
        format_weather(data),  # kept for chat download / follow-up questions; the card is drawn from weather_data
        f"{_num(current.get('temperature_2m'))}°C and {desc.lower()} in {weather_place_name(data)}.",
        sources=["https://open-meteo.com/"],
        tools=["Weather (Open-Meteo)"],
        elapsed=time.perf_counter() - started,
    )
    item["weather_data"] = data
    return item


# -------------------------------------------------
# 3. PAGE + UI STYLING
# -------------------------------------------------
st.set_page_config(
    page_title="NOVA · Research & Calculator Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {
  background: radial-gradient(circle at 85% 0%, #eeeaff 0, transparent 28rem),
              linear-gradient(180deg,#fafaff 0%,#f5f6fb 100%);
  color:#20223a;
}
.block-container {max-width:1120px;padding-top:1.5rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#1e1b38;}
[data-testid="stSidebar"] * {color:#f7f5ff;}
.hero {
  padding:30px 32px;border-radius:24px;color:white;
  background:linear-gradient(115deg,#242044 0%,#493b91 58%,#8067ff 100%);
  box-shadow:0 16px 40px rgba(66,52,135,.18);margin-bottom:20px;
}
.hero-kicker {font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#d8d1ff;font-weight:700;}
.hero h1 {font:800 30px Manrope,sans-serif;color:white;margin:8px 0;}
.hero p {font-size:14px;color:#e5e0ff;margin:0;}
.stat {
  background:white;border:1px solid #e9e8f2;border-radius:16px;
  padding:15px 17px;box-shadow:0 4px 15px rgba(35,35,70,.035);
}
.stat small {color:#858ba0;text-transform:uppercase;letter-spacing:1px;font-size:10px;font-weight:700;}
.stat strong {display:block;font:800 19px Manrope,sans-serif;color:#252541;margin-top:5px;}
.pill {
  display:inline-block;padding:5px 10px;border-radius:20px;
  background:#efedff;color:#5946c8;font-size:11px;font-weight:700;margin:3px 4px 3px 0;
}
[data-testid="stSidebar"] .pill {background:#efedff;color:#5946c8;}
div.stButton>button, div[data-testid="stDownloadButton"]>button {
  width:100%;border-radius:11px;border:1px solid #e3dfff;background:white;
  color:#5847c8;font-weight:700;min-height:40px;
}
div.stButton>button:hover, div[data-testid="stDownloadButton"]>button:hover {
  border-color:#806aff;background:#f7f5ff;color:#4937b6;
}
div[data-testid="stChatInput"] {
  border:1px solid #ded9fa;border-radius:16px;background:white;
  box-shadow:0 7px 22px rgba(55,44,120,.07);
}
.stChatMessage {border-radius:15px;}
/* Tool selector: one line, full width, even distance between the six buttons */
div[data-testid="stButtonGroup"] {width:100%;display:flex;flex-wrap:nowrap;gap:10px;overflow-x:auto;}
div[data-testid="stButtonGroup"] button {
  flex:1 1 0;min-width:max-content;min-height:46px;justify-content:center;
  white-space:nowrap;border-radius:12px;font-weight:600;
}
div[role="radiogroup"][aria-label="Tool"] {width:100%;flex-wrap:nowrap;gap:10px;}
[data-testid="stSidebar"] div.stButton > button,
[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button {
    background:#ffffff !important;color:#4937b6 !important;
    border:1px solid #ded9fa !important;opacity:1 !important;
}
[data-testid="stSidebar"] div.stButton > button p,
[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button p {
    color:#4937b6 !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# 4. STRUCTURED RESPONSE + TOOLS + AGENT
# -------------------------------------------------
class AgentResponse(BaseModel):
    answer: str = Field(description="The final answer to the user's question, in markdown.")
    summary: str = Field(description="A short one or two sentence summary.")
    sources: list[str] = Field(
        default_factory=list,
        description="URLs that came from search results (empty if no search was used).",
    )


def _run_math(expression):
    try:
        return format_number(safe_calculate(expression))
    except ValueError as exc:
        return f"Error: {exc}"


@tool
def calculator(expression: str) -> str:
    """Evaluate any arithmetic expression exactly. Supports + - * / // % ** and parentheses.
    Example: '(450 + 275) * 3'."""
    return _run_math(expression)


@tool
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return _run_math(f"({a}) + ({b})")


@tool
def subtract(a: float, b: float) -> str:
    """Subtract b from a (a - b)."""
    return _run_math(f"({a}) - ({b})")


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    return _run_math(f"({a}) * ({b})")


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b (a / b)."""
    return _run_math(f"({a}) / ({b})")


@tool
def power(base: float, exponent: float) -> str:
    """Raise base to the given exponent (base ** exponent)."""
    return _run_math(f"({base}) ** ({exponent})")


@tool
def modulus(a: float, b: float) -> str:
    """Remainder when a is divided by b (a % b)."""
    return _run_math(f"({a}) % ({b})")


@tool
def square_root(number: float) -> str:
    """Square root of a non-negative number."""
    return _run_math(f"({number}) ** 0.5")


@tool
def percentage_of(percent: float, value: float) -> str:
    """Calculate percent% of value, e.g. 20% of 500."""
    return _run_math(f"({percent}) / 100 * ({value})")


@tool
def current_datetime() -> str:
    """Get the current local date, weekday and time of the machine running this app."""
    return datetime.now().astimezone().strftime("%A, %d %B %Y, %I:%M %p %Z (UTC%z)")


@tool
def get_weather(city: str) -> str:
    """Get current weather and a 3-day forecast for a city (free, no key). Example: 'Mumbai'."""
    try:
        return format_weather(fetch_weather(city))
    except LookupError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: the weather service is unavailable ({exc})."


AGENT_TOOLS = [
    calculator, add, subtract, multiply, divide,
    power, modulus, square_root, percentage_of, current_datetime, get_weather,
]


@st.cache_resource(show_spinner=False)
def get_agent(provider_id: str, model_name: str, temperature: float, max_results: int):
    """Build the agent once per provider/settings combination (cached for fast requests)."""
    llm = build_llm(provider_id, model_name, temperature)
    logger.info("Creating agent | provider=%s model=%s", provider_id, model_name)
    return create_agent(
        model=llm,
        tools=[make_search_tool(max_results), *AGENT_TOOLS],
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(AgentResponse),
    )


def make_item(answer, summary="", sources=None, tools=None, elapsed=None,
              youtube_search=None, model_used=None):
    return {
        "role": "assistant",
        "answer": answer,
        "summary": summary,
        "sources": sources or [],
        "tools": tools or [],
        "elapsed": elapsed,
        "youtube_search": youtube_search,
        "model_used": model_used,
    }


def run_agent(messages, provider, temperature, max_results, status):
    """Stream the agent so the user sees live progress (search / calculate steps)."""
    agent = get_agent(provider["id"], provider["model"], temperature, max_results)
    final_state = None
    seen = len(messages)

    for state in agent.stream({"messages": messages}, stream_mode="values"):
        final_state = state
        current = state.get("messages", [])
        for msg in current[seen:]:
            for call in getattr(msg, "tool_calls", None) or []:
                name = _call_name(call)
                if name and name != STRUCTURED_TOOL_NAME:
                    status.write(tool_status_text(name, _call_args(call)))
        seen = max(seen, len(current))

    if final_state is None:
        raise RuntimeError("The agent returned no result.")

    all_messages = final_state.get("messages", [])
    structured = final_state.get("structured_response")
    if structured is not None:
        answer = get_field(structured, "answer", "")
        summary = get_field(structured, "summary", "")
        model_sources = get_field(structured, "sources", []) or []
    else:
        raw = content_to_text(all_messages[-1].content) if all_messages else ""
        try:
            data = json.loads(raw)
            answer = data.get("answer", raw)
            summary = data.get("summary", "")
            model_sources = data.get("sources", []) or []
        except (json.JSONDecodeError, TypeError, AttributeError):
            answer, summary, model_sources = raw, "", []

    sources = merge_sources(model_sources, extract_search_urls(all_messages))
    return answer, summary, sources, get_tool_names(all_messages)


def tavily_direct_item(search_query, max_results, started):
    """Toolbar search: Tavily answers directly (no language model needed)."""
    try:
        results = _search_tavily(search_query, max_results)
    except Exception as exc:
        logger.exception("Tavily toolbar search failed")
        return make_item(
            f"Tavily search failed: {redact_secrets(str(exc))}",
            "Check TAVILY_API_KEY in the .env file and confirm the key is active.",
            tools=["Tavily Search"],
            elapsed=time.perf_counter() - started,
        )

    if not results:
        return make_item(
            f"Tavily did not return results for **{search_query}**.",
            "Try different keywords or check that your Tavily API key has search credits.",
            tools=["Tavily Search"],
            elapsed=time.perf_counter() - started,
        )

    # Answer-only mode: show Tavily's answer, plus source links in a collapsed list.
    answer_text = str(results[0].get("tavily_answer", "")).strip()
    if not answer_text:
        for result in results[:3]:
            candidate = re.sub(r"\s+", " ", str(result.get("snippet") or "")).strip()
            if candidate:
                answer_text = candidate
                break
    if not answer_text:
        answer_text = "Tavily found no usable answer. Please try rephrasing your question."

    links = [
        str(r.get("url", "")).strip() for r in results
        if str(r.get("url", "")).strip().startswith(("https://", "http://"))
    ]
    return make_item(
        answer_text, "", sources=links[:5], tools=["Tavily Search"],
        elapsed=time.perf_counter() - started,
    )


def answer_question(question, chain, temperature, max_results, isolated_request=False):
    """Route a question: Tavily toolbar -> YouTube -> weather -> calculator -> AI agent (with fallback)."""
    started = time.perf_counter()

    # The Tavily toolbar sends this prefix so Tavily runs the search directly.
    tavily_prefix = "Search the web for "
    if question.strip().lower().startswith(tavily_prefix.lower()):
        search_query = question.strip()[len(tavily_prefix):].strip()
        if not search_query:
            raise ValueError("Enter a topic to search with Tavily.")
        return tavily_direct_item(search_query, max_results, started)

    youtube_query = extract_youtube_query(question)
    if youtube_query:
        return make_item(
            f"Ready to search YouTube for **{youtube_query}**.",
            "Open the YouTube results and choose a video to play. No YouTube API key is needed.",
            tools=["YouTube Search link"],
            elapsed=time.perf_counter() - started,
            youtube_search={"query": youtube_query, "url": create_youtube_search_url(youtube_query)},
        )

    weather_city = extract_weather_city(question)
    if weather_city:
        try:
            return weather_item(fetch_weather(weather_city), started)
        except Exception as exc:  # not found / offline: let the agent try instead
            logger.info("Direct weather lookup failed (%s); falling back to the agent", exc)

    local = calculate_locally(question)
    if local is not None:
        answer, summary = local
        return make_item(
            answer, summary, tools=["Calculator"], elapsed=time.perf_counter() - started
        )

    if not chain:
        raise ConfigError(
            "No AI model key found. Add at least one of GEMINI_API_KEY, GROQ_API_KEY, "
            "OPENROUTER_API_KEY or HF_TOKEN to your .env file."
        )

    # Toolbar actions are independent tasks, not follow-ups to the previous chat.
    messages = (
        [{"role": "user", "content": question}]
        if isolated_request
        else build_agent_messages(st.session_state.chat)
    )

    first_error = None
    errors = []
    with st.status("Thinking…", expanded=True) as status:
        for index, provider in enumerate(chain):
            for attempt in range(2):  # one quick retry for temporary errors
                try:
                    answer, summary, sources, tools_used = run_agent(
                        messages, provider, temperature, max_results, status
                    )
                except Exception as exc:
                    first_error = first_error or exc
                    logger.warning(
                        "%s (%s) failed on attempt %d: %s",
                        provider["label"], provider["model"], attempt + 1, exc,
                    )
                    if attempt == 0 and is_retryable(exc) and not is_quota_error(exc):
                        status.write(f"⏳ {provider['label']} is busy, retrying…")
                        time.sleep(3)
                        continue
                    errors.append(f"{provider['label']} ({provider['model']}): {str(exc)[:220]}")
                    break
                else:
                    status.update(label="Done", state="complete", expanded=False)
                    return make_item(
                        answer, summary, sources, tools_used,
                        elapsed=time.perf_counter() - started,
                        model_used=None if index == 0 else f"{provider['label']} · {provider['model']}",
                    )
            if index + 1 < len(chain):
                status.write(
                    f"↪ {provider['label']} is not working — switching to {chain[index + 1]['label']}…"
                )

        if not isinstance(first_error, ConfigError):
            fallback = search_only_item(question, max_results, started)
            if fallback is not None:
                status.update(label="AI unavailable, showing web results", state="complete", expanded=False)
                return fallback
        raise AllProvidersFailed(errors)


# -------------------------------------------------
# 5. RENDERING
# -------------------------------------------------
def show_youtube_videos(youtube_search):
    st.link_button("▶ Open YouTube results", youtube_search["url"])


def render_weather_card(data):
    """Render current conditions and a short forecast in a clean card."""
    current = data["forecast"].get("current", {})
    daily = data["forecast"].get("daily", {})
    description, icon = weather_text(current.get("weather_code"))

    with st.container(border=True):
        st.markdown(f"### {icon} {weather_place_name(data)}")
        st.markdown(
            f"<div style='font-size:38px;font-weight:750;line-height:1.2'>"
            f"{_num(current.get('temperature_2m'))}°C</div>"
            f"<div style='color:#777;margin-bottom:14px'>{description} · "
            f"Feels like {_num(current.get('apparent_temperature'))}°C</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Humidity", f"{_num(current.get('relative_humidity_2m'))}%")
        m2.metric("Wind", f"{_num(current.get('wind_speed_10m'))} km/h")
        m3.metric("Rain now", f"{current.get('precipitation') or 0:g} mm")

        days = daily.get("time", [])
        if days:
            st.markdown("---")
            st.markdown("**3-day forecast**")
            forecast_cols = st.columns(min(3, len(days)))
            for i, col in enumerate(forecast_cols):
                try:
                    day_label = datetime.strptime(days[i], "%Y-%m-%d").strftime("%a %d %b")
                except ValueError:
                    day_label = days[i]
                day_desc, day_icon = weather_text(_at(daily.get("weather_code"), i))
                low = _num(_at(daily.get("temperature_2m_min"), i))
                high = _num(_at(daily.get("temperature_2m_max"), i))
                rain = _num(_at(daily.get("precipitation_probability_max"), i))
                with col:
                    st.markdown(
                        f"**{day_label}**  \n{day_icon} {day_desc}  \n"
                        f"**{low}° / {high}°**  \nRain: {rain}%"
                    )
        st.markdown("[Weather source: Open-Meteo](https://open-meteo.com/)")


def render_assistant(item):
    """Show a clean chat response: answer first, then source links only."""
    if item.get("weather_data"):
        render_weather_card(item["weather_data"])
        return

    if item.get("error"):
        st.error(item["answer"])
        if item.get("details"):
            with st.expander("Technical details"):
                st.text(item["details"])
        return

    st.markdown(item.get("answer") or "The agent returned an empty answer.")

    # Source URLs only: no titles, snippets or tool metadata.
    source_links = [
        source for source in (item.get("sources") or [])
        if isinstance(source, str) and source.startswith(("https://", "http://"))
    ]
    if source_links:
        with st.expander("Source links"):
            for index, source in enumerate(source_links, start=1):
                st.markdown(f"[Source {index}]({source})")

    if item.get("youtube_search"):
        show_youtube_videos(item["youtube_search"])


# -------------------------------------------------
# 6. SESSION STATE + SIDEBAR
# -------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []


def reset_chat():
    """Clear the conversation and any queued toolbar request."""
    st.session_state["chat"] = []
    st.session_state.pop("pending_question", None)
    st.session_state.pop("pending_isolated", None)


def on_tool_change():
    """Runs the moment the user picks another tool: start fresh (old result disappears)."""
    if st.session_state.get("auto_refresh", True):
        reset_chat()


LLM_CHAIN = llm_chain()
PRIMARY_LABEL = LLM_CHAIN[0]["model"] if LLM_CHAIN else "No API key"

with st.sidebar:
    st.markdown("## ✦ NOVA")
    st.caption("RESEARCH & CALCULATOR AGENT")
    st.markdown("---")
    st.markdown("### Connected tools")
    st.markdown(
        '<span class="pill">⌕ Web Search</span>'
        '<span class="pill">🧮 Arithmetic</span>'
        '<span class="pill">🕒 Date &amp; Time</span>'
        '<span class="pill">▶ YouTube</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Settings")
    st.checkbox(
        "Auto-refresh when I switch tools", value=True, key="auto_refresh",
        help="When on, picking another tool clears the previous result so each tool starts fresh.",
    )
    temperature = st.slider(
        "Creativity (temperature)", 0.0, 1.0, 0.0, 0.1,
        help="0 = precise and consistent. Higher = more creative.",
    )
    max_results = st.slider("Web results per search", 1, 6, 3)
    st.markdown("---")
    st.markdown("### Status")
    st.caption(
        ".env found ✅" if ENV_PATH.exists() else f".env NOT found ❌ (put it in {ENV_PATH.parent})"
    )
    st.caption("Active order: " + (" → ".join(c["label"] for c in LLM_CHAIN) or "none"))
    st.caption("AI models (🟢 = key found)")
    for provider_id in DEFAULT_PROVIDER_ORDER.split(","):
        cfg = LLM_PROVIDERS[provider_id]
        st.caption(f"{'🟢' if get_provider_key(cfg) else '🔴'} {cfg['label']}")
    st.caption("Web search (tried in order)")
    st.caption(f"{'🟢' if os.getenv('TAVILY_API_KEY') else '🔴'} Tavily")
    st.caption(f"{'🟢' if os.getenv('SERPER_API_KEY') else '🔴'} Serper")
    st.caption("🟢 DuckDuckGo (no key needed)")
    st.markdown("---")
    st.button("＋ New chat", key="new_chat_button", on_click=reset_chat, type="secondary", **STRETCH)
    st.download_button(
        "⬇ Download chat",
        data=chat_to_markdown(st.session_state.chat),
        file_name="nova_chat.md",
        mime="text/markdown",
        disabled=not st.session_state.chat,
        **STRETCH,
    )
    st.markdown("---")
    st.caption("Tip: Ask a calculation, a research question, a follow-up, or say “Play <song>”.")

# -------------------------------------------------
# 7. HEADER
# -------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">✦ Your intelligent workspace</div>
  <h1>Research smarter. Calculate faster.</h1>
  <p>Search with Tavily, calculate, find YouTube videos, check weather, use Gmail, and Google Calendar.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="stat"><small>Language model</small><strong>{html.escape(PRIMARY_LABEL)}</strong></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="stat"><small>Available tools</small><strong>{len(AGENT_TOOLS) + 2} tools</strong></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown('<div class="stat"><small>Output</small><strong>Structured response</strong></div>', unsafe_allow_html=True)

# -------------------------------------------------
# 8. NOVA TOOLBAR: pick a tool, only that tool's panel is shown.
#    Switching tools triggers on_tool_change() = automatic refresh.
# -------------------------------------------------
# Space between the three stat boxes above and the tool selector below.
st.markdown('<div style="height:34px"></div>', unsafe_allow_html=True)

if hasattr(st, "segmented_control"):
    # Newer Streamlit can stretch the control over the full width by itself.
    seg_kwargs = {"width": "stretch"} if "width" in inspect.signature(st.segmented_control).parameters else {}
    st.segmented_control(
        "Tool", TOOLBAR, default=TOOLBAR[0], key="active_tool",
        on_change=on_tool_change, label_visibility="collapsed", **seg_kwargs,
    )
else:  # older Streamlit versions
    st.radio(
        "Tool", TOOLBAR, horizontal=True, key="active_tool",
        on_change=on_tool_change, label_visibility="collapsed",
    )
active_tool = st.session_state.get("active_tool") or TOOLBAR[0]

# Small gap between the tool selector and the tool panel.
st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

with st.container(border=True):
    if active_tool == TOOLBAR[0]:
        st.markdown("### ⌕ Tavily Search")
        tavily_query = st.text_input(
            "What would you like to research?",
            placeholder="e.g. Latest developments in AI agents", key="tavily_toolbar_query",
        )
        if st.button("Search with Tavily", key="tavily_toolbar_go", **STRETCH):
            if tavily_query.strip():
                st.session_state.pending_question = "Search the web for " + tavily_query.strip()
                st.session_state.pending_isolated = True
                st.rerun()
            else:
                st.warning("Enter a topic to search.")

    elif active_tool == TOOLBAR[1]:
        st.markdown("### 🧮 Arithmetic")
        arith_op = st.selectbox("Operation", ARITH_OPS, key="arith_op")
        col_a, col_b = st.columns(2)
        with col_a:
            num_a = st.number_input("A", value=0.0, format="%g", key="arith_a")
        with col_b:
            num_b = st.number_input("B", value=0.0, format="%g", key="arith_b")
        if st.button("Calculate", key="arith_go", **STRETCH):
            arith_q = arithmetic_question(arith_op, num_a, num_b)
            if arith_q:
                st.session_state.pending_question = arith_q
                st.session_state.pending_isolated = True
                st.rerun()

    elif active_tool == TOOLBAR[2]:
        st.markdown("### ▶ YouTube Search")
        yt_input = st.text_input(
            "Song or video name",
            placeholder="e.g. Kesariya Arijit Singh", key="youtube_toolbar_query",
        )
        if st.button("Search YouTube", key="youtube_toolbar_go", **STRETCH):
            if yt_input.strip():
                st.session_state.pending_question = "Play " + yt_input.strip() + " on YouTube"
                st.session_state.pending_isolated = True
                st.rerun()
            else:
                st.warning("Enter a song or video name.")

    elif active_tool == TOOLBAR[3]:
        st.markdown("### 🌦 Weather")
        weather_input = st.text_input("City", key="weather_city", placeholder="e.g. Mumbai")
        if st.button("Get weather", key="weather_go", **STRETCH):
            if weather_input.strip():
                st.session_state.pending_question = f"Weather in {weather_input.strip()}"
                st.session_state.pending_isolated = True
                st.rerun()
            else:
                st.warning("Type a city name first.")

    elif active_tool == TOOLBAR[4]:
        st.markdown("### ✉ Gmail Search & Draft")
        st.caption("Uses the Gmail account signed in to your browser.")
        gmail_query = st.text_input(
            "Search Gmail", placeholder="e.g. from:teacher subject:project", key="gmail_search_query",
        )
        if gmail_query.strip():
            st.link_button(
                "Open Gmail search",
                "https://mail.google.com/mail/u/0/#search/" + quote_plus(gmail_query.strip()),
            )

        st.markdown("#### Draft an email")
        draft_to = st.text_input("To", key="gmail_draft_to", placeholder="name@example.com")
        draft_subject = st.text_input("Subject", key="gmail_draft_subject")
        draft_body = st.text_area("Message", key="gmail_draft_body", height=110)
        if draft_to.strip() or draft_subject.strip() or draft_body.strip():
            st.link_button(
                "Open prefilled draft in Gmail",
                "https://mail.google.com/mail/u/0/?view=cm&fs=1"
                + "&to=" + quote_plus(draft_to.strip())
                + "&su=" + quote_plus(draft_subject.strip())
                + "&body=" + quote_plus(draft_body),
            )

    elif active_tool == TOOLBAR[5]:
        st.markdown("### 📅 Google Calendar")
        st.caption("Opens Google Calendar with the event filled in, so you can review it before saving.")
        calendar_title = st.text_input(
            "Event title", placeholder="e.g. Project presentation", key="google_calendar_title",
        )
        calendar_date = st.date_input("Event date", value=datetime.now().date(), key="google_calendar_date")
        cal_col_start, cal_col_end = st.columns(2)
        with cal_col_start:
            calendar_start = st.time_input(
                "Start time",
                value=datetime.now().time().replace(second=0, microsecond=0),
                key="google_calendar_start",
            )
        with cal_col_end:
            calendar_end = st.time_input(
                "End time",
                value=(datetime.now() + timedelta(hours=1)).time().replace(second=0, microsecond=0),
                key="google_calendar_end",
            )
        calendar_location = st.text_input("Location (optional)", key="google_calendar_location")
        calendar_details = st.text_area("Description (optional)", key="google_calendar_details", height=90)

        if calendar_title.strip():
            event_start = datetime.combine(calendar_date, calendar_start)
            event_end = datetime.combine(calendar_date, calendar_end)
            if event_end <= event_start:
                event_end += timedelta(days=1)
            st.link_button(
                "Review and save in Google Calendar",
                "https://calendar.google.com/calendar/render?action=TEMPLATE"
                + "&text=" + quote_plus(calendar_title.strip())
                + "&dates=" + event_start.strftime("%Y%m%dT%H%M%S")
                + "/" + event_end.strftime("%Y%m%dT%H%M%S")
                + "&details=" + quote_plus(calendar_details.strip())
                + "&location=" + quote_plus(calendar_location.strip()),
            )
        else:
            st.info("Enter an event title to get the Google Calendar button.")

# -------------------------------------------------
# 9. CHAT HISTORY
# -------------------------------------------------
for item in st.session_state.chat:
    with st.chat_message(item["role"], avatar="🧑‍💻" if item["role"] == "user" else "✨"):
        if item["role"] == "user":
            st.markdown(item["content"])
        else:
            render_assistant(item)

# -------------------------------------------------
# 10. INPUT + RUN
# -------------------------------------------------
question = st.chat_input("Ask a question, calculate, or search the web...")
isolated_request = False

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    isolated_request = st.session_state.pop("pending_isolated", False)

if question:
    question = question.strip()
    if len(question) > MAX_INPUT_CHARS:
        st.warning(f"Your message was shortened to {MAX_INPUT_CHARS} characters.")
        question = question[:MAX_INPUT_CHARS]

if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        try:
            result_item = answer_question(
                question, LLM_CHAIN, temperature, max_results, isolated_request=isolated_request
            )
        except Exception as exc:
            logger.exception("Request failed")
            result_item = {
                "role": "assistant",
                "answer": friendly_error(exc),
                "error": True,
                "details": redact_secrets(f"{type(exc).__name__}: {exc}")[:1500],
            }
        render_assistant(result_item)
        st.session_state.chat.append(result_item)
    st.rerun()