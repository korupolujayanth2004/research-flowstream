import os
import uuid
import asyncio
import json
import requests
from typing import AsyncGenerator
from dotenv import load_dotenv

# Load .env so env vars are available when starting Uvicorn directly
load_dotenv()

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Optional switch to simulate local behavior (no external calls)
GROQ_DISABLED = os.getenv("GROQ_DISABLED", "").lower() in {"1", "true", "yes"}

# Reasonable connect/read timeouts for generation/streaming
DEFAULT_TIMEOUT = (10, 120)

# Base headers for Groq API
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}" if GROQ_API_KEY else "",
    "Content-Type": "application/json",
}


def generate_report_id() -> str:
    """Create a unique ID for each report."""
    return str(uuid.uuid4())


def stream_event(kind: str, data):
    """
    Serialize events as proper JSON for SSE.
    The FastAPI route will send lines like: `data: <json>\n\n`
    Frontend can safely parse with json.loads(payload).
    """
    return json.dumps({"kind": kind, "data": data}, ensure_ascii=False)


def _chunk(text: str, n: int):
    """Split text into small pieces to render a smoother streaming experience."""
    for i in range(0, len(text), n):
        yield text[i : i + n]


async def run_researcher_async(topic: str) -> str:
    """
    Researcher step: produce compact factual bullets.
    Fallback text is returned if GROQ is disabled or unavailable.
    """
    if GROQ_DISABLED or not GROQ_API_KEY:
        return (
            f"- What is '{topic}'?\n"
            f"- 3–5 key facts\n"
            f"- Common use cases\n"
            f"- Simple examples\n"
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise researcher."},
            {
                "role": "user",
                "content": f"Provide compact, factual bullet points about '{topic}'. "
                           f"Max 8 bullets. Avoid filler text.",
            },
        ],
        "temperature": 0.5,
    }
    try:
        r = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        # Fallback on any network/API error
        return f"[fallback researcher due to error: {e}]\n- Background\n- Key points\n- Examples"


async def run_analyst_async(researcher_notes: str) -> str:
    """
    Analyst step: extract key insights and implications from researcher notes.
    Fallback text is returned if GROQ is disabled or unavailable.
    """
    if GROQ_DISABLED or not GROQ_API_KEY:
        return "- 3 key insights\n- 2 implications\n- 1 trade-off\n"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You extract insights cleanly."},
            {
                "role": "user",
                "content": f"From these notes, produce exactly 3 insights and 2 implications:\n{researcher_notes}",
            },
        ],
        "temperature": 0.5,
    }
    try:
        r = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[fallback analyst due to error: {e}]\n- Insight 1\n- Insight 2\n- Insight 3\n- Implication A\n- Implication B"


async def run_writer_token_stream(
    topic: str,
    researcher_notes: str,
    analyst_notes: str,
) -> AsyncGenerator[str, None]:
    """
    Writer step: stream the final report as small token-like chunks for smooth UI updates.
    Yields strings (small chunks). Caller accumulates or forwards as SSE tokens.
    """
    writer_prompt = (
        "Write a clear, beginner-friendly report with markdown headings:\n"
        "Sections: Introduction, Key Concepts, Insights, Practical Tips, Conclusion.\n"
        "Use concise language and bullets where helpful.\n\n"
        f"Topic: {topic}\n\n"
        f"Researcher Notes:\n{researcher_notes}\n\n"
        f"Analyst Notes:\n{analyst_notes}\n"
    )

    # Local simulated streaming if GROQ is disabled or key missing
    if GROQ_DISABLED or not GROQ_API_KEY:
        simulated = [
            f"## {topic}\n\n",
            "### Introduction\n",
            "This response is streaming locally to simulate real-time typing.\n\n",
            "### Key Concepts\n",
            "- Concept A\n- Concept B\n\n",
            "### Insights\n",
            "- Insight 1\n- Insight 2\n\n",
            "### Practical Tips\n",
            "- Tip 1\n- Tip 2\n\n",
            "### Conclusion\n",
            "Short summary.\n",
        ]
        for piece in simulated:
            for small in _chunk(piece, 20):
                yield small
                await asyncio.sleep(0.015)
        return

    # Real streaming via Groq's OpenAI-compatible API
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a clear, helpful technical writer."},
            {"role": "user", "content": writer_prompt},
        ],
        "temperature": 0.6,
        "stream": True,
    }

    # Using requests stream; iterate server-sent "data: ..." lines
    with requests.post(
        GROQ_URL, headers=HEADERS, json=payload, stream=True, timeout=DEFAULT_TIMEOUT
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                # Yield tiny chunks to update UI frequently
                for small in _chunk(delta, 20):
                    yield small
            except Exception:
                # Skip malformed lines gracefully
                continue
