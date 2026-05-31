"""Shared LLM client.

Every chat/LLM call in the system should go through this client so model
configuration and credentials live in one place. Embeddings are handled
separately and locally in app/vectorstore/embeddings.py (FastEmbed).
"""

import json
import re

from groq import Groq
from langsmith import traceable

from app.config import settings

# Default chat model served by Groq.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Groq client for chat completions (evaluation, rewriting, generation, etc.).
groq_client = Groq(api_key=settings.groq_api_key)


@traceable(run_type="llm", name="groq_chat")
def chat(
    system: str,
    user: str,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Run a single system+user chat completion and return the stripped text.

    All agents should call this instead of touching groq_client directly, so
    model choice and request shape live in one place.

    When json_mode is True, Groq is asked to return a valid JSON object
    (response_format json_object). The caller's prompt must mention "JSON",
    which is a Groq requirement for this mode.
    """
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content.strip()


# A ```json ... ``` (or bare ``` ... ```) fence the model sometimes wraps output in.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(raw: str) -> dict:
    """Parse a JSON object out of an LLM response, tolerating common noise.

    Handles markdown code fences and surrounding prose by stripping fences and,
    if needed, extracting the first balanced-looking {...} block before parsing.
    Raises json.JSONDecodeError (or ValueError) if no JSON object can be found.
    """
    text = _FENCE_RE.sub("", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first {...} span in the text.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(text[start : end + 1])
