"""Shared test setup.

Ensure dummy API keys exist before any app module is imported, so client
construction (Groq, Tavily) succeeds without real credentials. Tests mock out
all network calls, so the keys are never actually used.
"""

import os
import sys
from pathlib import Path

# Make the repo root importable (so `import app.*` works when running pytest).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
