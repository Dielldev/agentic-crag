#!/usr/bin/env python3
"""Interactive CLI for the Agentic CRAG pipeline.

A cmd-based shell that drives the LangGraph CRAG workflow. While a query runs,
only a spinner is shown — all agent/HTTP log output is silenced from the console
(it still goes to logs/crag.log). The final answer is rendered in a formatted box.

A simple client-side rate limiter throttles queries to stay within the Groq
free-tier request budget (one query fans out into several LLM calls).

Commands: query, history, clear, exit.

Run with:  python scripts/cli.py
"""

import cmd
import os
import sys
import time
import logging
import textwrap
import threading
from collections import deque
from statistics import mean

# `python scripts/cli.py` puts scripts/ on sys.path, not the project root.
# Insert the repo root so `import app.*` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.workflow import app as graph


# ───────────────────────────── Gemini palette ─────────────────────────────
PRIMARY = "\033[38;2;138;180;248m"   # bright blue
SECONDARY = "\033[38;2;66;133;244m"  # Google blue
PURPLE = "\033[38;2;187;134;252m"    # purple
PINK = "\033[38;2;236;130;162m"      # coral-pink (banner gradient end)
MAUVE = "\033[38;2;176;142;170m"     # muted mauve (banner subtitle)
GREEN = "\033[38;2;129;201;149m"     # green
YELLOW = "\033[38;2;253;214;99m"     # yellow
RED = "\033[38;2;242;139;130m"       # red / coral
WHITE = "\033[38;2;232;234;237m"     # off-white
DIM = "\033[38;2;154;160;166m"       # muted gray
BOLD = "\033[1m"
RESET = "\033[0m"

# Horizontal gradient stops (blue → purple → pink) for the banner ASCII art,
# matching the reference image: cool blue on the left, warm pink on the right.
_GRADIENT_STOPS = [
    (0.00, (84, 130, 245)),
    (0.50, (170, 100, 228)),
    (1.00, (240, 112, 150)),
]


def _gradient_rgb(t: float) -> tuple[int, int, int]:
    """Interpolate the gradient color at position t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_GRADIENT_STOPS, _GRADIENT_STOPS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(round(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))
    return _GRADIENT_STOPS[-1][1]


def gradient_lines(lines: list[str]) -> list[str]:
    """Color each non-space glyph by its column, left→right across the gradient."""
    width = max(len(line) for line in lines)
    out = []
    for line in lines:
        buf = []
        for x, ch in enumerate(line):
            if ch == " ":
                buf.append(" ")
            else:
                r, g, b = _gradient_rgb(x / (width - 1) if width > 1 else 0.0)
                buf.append(f"\033[38;2;{r};{g};{b}m{ch}")
        out.append("".join(buf) + RESET)
    return out

# Free-tier rate limiting: at most RATE_LIMIT_MAX full queries per
# RATE_LIMIT_WINDOW seconds. One query = several Groq calls (evaluate per chunk
# + generate + verify, plus rewrite on retries), so keep this conservative.
RATE_LIMIT_MAX = 4
RATE_LIMIT_WINDOW = 60.0


# ───────────────────────────── color helpers ─────────────────────────────
def score_color(value: float) -> str:
    """Color a 0-100 score by quality threshold."""
    if value >= 70:
        return GREEN
    if value >= 40:
        return YELLOW
    return RED


# ───────────────────────────── log silencing ─────────────────────────────
def quiet_logging() -> None:
    """Hide all console log output so only the spinner shows during a query.

    The app's agents attach their own console + file handlers (app/logger.py).
    We drop every console (stdout/stderr) handler but keep FileHandlers, so the
    full trace is still written to logs/crag.log. Chatty HTTP client libraries
    are also raised to WARNING so request lines never leak to the terminal.
    """
    loggers = [logging.getLogger()]
    loggers += [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for lg in loggers:
        for h in lg.handlers[:]:
            # FileHandler subclasses StreamHandler — keep files, drop consoles.
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                lg.removeHandler(h)
    for name in ("httpx", "httpcore", "groq", "urllib3", "openai", "langsmith"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ───────────────────────────── rate limiter ──────────────────────────────
class RateLimiter:
    """Sliding-window limiter: at most `max_calls` recorded calls per `window`."""

    def __init__(self, max_calls: int, window: float) -> None:
        self.max_calls = max_calls
        self.window = float(window)
        self._calls: "deque[float]" = deque()

    def seconds_until_allowed(self) -> float:
        """Return 0.0 if a call is allowed now, else seconds until one is."""
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self.window:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            return self.window - (now - self._calls[0])
        return 0.0

    def record(self) -> None:
        self._calls.append(time.monotonic())


# ───────────────────────────── spinner ───────────────────────────────────
def run_with_spinner(fn, *args, label: str = "generating answer…"):
    """Run `fn(*args)` in a worker thread while a spinner ticks in place.

    No log output is shown — just the spinner — until the call returns. The
    function's result is returned, or its exception re-raised.
    """
    box: dict = {}

    def worker():
        try:
            box["value"] = fn(*args)
        except Exception as e:  # surfaced to caller after the thread joins
            box["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    frames = "|/-\\"
    i = 0
    while t.is_alive():
        sys.stdout.write(f"\r{PRIMARY}{frames[i % 4]}{RESET} {DIM}{label}{RESET}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)

    sys.stdout.write("\r\033[K")  # erase the spinner line
    sys.stdout.flush()

    if "error" in box:
        raise box["error"]
    return box.get("value")


# ───────────────────────────── answer ────────────────────────────────
def render_answer(answer, sources, route, score, verified, elapsed, notes="") -> str:
    """Render the answer cleanly (borderless), in the same style as the intro."""
    out = [f"  {PINK}❯{RESET} {BOLD}{WHITE}Answer{RESET}", ""]
    for line in textwrap.wrap(answer.strip(), width=76) or ["(no answer)"]:
        out.append(f"  {WHITE}{line}{RESET}")
    out.append("")

    # Prominent caution when the verifier judged the answer ungrounded.
    if verified is False:
        out.append(f"  {RED}⚠ This answer may not be fully grounded in the sources.{RESET}")
        for line in textwrap.wrap(notes.strip(), width=72):
            out.append(f"  {DIM}{line}{RESET}")
        out.append("")

    if score is None:
        score_str, score_clr = "N/A", DIM
    else:
        score_str, score_clr = f"{score}/100", score_color(score)

    if verified is None:
        v_str, v_clr = "—", DIM
    elif verified:
        v_str, v_clr = "✓ grounded", GREEN
    else:
        v_str, v_clr = "✗ ungrounded", RED

    meta = [
        ("Sources", sources or "—", PURPLE),
        ("Route", route, SECONDARY),
        ("Score", score_str, score_clr),
        ("Verified", v_str, v_clr),
        ("Time", f"{elapsed:.1f}s", DIM),
    ]
    for label, value, color in meta:
        out.append(f"  {MAUVE}{label:<8}{RESET}  {color}{value}{RESET}")
    return "\n".join(out)


# ───────────────────────────── banner ─────────────────────────────────────
_CHEVRON = [
    "██╗   ",
    "╚██╗  ",
    " ╚██╗ ",
    " ██╔╝ ",
    "██╔╝  ",
    "╚═╝   ",
]
_C = [
    " ██████╗",
    "██╔════╝",
    "██║     ",
    "██║     ",
    "╚██████╗",
    " ╚═════╝",
]
_R = [
    "██████╗ ",
    "██╔══██╗",
    "██████╔╝",
    "██╔══██╗",
    "██║  ██║",
    "╚═╝  ╚═╝",
]
_A = [
    " █████╗ ",
    "██╔══██╗",
    "███████║",
    "██╔══██║",
    "██║  ██║",
    "╚═╝  ╚═╝",
]
_G = [
    " ██████╗ ",
    "██╔════╝ ",
    "██║  ███╗",
    "██║   ██║",
    "╚██████╔╝",
    " ╚═════╝ ",
]
# Chevron prompt + "CRAG" wordmark, row by row (ANSI Shadow style).
_ART = [f"{_CHEVRON[i]} {_C[i]}{_R[i]}{_A[i]}{_G[i]}" for i in range(6)]

_COMMANDS = [
    ("query <question>", "run the corrective-RAG pipeline"),
    ("history", "show recent queries this session"),
    ("clear", "clear the screen"),
    ("exit", "quit"),
]


def build_banner() -> str:
    """Render the gradient CRAG wordmark with title, subtitle, and commands."""
    out = gradient_lines(_ART)
    out.append("")
    out.append(f"{BOLD}{WHITE}Dielldev CRAG{RESET}")
    out.append(f"{MAUVE}Corrective Retrieval-Augmented Generation{RESET}")
    out.append("")
    out.append(f"{MAUVE}Commands:{RESET}")
    for name, desc in _COMMANDS:
        out.append(f"  {PINK}{name:<18}{RESET}{DIM}{desc}{RESET}")
    return "\n".join(out)


# ───────────────────────────── the shell ─────────────────────────────
def _np(seq: str) -> str:
    """Wrap a non-printing escape so readline counts prompt width correctly."""
    return "\001" + seq + "\002"


class CragCLI(cmd.Cmd):
    """Interactive shell for the CRAG pipeline."""

    prompt = (
        _np(PINK) + "dielldev" + _np(RESET) + _np(DIM) + "@" + _np(RESET)
        + _np(SECONDARY) + "crag" + _np(RESET) + " ❯ "
    )

    def __init__(self) -> None:
        super().__init__()
        self.intro = build_banner()
        self.history: list[dict] = []
        self.limiter = RateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)

    # ---- output helpers ----
    def error(self, msg: str) -> None:
        print(f"{RED}✗ {msg}{RESET}")

    def warn(self, msg: str) -> None:
        print(f"{YELLOW}{msg}{RESET}")

    # ---- commands ----
    def do_query(self, arg: str) -> None:
        """query <question>  — run the full CRAG pipeline and show the answer."""
        q = arg.strip()
        if not q:
            self.warn("usage: query <question>")
            return

        wait = self.limiter.seconds_until_allowed()
        if wait > 0:
            self.warn(
                f"⏳ rate limit reached ({RATE_LIMIT_MAX} queries / "
                f"{int(RATE_LIMIT_WINDOW)}s on free tier) — wait {wait:.0f}s"
            )
            return
        self.limiter.record()

        try:
            start = time.time()
            result = run_with_spinner(graph.invoke, {"question": q, "iteration": 0})
            elapsed = time.time() - start
        except Exception as e:
            self.error(f"pipeline failed: {e}")
            return

        chunks = result.get("chunks", []) or []
        sources = ", ".join(dict.fromkeys(c.source for c in chunks))

        # eval_result is a list of per-chunk ChunkEval; average them like the API.
        evals = result.get("eval_result") or []
        score = (
            round(mean((e.relevance + e.completeness + e.confidence) / 3 for e in evals))
            if evals
            else None
        )

        if result.get("search_type") == "web":
            route = "web-search"
        elif result.get("iteration", 0) > 0:
            route = "vector+rewrite"
        else:
            route = "vector-only"

        verification = result.get("verification")
        verified = verification.grounded if verification else None
        notes = verification.notes if verification else ""

        print()
        print(
            render_answer(
                result.get("answer", ""), sources, route, score, verified, elapsed, notes
            )
        )
        self.history.append(
            {"question": q, "route": route, "score": score, "verified": verified}
        )

    def do_history(self, arg: str) -> None:
        """history  — show the last 10 queries with routes, scores, grounding."""
        if not self.history:
            self.warn("no queries yet this session.")
            return

        rows = self.history[-10:]
        header = f"{'#':<3}{'QUESTION':<34}{'ROUTE':<16}{'SCORE':<8}{'VRF'}"
        print(f"\n{PRIMARY}{BOLD}{header}{RESET}")
        print(f"{DIM}{'─' * 64}{RESET}")
        for i, h in enumerate(rows, 1):
            base = WHITE if i % 2 else DIM
            q = h["question"]
            q = q[:31] + "…" if len(q) > 32 else q

            if h["score"] is None:
                sc_str, sc_clr = "—", DIM
            else:
                sc_str, sc_clr = str(h["score"]), score_color(h["score"])

            if h["verified"] is None:
                vrf_str, vrf_clr = "—", DIM
            elif h["verified"]:
                vrf_str, vrf_clr = "✓", GREEN
            else:
                vrf_str, vrf_clr = "✗", RED

            print(
                f"{base}{i:<3}{q:<34}{h['route']:<16}{RESET}"
                f"{sc_clr}{sc_str:<8}{RESET}{vrf_clr}{vrf_str}{RESET}"
            )

    def do_clear(self, arg: str) -> None:
        """clear  — clear the terminal."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print(self.intro)

    def do_exit(self, arg: str) -> bool:
        """exit  — leave the shell."""
        print(f"{PINK}Have a nice one {RESET}")
        return True

    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl-D as exit."""
        print()
        return self.do_exit(arg)

    def emptyline(self) -> None:
        """Do nothing on an empty line (default would repeat last command)."""
        pass

    def default(self, line: str) -> None:
        cmd_name = line.split()[0] if line.split() else line
        self.error(f"unknown command: {cmd_name!r}  (try: query  history  clear  exit)")


def main() -> None:
    quiet_logging()
    cli = CragCLI()
    while True:
        try:
            cli.cmdloop()
            break
        except KeyboardInterrupt:
            print()
            cli.warn("^C  (use 'exit' to quit)")
            cli.intro = None  # don't reprint the banner on resume


if __name__ == "__main__":
    main()
