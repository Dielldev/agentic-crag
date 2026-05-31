"""Fetch a set of security-focused Wikipedia articles into data/raw/.

Each article is saved as a plain .txt file named by its title, containing
the title, summary, and full page content.
"""

import time
from pathlib import Path

import requests

ARTICLES = [
    "SQL injection",
    "Cross-site scripting",
    "Buffer overflow",
    "Penetration testing",
    "Burp Suite",
    "OWASP",
    "Metasploit",
    "Network security",
    "Cryptography",
    "Public key infrastructure",
    "Zero-day vulnerability",
    "Social engineering",
    "Ransomware",
    "Firewall",
    "VPN",
]

# scripts/ -> project root -> data/raw
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "agentic-crag/1.0 (educational project)"}


def slugify(title: str) -> str:
    """Make a filesystem-safe filename from an article title."""
    return title.replace("/", "-").replace(" ", "_")


def fetch_article(title: str) -> str | None:
    """Return formatted text for an article, or None if it can't be fetched."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
        "format": "json",
    }
    for attempt in range(5):
        resp = requests.get(_API, params=params, headers=_HEADERS, timeout=15)
        if resp.status_code == 429:
            wait = 2 ** attempt * 3
            print(f"  rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        print(f"  skipped (rate limited): {title}")
        return None
    data = resp.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        print(f"  skipped (not found): {title}")
        return None

    content = page.get("extract", "")
    # Split summary (first paragraph) from the rest
    parts = content.split("\n\n", 1)
    summary = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    return (
        f"Title: {page['title']}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Content:\n{body}\n"
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for title in ARTICLES:
        out_path = RAW_DIR / f"{slugify(title)}.txt"
        if out_path.exists():
            print(f"  skipping (already exists): {out_path.name}")
            saved += 1
            continue
        text = fetch_article(title)
        if text is None:
            continue

        out_path.write_text(text, encoding="utf-8")
        print(f"  saved: {out_path.name}")
        saved += 1
        time.sleep(1)

    print(f"\nSaved {saved} of {len(ARTICLES)} articles to {RAW_DIR}")


if __name__ == "__main__":
    main()
