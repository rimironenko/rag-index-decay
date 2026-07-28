"""Strip Hugo front-matter/shortcodes and normalize handbook Markdown.

Nav/menu-stub filtering is content-length based, not filename based: real
handbook _index.md files (e.g. content/handbook/_index.md,
content/handbook/engineering/_index.md) are substantial prose, not stubs, so
a naive "drop every _index.md" rule would delete real content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
SHORTCODE_RE = re.compile(r"\{\{[%<]\s*.*?\s*[%>]\}\}", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
BLANK_RUN_RE = re.compile(r"\n{3,}")
TRAILING_WS_RE = re.compile(r"[ \t]+\n")

STUB_TOKEN_THRESHOLD = 30


@dataclass
class CleanedDoc:
    title: str
    body: str
    source_path: str


def strip_frontmatter(text: str) -> tuple[dict, str]:
    """Split leading `---\\n...\\n---\\n` YAML block off; parse it with PyYAML."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml = match.group(1)
    body = text[match.end() :]
    try:
        front_matter = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError:
        front_matter = {}
    if not isinstance(front_matter, dict):
        front_matter = {}
    return front_matter, body


def strip_shortcodes(text: str) -> str:
    """Remove Hugo `{{< ... >}}` / `{{% ... %}}` shortcodes and raw embedded HTML tags."""
    text = SHORTCODE_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = TRAILING_WS_RE.sub("\n", text)
    text = BLANK_RUN_RE.sub("\n\n", text)
    return text.strip() + "\n"


def is_stub(body: str, threshold: int = STUB_TOKEN_THRESHOLD) -> bool:
    """Rough pre-tokenizer stub filter: a whitespace-split word count under
    `threshold` is treated as a nav/menu placeholder rather than real content.
    (The exact token count used for chunking comes later, from the BGE-M3
    tokenizer in corpus/chunk.py - this is just a cheap upfront filter.)
    """
    return len(body.split()) < threshold


def clean_text(text: str, source_path: str) -> CleanedDoc | None:
    """Same cleaning pipeline as clean_file, but for text already in memory -
    e.g. a git blob read via GitPython in churn/replay.py, which has no
    filesystem path to read from. Returns None if the file is a stub (see
    is_stub)."""
    front_matter, body = strip_frontmatter(text)
    body = strip_shortcodes(body)
    body = normalize_whitespace(body)

    if is_stub(body):
        return None

    title = front_matter.get("title") or Path(source_path).stem
    return CleanedDoc(title=str(title), body=body, source_path=source_path)


def clean_file(path: Path) -> CleanedDoc | None:
    """Returns None if the file is a stub (see is_stub)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return clean_text(raw, str(path))
