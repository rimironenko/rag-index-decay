"""Recursive 512-token / 15% overlap chunker using the embedding model's tokenizer.

Token counts always go through the BGE-M3 tokenizer's `encode()` - never
`len(text)` or a word count - to keep token counts consistent with the
embedding model's own tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-m3"
# Pinned HF revision (fetched 2026-07-12 via the HF API; latest commit on
# BAAI/bge-m3 as of that date). Kept in sync with pyproject.toml's pin.
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

MAX_TOKENS = 512
OVERLAP_TOKENS = 77  # ~15% of 512

PARAGRAPH_RE = re.compile(r"\n\s*\n")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    return _tokenizer


@dataclass
class ChunkSpan:
    text: str
    ordinal: int
    token_count: int


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_paragraph_into_sentences(paragraph: str) -> list[str]:
    return [s for s in SENTENCE_RE.split(paragraph) if s.strip()]


def _hard_split_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Last-resort fallback for a unit with no sentence-ending punctuation to
    split on (tables, long unbroken runs of text/URLs) that is still over
    max_tokens after paragraph- and sentence-level splitting. Slices by the
    tokenizer's own character offsets so the fallback still respects the
    "size chunks by the embedding model's tokenizer" rule, rather than an
    arbitrary character count."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    if len(offsets) <= max_tokens:
        return [text]
    pieces = []
    for start in range(0, len(offsets), max_tokens):
        window = offsets[start : start + max_tokens]
        char_start, char_end = window[0][0], window[-1][1]
        pieces.append(text[char_start:char_end])
    return pieces


def chunk_text(
    text: str,
    tokenizer=None,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[ChunkSpan]:
    """Paragraph-first recursive split, falling back to sentence boundaries for
    any paragraph over `max_tokens`, then to a hard token-offset split for any
    sentence still over `max_tokens` (tables/unbroken text with no sentence
    punctuation) - every unit is guaranteed <= max_tokens before packing.
    Greedily packs units up to `max_tokens` per chunk, then slides back
    `overlap_tokens` worth of trailing units to seed the next window.
    """
    tokenizer = tokenizer or get_tokenizer()

    paragraphs = [p for p in PARAGRAPH_RE.split(text) if p.strip()]

    # Flatten into atomic units, each guaranteed <= max_tokens, so the packing
    # loop below never has to accept an oversized unit into an empty window.
    units: list[str] = []
    for para in paragraphs:
        if _token_len(para, tokenizer) <= max_tokens:
            units.append(para)
            continue
        for sentence in _split_paragraph_into_sentences(para):
            if _token_len(sentence, tokenizer) <= max_tokens:
                units.append(sentence)
            else:
                units.extend(_hard_split_by_tokens(sentence, tokenizer, max_tokens))

    chunks: list[ChunkSpan] = []
    ordinal = 0
    i = 0
    while i < len(units):
        window: list[str] = []
        window_tokens = 0
        j = i
        while j < len(units):
            unit_tokens = _token_len(units[j], tokenizer)
            if window and window_tokens + unit_tokens > max_tokens:
                break
            window.append(units[j])
            window_tokens += unit_tokens
            j += 1

        chunk_str = "\n\n".join(window)
        # Recompute from the joined string rather than trusting the summed
        # per-unit counts: tokenization isn't perfectly additive across a
        # join boundary, so the sum can be off by a token or two.
        actual_token_count = _token_len(chunk_str, tokenizer)
        chunks.append(
            ChunkSpan(text=chunk_str, ordinal=ordinal, token_count=actual_token_count)
        )
        ordinal += 1

        if j >= len(units):
            break

        # Slide back from j by ~overlap_tokens worth of units for the next window.
        back = j - 1
        overlap_count = 0
        while back > i and overlap_count < overlap_tokens:
            overlap_count += _token_len(units[back], tokenizer)
            back -= 1
        next_i = max(back + 1, i + 1)  # guarantee forward progress
        i = next_i

    return chunks
