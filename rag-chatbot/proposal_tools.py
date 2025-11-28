# proposal_tools.py

"""
Utility helpers for the proposal agent.

They take care of:
- Loading and summarising the official BFH proposal template.
- Formatting retrieved Creswell/BFH context for prompts.
- Formatting uploaded paper summaries so that the proposal agent can reuse them.

All prompts rely on the `llm_complete` helper from `rag_tools`, which in turn
uses the shared BFH LLM service.
"""

import html
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_tools import llm_complete  # re-exported for backwards compatibility

TEMPLATE_PATH = Path("data/BScDigi_Proposal_Template_EN_v2.docx")
TEMPLATE_METADATA = [
    {
        "title": "BFH Proposal Template (EN)",
        "quelle": TEMPLATE_PATH.name,
        "chunk_id": "template_outline",
    }
]

SECTION_TITLES = [
    "Introduction & relevance of the topic",
    "Objective",
    "Theoretical basics",
    "Research design/method",
    "Expected results of the work",
    "Outline",
    "Project planning",
    "Literature",
]

FALLBACK_TEMPLATE_OUTLINE = """### Introduction & relevance of the topic
- Explain the central problem or phenomenon and why it matters for BFH/Bern Business School stakeholders.
- State the scope and explicitly list what is out of scope.

### Objective
- Provide the overall aim and concrete research question(s) that are measurable.
- Mention the beneficiaries (industry, academia) and expected value.

### Theoretical basics
- List the key theories/concepts and core literature you will rely on.
- Explain why these theories fit the objective.

### Research design/method
- Describe the methodological approach, instruments, sampling, partners, and risks.

### Expected results
- Describe tangible outputs, deliverables, and how they answer the research question.

### Outline
- Present the proposed thesis structure (chapters + short description).

### Project planning
- Outline the work packages, timeline, and slack for writing/review.

### Literature
- Provide the preliminary bibliography that underpins the sections above.
"""

EXAMPLES_DIRS = [
    Path(__file__).parent.parent / "examples",
    Path(__file__).parent / "examples",
    Path(__file__).parent.parent,
    Path(__file__).parent,
]


def _humanize_stem(stem: str) -> str:
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", stem).strip().title()


@lru_cache(maxsize=1)
def discover_specialization_examples() -> Dict[str, Path]:
    """
    Discover available specialization example files.

    Rules:
    - Look for Markdown files in an `examples/` folder, or fallback to repo roots.
    - Skip generic files like README.md.
    - Use the stem as the display label (title-cased, underscores/dashes -> spaces).
    """
    found: Dict[str, Path] = {}
    for base in EXAMPLES_DIRS:
        if base.is_dir():
            for path in base.glob("*.md"):
                if "readme" in path.stem.lower():
                    continue
                label = _humanize_stem(path.stem)
                found.setdefault(label, path)
        elif base.is_file() and base.suffix == ".md":
            if "readme" in base.stem.lower():
                continue
            label = _humanize_stem(base.stem)
            found.setdefault(label, base)

    return found


@lru_cache(maxsize=1)
def load_proposal_examples(max_chars: int = 4000) -> str:
    """
    Legacy helper: returns an empty string when no specialization example is found.
    """
    return ""


def load_specialization_example(label: str) -> str:
    """
    Load the proposal example that matches the given specialization label.
    Returns the full file (no truncation).
    """
    mapping = discover_specialization_examples()
    path = mapping.get(label)
    if not path or not path.exists():
        return load_proposal_examples()
    raw = path.read_text(encoding="utf-8")
    clean = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return clean


@lru_cache(maxsize=1)
def _load_template_text() -> str:
    if not TEMPLATE_PATH.exists():
        return ""
    try:
        with zipfile.ZipFile(TEMPLATE_PATH) as doc:
            xml = doc.read("word/document.xml").decode("utf-8")
    except Exception:
        return ""

    text = re.sub(r"<.*?>", " ", xml)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_section(raw: str, title: str, next_title: Optional[str]) -> str:
    lower = raw.lower()
    start = lower.find(title.lower())
    if start == -1:
        return ""
    body_start = start + len(title)
    end = lower.find(next_title.lower(), body_start) if next_title else len(raw)
    snippet = raw[body_start:end].strip()
    placeholder_idx = snippet.lower().find("x xxxx")
    if placeholder_idx != -1:
        snippet = snippet[:placeholder_idx]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet


@lru_cache(maxsize=1)
def get_template_outline() -> str:
    """
    Returns a concise markdown outline derived from the official BFH template.
    Falls back to hand-written guidance if the DOCX cannot be read.
    """
    raw = _load_template_text()
    if not raw:
        return FALLBACK_TEMPLATE_OUTLINE.strip()

    outline_parts: List[str] = []
    for idx, title in enumerate(SECTION_TITLES):
        next_title = SECTION_TITLES[idx + 1] if idx + 1 < len(SECTION_TITLES) else None
        body = _extract_section(raw, title, next_title)
        if not body:
            continue
        outline_parts.append(f"### {title}\n{body}")

    if not outline_parts:
        return FALLBACK_TEMPLATE_OUTLINE.strip()
    return "\n\n".join(outline_parts)


def format_retrieved_context(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    if not docs:
        return "None"
    lines = []
    for idx, doc in enumerate(docs):
        meta = metas[idx] if idx < len(metas) else {}
        title = meta.get("quelle") or meta.get("title") or f"Doc {idx + 1}"
        lines.append(f"[{title}]\n{doc.strip()}")
    return "\n\n".join(lines)


def format_paper_summaries(paper_summaries: Dict[str, str]) -> str:
    if not paper_summaries:
        return "None provided."
    parts = []
    for title, summary in paper_summaries.items():
        clean_summary = summary.strip()
        parts.append(f"### {title}\n{clean_summary}")
    return "\n\n".join(parts)


def proposal_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    """
    Thin wrapper that keeps backwards compatibility with earlier experiments.
    """
    prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    text = llm_complete(prompt, max_tokens=max_tokens, temperature=temperature)
    return {
        "text": text,
        "model": "proposal/llm_complete",
        "usage": {},
    }
