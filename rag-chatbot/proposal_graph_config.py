# proposal_graph_config.py

import re
import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from prompts import (
    MODE_INSTR,
    PERSONA_MAP,
    PROPOSAL_GUIDANCE_PROMPT,
    PROPOSAL_REFINEMENT_PROMPT,
    PROPOSAL_ROUTER_PROMPT,
    RAG_SAFETY_PREAMBLE,
)
from proposal_tools import (
    TEMPLATE_METADATA,
    format_paper_summaries,
    format_retrieved_context,
    get_template_outline,
)
from rag_tools import llm_complete, retrieve_kb_context


class ProposalState(TypedDict):
    question: str
    mode: str
    persona: str
    summary: str
    recent_qas: str
    task: str
    answer: str
    paper_summaries: Dict[str, str]
    metadatas: List[Dict[str, Any]]
    context_docs: List[str]
    next_step: str
    last_task: str


REFINE_KEYWORDS = [
    "working title",
    "introduction",
    "objective",
    "theoretical basics",
    "research design",
    "expected results",
    "outline",
    "project planning",
    "literature",
    "proposal draft",
    "draft",
]


def _get_style(persona: str) -> Dict[str, Any]:
    return PERSONA_MAP.get(persona, PERSONA_MAP["Helper"])


def should_force_refine(question: str, has_user_context: bool) -> bool:
    """
    Decide if the student clearly provided draft text.

    Only force refinement when:
      * The chat message contains multiple sections AND
      * Either the user already uploaded summaries OR you see template keywords.
    """
    if not question:
        return False
    text = question.strip()
    lower = text.lower()

    newline_count = lower.count("\n")
    has_sections = newline_count >= 6 or "###" in lower or "##" in lower
    long_text = len(text) > 900

    keyword_hit = any(kw in lower for kw in REFINE_KEYWORDS)

    if not (has_sections or long_text):
        return False

    if has_user_context or keyword_hit:
        return True
    return False


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).lower()


def _wrap_with_highlight(line: str) -> str:
    if "<span" in line:
        return line
    highlight_tpl = '<span style="color:#991d38ff;"><strong>{}</strong></span>'

    heading_match = re.match(r"^(\s*#+\s*)(.*)$", line)
    bullet_match = re.match(r"^(\s*[-*]\s+)(.*)$", line)
    number_match = re.match(r"^(\s*\d+[\.\)]\s+)(.*)$", line)

    if heading_match:
        prefix, rest = heading_match.groups()
        return f"{prefix}{highlight_tpl.format(rest.strip())}"
    if bullet_match:
        prefix, rest = bullet_match.groups()
        return f"{prefix}{highlight_tpl.format(rest.strip())}"
    if number_match:
        prefix, rest = number_match.groups()
        return f"{prefix}{highlight_tpl.format(rest.strip())}"

    indent_match = re.match(r"^(\s*)(.*)$", line)
    indent, rest = indent_match.groups() if indent_match else ("", line)
    return f"{indent}{highlight_tpl.format(rest.strip())}"


def highlight_changes(original: str, updated: str) -> str:
    original_lines = {
        _normalize_line(line)
        for line in original.splitlines()
        if line.strip()
    }

    highlighted_lines: List[str] = []
    for line in updated.splitlines():
        stripped = line.strip()
        if not stripped:
            highlighted_lines.append(line)
            continue
        lower = stripped.lower()
        if lower.startswith("sources"):
            highlighted_lines.append(line)
            continue
        normalized = _normalize_line(stripped)
        if normalized in original_lines:
            highlighted_lines.append(line)
        else:
            highlighted_lines.append(_wrap_with_highlight(line))

    return "\n".join(highlighted_lines)


def _truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _shorten_for_retrieval(text: str, max_chars: int = 1200) -> str:
    """
    Keep retrieval queries compact so embeddings are not dominated by full drafts.
    """
    return _truncate_text(text, max_chars)


def router_node(state: ProposalState) -> ProposalState:
    question = state.get("question", "")
    has_user_context = bool(state.get("paper_summaries")) or state.get("last_task") == "proposal_refine"
    if should_force_refine(question, has_user_context):
        state["next_step"] = "refine"
        state["task"] = "proposal_router"
        state["answer"] = ""
        state["metadatas"] = []
        state["context_docs"] = []
        return state

    prompt = PROPOSAL_ROUTER_PROMPT.format(
        preamble=RAG_SAFETY_PREAMBLE,
        mode=state.get("mode", ""),
        persona=state.get("persona", ""),
        summary=state.get("summary", "None"),
        recent_qas=state.get("recent_qas", "None"),
        question=question,
    )
    decision = llm_complete(prompt, max_tokens=20, temperature=0.0).lower()
    next_step = "refine" if "refine" in decision else "guidance"

    if next_step == "refine" and not should_force_refine(question, has_user_context):
        next_step = "guidance"

    state["next_step"] = next_step
    state["task"] = "proposal_router"
    state["answer"] = ""
    state["metadatas"] = []
    state["context_docs"] = []
    return state


def _route_after_router(state: ProposalState) -> str:
    return "proposal_refine" if state.get("next_step") == "refine" else "proposal_guidance"


def proposal_guidance_node(state: ProposalState) -> ProposalState:
    docs, metas = retrieve_kb_context(state["question"], n_results=5)
    context_block = _truncate_text(format_retrieved_context(docs, metas), 2500)
    style = _get_style(state.get("persona", "Helper"))
    mode_instr = MODE_INSTR.get(
        "Proposal refinement assistant",
        "Guide refinement of the BFH proposal template.",
    )

    prompt = PROPOSAL_GUIDANCE_PROMPT.format(
        preamble=RAG_SAFETY_PREAMBLE,
        mode=state.get("mode", ""),
        mode_instr=mode_instr,
        persona=state.get("persona", ""),
        persona_instr=style["instr"],
        summary=state.get("summary", "None"),
        recent_qas=state.get("recent_qas", "None"),
        context=context_block,
        question=state.get("question", ""),
    )

    answer = llm_complete(prompt, max_tokens=1800, temperature=style["temp"])
    state["answer"] = answer
    state["task"] = "proposal_guidance"
    state["metadatas"] = metas
    state["context_docs"] = docs
    return state


def proposal_refine_node(state: ProposalState) -> ProposalState:
    query = _shorten_for_retrieval(state.get("question", ""))
    docs, metas = retrieve_kb_context(query, n_results=6, min_bfh=2)
    rag_context = _truncate_text(format_retrieved_context(docs, metas), 2500)
    template_outline = _truncate_text(get_template_outline(), 2000)
    paper_summaries = _truncate_text(
        format_paper_summaries(state.get("paper_summaries", {})), 2500
    )
    style = _get_style(state.get("persona", "Helper"))
    mode_instr = MODE_INSTR.get(
        "Proposal refinement assistant",
        "Guide refinement of the BFH proposal template.",
    )

    prompt = PROPOSAL_REFINEMENT_PROMPT.format(
        preamble=RAG_SAFETY_PREAMBLE,
        mode=state.get("mode", ""),
        mode_instr=mode_instr,
        persona=state.get("persona", ""),
        persona_instr=style["instr"],
        summary=state.get("summary", "None"),
        recent_qas=state.get("recent_qas", "None"),
        template_outline=template_outline,
        paper_summaries=paper_summaries,
        rag_context=rag_context,
        question=state.get("question", ""),
    )

    answer = llm_complete(prompt, max_tokens=1800, temperature=style["temp"])
    highlighted = highlight_changes(state.get("question", ""), answer)
    # Surface which context was used so the student can see how the draft was refined.
    used_titles = []
    seen = set()
    for meta in metas + TEMPLATE_METADATA:
        title = (meta.get("quelle") or meta.get("title") or "").strip()
        if title and title not in seen:
            seen.add(title)
            used_titles.append(title)
    paper_titles = [t.strip() for t in state.get("paper_summaries", {}).keys()]
    sources_note = ""
    if used_titles or paper_titles:
        parts = []
        if used_titles:
            parts.append("Guides: " + ", ".join(used_titles))
        if paper_titles:
            parts.append("Papers: " + ", ".join(paper_titles))
        sources_note = "\n\n_Source basis: {}_".format("; ".join(parts))

    state["answer"] = highlighted + sources_note
    state["task"] = "proposal_refine"
    state["metadatas"] = metas + TEMPLATE_METADATA
    state["context_docs"] = docs
    return state


graph_builder = StateGraph(ProposalState)
graph_builder.add_node("router", router_node)
graph_builder.add_node("proposal_guidance", proposal_guidance_node)
graph_builder.add_node("proposal_refine", proposal_refine_node)
graph_builder.set_entry_point("router")
graph_builder.add_conditional_edges(
    "router",
    _route_after_router,
    {
        "proposal_guidance": "proposal_guidance",
        "proposal_refine": "proposal_refine",
    },
)
graph_builder.add_edge("proposal_guidance", END)
graph_builder.add_edge("proposal_refine", END)

proposal_graph = graph_builder.compile()
