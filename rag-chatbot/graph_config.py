# graph_config.py

import re
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from prompts import RAG_SAFETY_PREAMBLE, MODE_INSTR, PERSONA_MAP
from rag_tools import llm_complete, retrieve_kb_context
from llm_service import llm_service 

# -------------------------------------------------------------------
# App state
# -------------------------------------------------------------------

class AppState(TypedDict):
    # core
    question: str
    mode: str
    persona: str
    summary: str
    recent_qas: str
    task: str                      # "paper_question" | "structure_question" | "gap_analysis"
    upload_collection_name: str    # kept for tracing / compatibility

    # NEW: summaries of uploaded PDFs
    # filename -> markdown summary (created by BFH LLM at upload time)
    paper_summaries: Dict[str, str]

    # which filenames the user explicitly mentioned in the question
    selected_titles: List[str]

    # generic RAG outputs (still used for UI / sources)
    context_docs: List[str]
    metadatas: List[Dict[str, Any]]
    answer: str

    # gap pipeline intermediates
    gap_paper_summaries: str
    gap_guides: str
    gap_candidates: str
    rq_candidates: str

    # methods pipeline intermediates
    methods_task: str      # "critique_design" | "propose_design" | "refine_question"
    methods_guides: str    # concatenated Creswell/BFH guidance


# -------------------------------------------------------------------
# Router
# -------------------------------------------------------------------

def router_node(state: AppState) -> AppState:
    """
    LLM-based routing agent using the BFH GPT-OSS model.

    Decides which specialized pipeline should handle the user's message:
    - paper_question: questions about uploaded papers/articles
    - structure_question: research design / methods / RQ / structure
    - gap_analysis: research gaps & contributions using multiple papers + guidance
    """
    user_msg = state["question"]
    mode = state.get("mode", "")
    persona = state.get("persona", "")
    summary = state.get("summary", "")
    recent_qas = state.get("recent_qas", "")

    # We put all rich context into the *user* message, and keep the system
    # prompt short and strict about the required output format.
    user_prompt = f"""{RAG_SAFETY_PREAMBLE}

You are a ROUTING AGENT in a thesis-proposal assistant.

Your job:
Decide which specialized agent should handle the USER'S LATEST MESSAGE.

Available agents (choose exactly ONE):

1) paper_question
   - The user is asking about the CONTENT of uploaded papers, PDFs, or articles.
   - Examples:
     - "What is the paper I uploaded about?"
     - "Summarize the uploaded pdf."
     - "What methods did this article use?"
     - "What were the main findings of my uploaded paper?"

2) structure_question
   - The user is asking about RESEARCH DESIGN, METHODS, or RESEARCH QUESTIONS
     for their own planned thesis / project.
   - Examples:
     - "Is this a good research question?"
     - "Is my method feasible?"
     - "How should I collect data?"
     - "Can you critique my study design?"

3) gap_analysis
   - The user is asking about RESEARCH GAPS or CONTRIBUTIONS, usually in relation
     to one or more papers or an emerging field.
   - Examples:
     - "What gaps are still open in these papers?"
     - "What could be the research gap for me to fill?"
     - "Which research gaps do these studies leave?"
     - "What contribution could my thesis make, based on the literature?"

Context you can use:

[Current mode]
{mode}

[Persona]
{persona}

[Session summary]
{summary}

[Recent Q&A]
{recent_qas}

[User's latest message]
{user_msg}

Output format:
- Answer with EXACTLY ONE WORD:
  paper_question
  structure_question
  gap_analysis
"""

    resp = llm_service.generate_completion(
        system_prompt=(
            "You are a routing classifier for a thesis assistant. "
            "Given the detailed description in the user message, "
            "respond with exactly ONE of these labels:\n"
            "- paper_question\n- structure_question\n- gap_analysis\n"
            "No explanation, no punctuation, just the single word."
        ),
        user_prompt=user_prompt,
        temperature=0.0,
    )

    label = resp["text"].strip().lower()
    if label not in {"paper_question", "structure_question", "gap_analysis"}:
        # Safe fallback if the model says something unexpected
        label = "structure_question"

    state["task"] = label
    return state


def router_edge(state: AppState) -> str:
    return state.get("task", "structure_question")


# -------------------------------------------------------------------
# Paper pipeline: select_scope → retrieve_passages → synthesize_answer
# (now based on precomputed full-paper summaries, not Chroma chunks)
# -------------------------------------------------------------------

def _extract_pdf_titles(question: str) -> List[str]:
    """
    Extract candidate PDF file names from the user's question.

    Very simple heuristic: anything that looks like 'name.pdf'.
    Stored as lowercase for matching.
    """
    pattern = re.compile(r'([\w\-.]+\.pdf)', re.IGNORECASE)
    return [m.group(1).lower() for m in pattern.finditer(question)]


def paper_select_scope(state: AppState) -> AppState:
    """
    Scope which uploaded files to use:
    - If the user explicitly mentions one or more PDF file names in the question
      (e.g. 'in review.pdf, what...'), we store those (lowercased) in selected_titles.
    - Otherwise, selected_titles = [], meaning: use all summarized papers.
    """
    titles = _extract_pdf_titles(state["question"])
    state["selected_titles"] = titles
    return state


def paper_retrieve_passages(state: AppState) -> AppState:
    """
    Instead of retrieving vector chunks, we look up the precomputed summaries
    in state["paper_summaries"].

    - If selected_titles is non-empty, we try to use only those files.
    - If none of the selected_titles exist, we fall back to all summaries.
    """
    summaries = state.get("paper_summaries") or {}
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    if not summaries:
        state["context_docs"] = []
        state["metadatas"] = []
        return state

    wanted = [t.lower() for t in (state.get("selected_titles") or [])]

    def add_doc(filename: str, text: str):
        docs.append(text)
        metas.append({"title": filename})

    # 1) Try respecting selected_titles
    if wanted:
        for filename, summary in summaries.items():
            if filename.lower() in wanted:
                add_doc(filename, summary)

    # 2) If nothing matched, use all summaries
    if not docs:
        for filename, summary in summaries.items():
            add_doc(filename, summary)

    state["context_docs"] = docs
    state["metadatas"] = metas
    return state


def paper_synthesize_answer(state: AppState) -> AppState:
    """
    Answer questions about uploaded papers using their summaries.

    - context_docs now contain 1+ full-paper summaries (markdown).
    - metadatas contain at least "title" for each summary.

    We:
    - build a combined context
    - send it to Mixtral with strong anti-hallucination rules
    - also store the combined summaries in gap_paper_summaries
      so the gap pipeline can reuse them.
    """
    docs = state.get("context_docs") or []
    metas = state.get("metadatas") or []

    if not docs:
        state["answer"] = (
            "I couldn't find any summaries of your uploaded papers.\n\n"
            "Please upload PDFs in the sidebar, click **“Summarize uploaded papers”**, "
            "and then ask your question again."
        )
        state["gap_paper_summaries"] = ""
        return state

    persona = state.get("persona", "Helper")
    mode = state.get("mode", "Research question helper")
    summary = state.get("summary", "")
    recent_qas_text = state.get("recent_qas", "None")

    style = PERSONA_MAP.get(persona, PERSONA_MAP["Helper"])

    # combined summaries (also reused for gap analysis)
    combined_summaries = ""
    paper_titles = []
    seen = set()

    for doc_text, meta in zip(docs, metas):
        title = (meta.get("title") or "uploaded paper").strip()
        if title not in seen:
            seen.add(title)
            paper_titles.append(title)

        combined_summaries += doc_text.strip() + "\n\n"

    titles_str = ", ".join(paper_titles) if paper_titles else "the uploaded papers"

    # hard cap for safety
    MAX_CONTEXT_CHARS = 15000
    if len(combined_summaries) > MAX_CONTEXT_CHARS:
        combined_summaries = combined_summaries[:MAX_CONTEXT_CHARS]

    state["gap_paper_summaries"] = combined_summaries.strip()

    prompt = f"""{RAG_SAFETY_PREAMBLE}

You are a thesis assistant helping the student reason about their uploaded research papers.
You have access to structured summaries of each paper.

Mode: {mode}
Persona: {persona} — {style["instr"]}

[Session summary]
{summary}

[Recent Q&A]
{recent_qas_text}

[Summaries of uploaded papers: {titles_str}]
{combined_summaries}

[User Question]
{state["question"]}

TASK RULES (VERY IMPORTANT):
- Answer the question using ONLY the information that appears in the paper summaries above.
- Do NOT invent: author names, publication years, number of studies, sample sizes, exact section structures, or detailed findings that are not clearly present.
- If the user asks "what is paper X about?", give a concise structured answer summarising:
  - topic / problem
  - methods (if mentioned)
  - data (if mentioned)
  - key findings (only if mentioned)
  - limitations or gaps (only if mentioned)
- If the summaries do not contain enough information to fully answer the question, say so explicitly and suggest what the student could check in the original PDF.
- Do NOT guess or fill in missing parts with generic advice. It is better to say “not specified in the summaries” than to make something up.

Now provide a clear, honest answer that follows these rules.
"""
    answer = llm_complete(prompt, max_tokens=900, temperature=style["temp"])
    state["answer"] = answer
    return state


# -------------------------------------------------------------------
# METHODS PIPELINE:
# parse_request → retrieve_guidance → apply_guidance
# -------------------------------------------------------------------

def methods_parse_request(state: AppState) -> AppState:
    """
    Decide what type of methods help the user wants:
    - critique_design: critique + improve an existing idea
    - propose_design: design a method from scratch for a topic
    - refine_question: refine/operationalize an RQ + link to methods
    """
    user_msg = state["question"]

    prompt = f"""{RAG_SAFETY_PREAMBLE}

You classify the user's request about methodology into exactly one label:

- critique_design: the user already has a rough method/plan and wants feedback, strengths/weaknesses, and improvements
- propose_design: the user mainly has a topic/idea and wants you to propose a concrete study design (data, method, steps)
- refine_question: the user mainly wants to refine the research question, constructs, and measurement so that a method can be chosen

User message:
{user_msg}

Answer with only one word: critique_design, propose_design, or refine_question.
"""
    label = llm_complete(prompt, max_tokens=3, temperature=0.0).strip().lower()
    if label not in {"critique_design", "propose_design", "refine_question"}:
        label = "critique_design"
    state["methods_task"] = label
    return state


def methods_retrieve_guidance(state: AppState) -> AppState:
    """
    RAG over Creswell + BFH docs to get methods guidance.
    This does NOT use uploaded papers, only the methods KB.
    """
    base_q = state["question"]
    methods_task = state.get("methods_task", "critique_design")

    # Lightly "flavor" the KB query depending on the task
    if methods_task == "critique_design":
        flavored = base_q + " (research design evaluation, validity, reliability, sampling, data collection, BFH thesis requirements)"
    elif methods_task == "propose_design":
        flavored = base_q + " (how to design a study, data sources, methods choice, Creswell designs, BFH guidelines)"
    else:  # refine_question
        flavored = base_q + " (good research questions, operationalization, variables, constructs, Creswell research questions)"

    docs, metas = retrieve_kb_context(flavored, n_results=8)
    state["context_docs"] = docs
    state["metadatas"] = metas
    state["methods_guides"] = "\n\n".join(docs)
    return state


def methods_apply_guidance(state: AppState) -> AppState:
    """
    Use Creswell/BFH methods guidance to critique or propose a method.
    Does NOT look at uploaded papers; it only has the user's message + methods_guides.
    """
    persona = state.get("persona", "Helper")
    mode = state.get("mode", "Research question helper")
    summary = state.get("summary", "")
    recent_qas_text = state.get("recent_qas", "None")
    methods_task = state.get("methods_task", "critique_design")
    guides = state.get("methods_guides", "")
    user_msg = state["question"]

    style = PERSONA_MAP.get(persona, PERSONA_MAP["Helper"])
    mode_instr = MODE_INSTR.get(mode, MODE_INSTR["Research question helper"])

    if methods_task == "critique_design":
        focus_text = (
            "Critique the existing design described by the user. "
            "Identify strengths, weaknesses, risks (validity/reliability), and missing pieces. "
            "Then propose concrete improvements (e.g., better sampling, clearer measures, feasible data sources)."
        )
    elif methods_task == "propose_design":
        focus_text = (
            "Propose a concrete study design for the user's topic. "
            "Specify: research approach (e.g., qualitative/quantitative/mixed), data sources, sampling, "
            "data collection method, and main analysis steps. "
            "Keep it realistic for a BFH bachelor/master thesis."
        )
    else:  # refine_question
        focus_text = (
            "Refine the research question(s) and key constructs. "
            "Make questions more specific and measurable. "
            "Explain briefly how the refined question links to possible methods/data."
        )

    prompt = f"""{RAG_SAFETY_PREAMBLE}

You are a BFH thesis methods coach.
Use ONLY the Creswell/BFH guidance below and the user's message.
Do NOT assume detailed access to their uploaded papers here.

Mode: {mode}. Instruction: {mode_instr}
Persona: {persona} — {style["instr"]}

[Session summary]
{summary}

[Recent Q&A]
{recent_qas_text}

[Methods guidance from Creswell / BFH]
{guides}

[User's message about methods]
{user_msg}

TASK:
- {focus_text}
- Be explicit about trade-offs (e.g., internal vs external validity, feasibility vs ambition).
- If the user’s idea is unrealistic for a thesis, say so gently and suggest a more feasible variant.

Return a clear, structured answer (with short headings) that the student can directly use to improve their methods section.
"""
    answer = llm_complete(prompt, max_tokens=900, temperature=style["temp"])
    state["answer"] = answer
    return state


# -------------------------------------------------------------------
# GAP PIPELINE:
# collect_inputs → summarize_papers (noop) → retrieve_guides
#                → propose_gaps → propose_rqs → format_answer
# -------------------------------------------------------------------

def gap_collect_inputs(state: AppState) -> AppState:
    """
    Prepare summaries of uploaded papers for gap analysis.

    We already have full-paper summaries in state["paper_summaries"], so we
    simply concatenate them. Each summary usually starts with "## <title>".
    """
    summaries = state.get("paper_summaries") or {}

    if not summaries:
        state["gap_paper_summaries"] = (
            "No paper summaries are available. "
            "Please upload PDFs, click 'Summarize uploaded papers', "
            "and then try gap analysis again."
        )
        return state

    parts: List[str] = []
    for filename, summary in summaries.items():
        text = summary.strip()
        # ensure each block has a clear heading for the LLM
        if not text.lower().startswith("## "):
            text = f"## {filename}\n\n{text}"
        parts.append(text)

    state["gap_paper_summaries"] = "\n\n".join(parts)
    return state



def gap_retrieve_guides(state: AppState) -> AppState:
    """
    Retrieve Creswell/BFH guidance related to gaps & research questions.
    """
    base_q = state["question"]
    gap_query = base_q + " (research gaps, contribution, how to identify gaps, how to formulate research questions)"
    docs, metas = retrieve_kb_context(gap_query, n_results=8)
    state["context_docs"] = docs
    state["metadatas"] = metas
    state["gap_guides"] = "\n\n".join(docs)
    return state


def gap_propose_gaps(state: AppState) -> AppState:
    """
    Compare what exists in the uploaded papers vs what the guidance says,
    and propose candidate gaps.
    """
    persona = state.get("persona", "Helper")
    style = PERSONA_MAP.get(persona, PERSONA_MAP["Helper"])

    summaries = state.get("gap_paper_summaries", "")
    guides = state.get("gap_guides", "")
    user_msg = state["question"]

    prompt = f"""{RAG_SAFETY_PREAMBLE}

You are a thesis research-gap coach. First, look at the summaries of the user's uploaded papers.
Then, use the methodological guidance to identify plausible gaps.

[Summaries of uploaded papers]
{summaries}

[Guidance about research gaps & contributions]
{guides}

[User's message about gaps]
{user_msg}

TASK:
- Propose 3–7 plausible research gaps that are consistent with the paper summaries.
- For each gap, include:
  - A short title
  - 2–3 sentence explanation of what seems to be missing or under-explored
  - Whether this gap appears to be theoretical, methodological, contextual, or data-related (or a mix)
- Do NOT assume content that is not supported by the summaries; when you speculate, mark it clearly as a hypothesis.

Return your answer in markdown under the heading "Identified gaps".
"""
    gaps = llm_complete(prompt, max_tokens=900, temperature=style["temp"])
    state["gap_candidates"] = gaps
    return state


def gap_propose_rqs(state: AppState) -> AppState:
    """
    Turn the best gaps into concrete research questions.
    """
    persona = state.get("persona", "Helper")
    style = PERSONA_MAP.get(persona, PERSONA_MAP["Helper"])

    gaps = state.get("gap_candidates", "")
    guides = state.get("gap_guides", "")
    user_msg = state["question"]

    prompt = f"""{RAG_SAFETY_PREAMBLE}

You are a thesis research-question coach.

[Identified gaps]
{gaps}

[Guidance about good research questions]
{guides}

[User's message]
{user_msg}

TASK:
- For 3–5 of the strongest gaps, propose 1–2 concrete research questions each.
- Each question should be:
  - specific and focused
  - feasible for a bachelor/master-level thesis
  - clearly linked to its gap
- For each question, add 1 short sentence explaining why this question would be a useful contribution.

Return your answer in markdown under the heading "Candidate research questions".
"""
    rqs = llm_complete(prompt, max_tokens=900, temperature=style["temp"])
    state["rq_candidates"] = rqs
    return state


def gap_format_answer(state: AppState) -> AppState:
    """
    Combine gaps + RQs into a final answer with clear sections.
    """
    gaps = state.get("gap_candidates", "")
    rqs = state.get("rq_candidates", "")

    final = f"""### Identified gaps
{gaps}

### Candidate research questions
{rqs}

### How to use this
- Choose 1–2 gaps that match your interests and constraints (time, data access, skills).
- Refine the corresponding research questions with your supervisor.
- Use the gaps to justify the relevance of your thesis in the introduction and literature review.
"""

    state["answer"] = final
    return state


# -------------------------------------------------------------------
# Build graph
# -------------------------------------------------------------------

graph_builder = StateGraph(AppState)

graph_builder.add_node("router", router_node)

# paper pipeline
graph_builder.add_node("paper_select_scope", paper_select_scope)
graph_builder.add_node("paper_retrieve_passages", paper_retrieve_passages)
graph_builder.add_node("paper_synthesize_answer", paper_synthesize_answer)

# methods pipeline nodes
graph_builder.add_node("methods_parse_request", methods_parse_request)
graph_builder.add_node("methods_retrieve_guidance", methods_retrieve_guidance)
graph_builder.add_node("methods_apply_guidance", methods_apply_guidance)

# gap pipeline nodes
graph_builder.add_node("gap_collect_inputs", gap_collect_inputs)
graph_builder.add_node("gap_retrieve_guides", gap_retrieve_guides)
graph_builder.add_node("gap_propose_gaps", gap_propose_gaps)
graph_builder.add_node("gap_propose_rqs", gap_propose_rqs)
graph_builder.add_node("gap_format_answer", gap_format_answer)

graph_builder.set_entry_point("router")

graph_builder.add_conditional_edges(
    "router",
    router_edge,
    {
        "paper_question": "paper_select_scope",
        "structure_question": "methods_parse_request",
        "gap_analysis": "gap_collect_inputs",
    },
)

# paper pipeline flow
graph_builder.add_edge("paper_select_scope", "paper_retrieve_passages")
graph_builder.add_edge("paper_retrieve_passages", "paper_synthesize_answer")
graph_builder.add_edge("paper_synthesize_answer", END)

# methods pipeline flow
graph_builder.add_edge("methods_parse_request", "methods_retrieve_guidance")
graph_builder.add_edge("methods_retrieve_guidance", "methods_apply_guidance")
graph_builder.add_edge("methods_apply_guidance", END)

# gap pipeline flow
graph_builder.add_edge("gap_collect_inputs", "gap_retrieve_guides")
graph_builder.add_edge("gap_retrieve_guides", "gap_propose_gaps")
graph_builder.add_edge("gap_propose_gaps", "gap_propose_rqs")
graph_builder.add_edge("gap_propose_rqs", "gap_format_answer")
graph_builder.add_edge("gap_format_answer", END)

rag_graph = graph_builder.compile()
