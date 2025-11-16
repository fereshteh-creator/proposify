# proposal_graph_config.py

# this is just a placeholder to test you can put whatever you want :)
from typing import TypedDict

from langgraph.graph import StateGraph, END

from prompts import RAG_SAFETY_PREAMBLE, MODE_INSTR, PERSONA_MAP
from rag_tools import llm_complete


class ProposalState(TypedDict):
    question: str
    mode: str
    persona: str
    summary: str
    recent_qas: str
    task: str
    answer: str


def proposal_refine_node(state: ProposalState) -> ProposalState:
    """
    Single-node proposal refinement agent.

    Used when the user selects "Proposal refinement assistant" in the UI.
    It does NOT use RAG yet – just the LLM – so your teammate can
    later expand this into a full graph if needed.
    """
    user_msg = state["question"]
    mode = state.get("mode", "Proposal refinement assistant")
    persona = state.get("persona", "Helper")
    summary = state.get("summary", "")
    recent_qas = state.get("recent_qas", "None")

    style = PERSONA_MAP.get(persona, PERSONA_MAP["Helper"])
    mode_instr = MODE_INSTR.get(
        "Proposal refinement assistant",
        MODE_INSTR["Proposal refinement assistant"],
    )

    prompt = f"""{RAG_SAFETY_PREAMBLE}

You are a BFH thesis PROPOSAL refinement assistant.

Mode: {mode}. Instruction: {mode_instr}
Persona: {persona} — {style["instr"]}

[Session summary]
{summary}

[Recent Q&A]
{recent_qas}

[Student's message about their proposal]
{user_msg}

TASK:
1. First, restate your understanding of the student's planned thesis in 2–3 bullet points.
2. Then give feedback under these headings:
   - Strengths
   - Weaknesses / risks
   - Suggestions to refine the research question
   - Suggestions to refine methods / data
3. Keep it concrete, realistic, and feasible for a BFH bachelor thesis.
4. If the student only writes a very short idea, intelligently suggest 1–2 clearer,
   more precise versions of their proposal.

Return a clear markdown answer with these headings and bullet points.
"""

    answer = llm_complete(prompt, max_tokens=900, temperature=style["temp"])
    state["answer"] = answer
    state["task"] = "proposal_refine"
    return state


# Build a tiny graph: entry → proposal_refine_node → END
graph_builder = StateGraph(ProposalState)
graph_builder.add_node("proposal_refine", proposal_refine_node)
graph_builder.set_entry_point("proposal_refine")
graph_builder.add_edge("proposal_refine", END)

proposal_graph = graph_builder.compile()
