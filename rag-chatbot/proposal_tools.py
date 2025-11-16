# proposal_tools.py

# this is just a placeholder to test you can put whatever you want :)

"""
Tools used by the Proposal refinement agent.

Right now this module is very small and mostly wraps the generic LLM call.
Your teammate can extend it later with:
- RAG over Creswell / BFH docs,
- calls to your paper summaries, etc.
"""

from typing import Dict, Any
from rag_tools import llm_complete  # reuse same Together LLM for now


def proposal_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    """
    Simple helper that your anna can use.

    It returns a dict so it's easy to extend later if needed.
    """
    prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    text = llm_complete(prompt, max_tokens=max_tokens, temperature=temperature)
    return {
        "text": text,
        "model": "proposal/llm_complete",  # just a tag
        "usage": {},                      # can be filled later if you track usage
    }
