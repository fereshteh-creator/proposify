"""
Visualize the LangGraphs used in Proposify.

Usage from host (with Docker):

    # From project root (where docker-compose.yml lives)
    docker exec -it rag_chatbot_app python /app/visualize_graph.py

If you run it locally (without Docker), you need requirements.txt
installed in your local environment.

Otherwise from rag-chatbot run:

    docker exec -it rag_chatbot_app bash
    python visualize_graph.py

This script now renders BOTH:
- rag_graph         (research question / methods / gaps)
- proposal_graph    (proposal refinement assistant)
"""

from graph_config import rag_graph
from proposal_graph_config import proposal_graph


def show_graph(graph, name: str, png_name: str) -> None:
    g = graph.get_graph()

    # 1) ASCII visualization in the terminal
    print(f"=== {name} ASCII visualization ===")
    try:
        ascii_art = g.draw_ascii()
        print(ascii_art)
    except ImportError as e:
        print(
            f"\n[WARN] Could not render ASCII graph for {name}. "
            "Missing optional dependency:\n"
            f"  {e}\n"
            "Install grandalf in this environment, e.g.:\n"
            "  pip install grandalf\n"
        )

    # 2) Mermaid PNG to file
    try:
        png_bytes = g.draw_mermaid_png()
    except ImportError as e:
        print(
            f"\n[WARN] Could not render PNG graph for {name}. "
            "Missing optional dependency:\n"
            f"  {e}\n"
            "Install grandalf in this environment, e.g.:\n"
            "  pip install grandalf\n"
        )
    else:
        with open(png_name, "wb") as f:
            f.write(png_bytes)
        print(f"\n[OK] Saved {name} PNG to: {png_name}")
        print("-" * 60)


def main() -> None:
    # 1) Your RQ / methods / gaps graph
    show_graph(rag_graph, "rag_graph", "rag_graph.png")

    # 2) Teammate's proposal graph
    show_graph(proposal_graph, "proposal_graph", "proposal_graph.png")


if __name__ == "__main__":
    main()
