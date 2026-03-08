from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from main import answer
import config


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    history: Annotated[list[dict], "Short-term conversation memory"]
    context_str: str
    response: str


# ── Nodes ────────────────────────────────────────────────────────────────────

def build_context_node(state: AgentState) -> dict:
    """Inject recent conversation history into a context string so the LLM
    can reference earlier turns."""
    history = state.get("history", [])
    recent = history[-config.MEMORY_MAX_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return {"context_str": "\n".join(lines)}


def retrieve_and_generate_node(state: AgentState) -> dict:
    """Core RAG node – retrieves documents, generates an answer and
    appends the exchange to short-term memory."""
    query = state["query"]
    response = answer(query)

    # Append current turn to history (short-term memory)
    history = list(state.get("history", []))
    history.append({"user": query, "assistant": response})

    # Trim to the configured window
    if len(history) > config.MEMORY_MAX_TURNS:
        history = history[-config.MEMORY_MAX_TURNS:]

    return {"response": response, "history": history}


# ── Graph construction ───────────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(AgentState)

    graph.add_node("build_context", build_context_node)
    graph.add_node("rag", retrieve_and_generate_node)

    graph.add_edge(START, "build_context")
    graph.add_edge("build_context", "rag")
    graph.add_edge("rag", END)

    return graph.compile()


# Singleton compiled graph
rag_agent = build_rag_graph()