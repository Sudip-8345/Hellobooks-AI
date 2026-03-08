import panel as pn
from agents import rag_agent

pn.extension()

# Persistent agent state (holds short-term memory across turns)
agent_state = {"history": []}


def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    """Called each time the user sends a message."""
    agent_state["query"] = contents
    result = rag_agent.invoke(dict(agent_state))

    # Carry forward updated history for short-term memory
    agent_state["history"] = result.get("history", [])

    return result["response"]


chat = pn.chat.ChatInterface(
    callback=callback,
    user="You",
    avatar="👤",
    callback_user="HelloBooks Assistant",
    callback_avatar="📚",
    placeholder_text="Thinking...",
    sizing_mode="stretch_width",
    min_height=600,
)

template = pn.template.FastListTemplate(
    title="HelloBooks RAG Q&A",
    main=[chat],
)

template.servable()
