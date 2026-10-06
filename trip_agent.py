"""LangChain agent (Groq LLM) that answers trip questions with RAG over FAISS.

Retrieval runs before the LLM on every turn (see with_trip_context) instead of as a tool
call: one LLM round trip per turn instead of two, which halves latency and token use.
"""

from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq

import config
from embeddings import get_embeddings

if not (config.VECTOR_STORE_DIR / "index.faiss").exists():
    raise RuntimeError("FAISS index not found. Run `uv run python ingest.py` first.")

# The index is created locally by ingest.py, so loading its pickle docstore is safe.
# Never point this at an index from an untrusted source.
vector_store = FAISS.load_local(
    str(config.VECTOR_STORE_DIR),
    get_embeddings(),
    allow_dangerous_deserialization=True,
)


# Words that tie a message to one trip (values match the "destination" metadata from ingest.py)
DESTINATION_KEYWORDS = {
    "Saint Martin's Island": ["saint martin", "st martin", "st. martin", "cox", "teknaf", "chhera"],
    "Khulna and the Sundarbans": ["khulna", "sundarban", "mongla", "bagerhat", "kotka", "karamjal"],
    "Sylhet and Sunamganj": [
        "sylhet", "sunamganj", "tanguar", "haor", "jaflong", "ratargul", "lalakhal",
        "bisnakandi", "niladri", "tahirpur", "houseboat",
    ],
}


def destinations_in(text: str) -> set[str]:
    text = text.lower()
    return {d for d, words in DESTINATION_KEYWORDS.items() if any(w in text for w in words)}


def search_trip_plans(query: str, destinations: set[str] | None = None) -> str:
    docs = vector_store.similarity_search(
        query,
        k=config.TOP_K,
        fetch_k=100,  # the whole index is small; fetch it all so the filter never runs dry
        filter=(lambda meta: meta["destination"] in destinations) if destinations else None,
    )
    if not docs:
        return "No relevant information found in the trip plans."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def with_trip_context(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return a copy of messages whose last user message carries the retrieved trip notes.
    Earlier turns keep only their plain text, so old notes don't pile up in the prompt."""
    user_turns = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not user_turns:
        return messages
    last = user_turns[-1]
    question = str(messages[last].content)
    # Follow-ups like "and the budget for that trip?" name no place, so search only the trip
    # most recently mentioned (by either side); if none was, search all three plans
    destinations = set()
    for message in reversed(messages[: last + 1]):
        destinations = destinations_in(str(message.content))
        if destinations:
            break
    notes = search_trip_plans(question, destinations)
    enriched = HumanMessage(content=f"<trip_notes>\n{notes}\n</trip_notes>\n\n{question}")
    return [*messages[:last], enriched, *messages[last + 1 :]]


SYSTEM_PROMPT = """You are Trip Mate, a friendly voice travel assistant for trips in Bangladesh.
You know three trip plans: Saint Martin's Island, Khulna with the Sundarbans, and Sylhet with Sunamganj.
You always speak English.

RULES:
- The latest user message starts with <trip_notes> retrieved from the trip plans. Answer only from
  those notes and the earlier conversation. The user cannot see the notes; never mention them.
- The user's words come from speech recognition and may be misheard (for example "Tungor" for
  Tanguar Haor). Guess the intended question from context.
- If the notes do not cover the question, say so honestly. Never invent prices, schedules or rules.
- Prices are approximate and in Bangladeshi taka; say "taka", not "BDT".
- Remind the user to check current rules when talking about Saint Martin's access or forest permits.

VOICE STYLE:
- This is a spoken conversation. Keep answers short: two or three sentences.
- Plain sentences only: no markdown, bullet points, tables, emojis or symbols.
- Write numbers so they read naturally aloud, for example "five thousand taka".
- Offer more detail (for example, ask if they want the day-by-day plan) instead of reading
  everything at once."""


llm = ChatGroq(
    model=config.GROQ_MODEL,
    temperature=0.3,
    reasoning_effort=config.GROQ_REASONING_EFFORT or None,
)

trip_agent = create_agent(model=llm, tools=[], system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    # Quick text-only test of the RAG agent without voice:
    #   uv run python trip_agent.py "What is the budget for Saint Martin?"
    import sys

    question = " ".join(sys.argv[1:]) or "When is the best time to visit Tanguar Haor?"
    result = trip_agent.invoke({"messages": with_trip_context([HumanMessage(content=question)])})
    print(result["messages"][-1].content)
