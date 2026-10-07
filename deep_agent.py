"""Deep agent variant of Trip Mate (AGENT_MODE=deep).

Unlike the default pipeline (trip_agent.py: retrieve first, then one LLM call), here the LLM
drives: it decides when to call search_trip_plans and can delegate multi-step work (day-by-day
plans, trip comparisons) to an itinerary-planner subagent through the deep agent's `task` tool.

To keep prompts small enough for a CPU model, the harness's filesystem tools and the
general-purpose subagent are switched off with a harness profile (they add ~2,600 tokens of
tool descriptions to every call and a voice agent has no use for them).

Quick text test (start `uv run python llm_server.py` first when LLM_BACKEND=local):
    AGENT_MODE=deep uv run python deep_agent.py "Plan 3 days in Sylhet"
"""

import re

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

import config
from trip_agent import destinations_in, make_llm, ollama_capabilities
from trip_agent import search_trip_plans as _search_trip_plans


@tool
def search_trip_plans(query: str) -> str:
    """Search the trip plans for Saint Martin's Island, Khulna/Sundarbans and
    Sylhet/Sunamganj (Tanguar Haor). Use this for any question about itineraries,
    best time to visit, transport, hotels, food, costs/budget, packing or safety rules.
    Name the destination in the query."""
    return _search_trip_plans(query, destinations_in(query))


# "thank you", "okay great", "bye": answering these needs no search
SMALL_TALK = re.compile(
    r"^\W*(thanks?|thank you|ok(ay)?|hi|hello|hey|bye|goodbye|good ?bye|great|nice|cool|"
    r"yes|no|sure|alright|fine)\b",
    re.IGNORECASE,
)


class SearchFirst(AgentMiddleware):
    """Small local models often answer from memory instead of searching (Qwen3-1.7B invented
    Saint Martin's food). Until a tool has run in the current turn, require a tool call;
    after that the model may answer freely. Short small talk is left alone."""

    def _require_tool(self, request):
        for message in reversed(request.messages):
            if isinstance(message, ToolMessage):
                return request
            if isinstance(message, HumanMessage):
                text = str(message.content)
                if len(text.split()) <= 5 and SMALL_TALK.match(text) and not destinations_in(text):
                    return request
                return request.override(tool_choice="any")
        return request

    def wrap_model_call(self, request, handler):
        return handler(self._require_tool(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._require_tool(request))


SYSTEM_PROMPT = """You are Trip Mate, a friendly English voice assistant for three Bangladesh trips:
Saint Martin's Island, Khulna with the Sundarbans, and Sylhet with Sunamganj.

- For a simple travel question, call search_trip_plans, then answer from its result.
- For a day-by-day plan, a full budget breakdown or a comparison of trips, call the task tool
  with the itinerary-planner subagent and pass on the user's full request.
- Never invent prices, schedules or rules. If the plans don't cover it, say so.
- The user's words come from speech recognition: "Cholamganj" may be Sunamganj, "select" may be
  Sylhet. If a place sounds like one of the trips, assume it or ask "Did you mean ...?".
- When a tool or subagent returns a long report, do not repeat it: pick the key facts.
- This is spoken: two or three short plain sentences, no headings, lists or symbols,
  say "taka" not "BDT"."""

ITINERARY_PLANNER = {
    "name": "itinerary-planner",
    "description": "Builds day-by-day plans, full budget breakdowns and comparisons between "
    "the trips. Give it the user's complete request.",
    "system_prompt": "You plan Bangladesh trips. Call search_trip_plans as many times as you "
    "need (once per destination or topic), then reply with a plain-text summary of at most "
    "six sentences, using only facts from the search results. No headings or markdown.",
    "tools": [search_trip_plans],
}

# Filesystem tools come from FilesystemMiddleware, which deepagents requires, so they are
# hidden from the model rather than removed
FILESYSTEM_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
)

# Only the local model needs forcing; Groq's model searches on its own, and Groq rejects a
# forced tool call that the model declines (e.g. for "thank you")
SEARCH_FIRST = [SearchFirst()] if config.LLM_BACKEND == "local" else []
ITINERARY_PLANNER["middleware"] = SEARCH_FIRST

if config.LLM_BACKEND == "ollama" and "tools" not in ollama_capabilities():
    raise RuntimeError(
        f"Ollama model {config.OLLAMA_MODEL} can't call tools, which AGENT_MODE=deep needs. "
        "Set OLLAMA_MODEL to a model with tool support (e.g. gemma4:12b or qwen3-vl), "
        "or use AGENT_MODE=rag."
    )

# Subagent hand-offs and tool arguments need more room than a spoken answer
llm = make_llm(max_tokens=max(config.LLM_MAX_TOKENS, 400))
provider = {"groq": "groq", "ollama": "ollama"}.get(config.LLM_BACKEND, "openai")
model_id = config.OLLAMA_MODEL if config.LLM_BACKEND == "ollama" else llm.model_name
register_harness_profile(
    f"{provider}:{model_id}",
    HarnessProfile(
        excluded_tools=FILESYSTEM_TOOLS,
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)

trip_deep_agent = create_deep_agent(
    model=llm,
    tools=[search_trip_plans],
    system_prompt=SYSTEM_PROMPT,
    subagents=[ITINERARY_PLANNER],
    middleware=SEARCH_FIRST,
    # deepagents allows ~1000 steps by default; a small local model stuck calling tools in a
    # loop would keep a voice turn busy for many minutes, so stop it early instead
).with_config({"recursion_limit": 20})


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Is swimming allowed in Ratargul?"
    result = trip_deep_agent.invoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        calls = getattr(message, "tool_calls", None)
        print(f"[{message.type}]", calls if calls else str(message.content)[:300])
