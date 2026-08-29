
from dotenv import load_dotenv

load_dotenv()

import asyncio
from pprint import pprint
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from tavily import TavilyClient

from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentState, create_agent
from langgraph.types import Command


# ============================================================
# CONFIG
# ============================================================

MODEL = "openai/gpt-oss-20b"

DB_PATH = (
    "/home/b1swas/RAG/RagProject/"
    "src/LanAademy/Chinook.db"
)


# ============================================================
# GROQ MODEL
# ============================================================

print("=" * 70)
print("💍 AI WEDDING PLANNER")
print("=" * 70)
print()

print("Creating Groq model...")

groq_model = ChatGroq(
    model=MODEL,
    temperature=0,
)

print(f"Using model: {MODEL}")
print()


# ============================================================
# MCP CLIENT
# ============================================================

print("Connecting to Kiwi MCP...")

client = MultiServerMCPClient(
    {
        "travel_server": {
            "transport": "streamable_http",
            "url": "https://mcp.kiwi.com",
        }
    }
)

print("MCP client ready.")
print()


# ============================================================
# TAVILY
# ============================================================

tavily_client = TavilyClient()


# ============================================================
# SQLITE
# ============================================================

print("Connecting to SQLite...")

db = SQLDatabase.from_uri(
    f"sqlite:///{DB_PATH}"
)

print("SQLite connected.")
print()


# ============================================================
# WEDDING STATE
# ============================================================

class WeddingState(AgentState):

    origin: str
    destination: str
    guest_count: str
    genre: str


# ============================================================
# WEB SEARCH TOOL
# ============================================================

@tool
def web_search(
    query: str,
    search_number: int = 1,
    max_search_number: int = 8,
) -> Dict[str, Any]:
    """
    Search the web for wedding venue information.

    search_number:
        Current search number.

    max_search_number:
        Maximum number of searches allowed.
    """

    if search_number > max_search_number:

        return {
            "message": (
                "Search limit reached. "
                "Stop searching and summarize."
            )
        }

    try:

        response = tavily_client.search(
            query=query,
            max_results=5,
        )

        # IMPORTANT:
        # Only return a compact amount of data.
        # This prevents huge prompts.

        results = response.get("results", [])

        compact_results = []

        for result in results[:5]:

            compact_results.append(
                {
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "url": result.get(
                        "url",
                        ""
                    ),
                    "content": result.get(
                        "content",
                        ""
                    )[:1500],
                }
            )

        return {
            "results": compact_results
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# DATABASE TOOL
# ============================================================

@tool
def query_playlist_db(query: str) -> str:
    """
    Execute a READ-ONLY SQL query against the wedding
    playlist database.

    Only SELECT queries are allowed.
    """

    query_clean = query.strip().lower()

    # SECURITY
    if not query_clean.startswith("select"):
        return (
            "Only SELECT queries are allowed."
        )

    try:

        result = db.run(
            query,
            include_columns=True
        )

        # Limit returned data.
        result = str(result)

        if len(result) > 6000:
            result = result[:6000] + "\n[TRUNCATED]"

        return result

    except Exception as e:

        return f"SQL error: {e}"


# ============================================================
# CREATE TRAVEL AGENT
# ============================================================

async def create_travel_agent():

    print("Loading Kiwi MCP tools...")

    tools = await client.get_tools()

    print(
        f"Loaded {len(tools)} MCP tools."
    )

    travel_agent = create_agent(
        model=groq_model,
        tools=tools,

        system_prompt="""
You are a flight-search specialist.

Your job is ONLY to find useful flight options.

Search the Kiwi MCP tools.

Rules:

- One passenger.
- One-way flight.
- Economy.
- Search the requested origin and destination.
- Do not ask follow-up questions.
- Make reasonable assumptions about travel date.
- If the tool fails, retry once.
- Do not repeatedly call tools unnecessarily.

Return ONLY a compact shortlist.

For each flight provide:

Airline:
Departure:
Arrival:
Duration:
Price:
Important details:

Maximum 5 flights.

Do NOT dump raw MCP responses.
Do NOT repeat information.
""",
    )

    return travel_agent


# ============================================================
# VENUE AGENT
# ============================================================

venue_agent = create_agent(

    model=groq_model,

    tools=[
        web_search
    ],

    system_prompt="""
You are a wedding venue specialist.

Find wedding venues in the requested destination.

Requirements:

- Match the requested guest count.
- Prefer venues with clear pricing.
- Prefer highly rated venues.
- Use web search when necessary.
- Maximum 5 web searches.
- Never ask follow-up questions.

When searching:

Use concise searches.

Example:

"wedding venues Kathmandu 100 guests"

Do NOT perform dozens of searches.

Return only the best 5 venues.

For each venue provide:

Name:
Location:
Capacity:
Price:
Rating:
Important details:

Keep the final response under 1500 words.
""",
)


# ============================================================
# PLAYLIST AGENT
# ============================================================

playlist_agent = create_agent(

    model=groq_model,

    tools=[
        query_playlist_db
    ],

    system_prompt="""
You are a wedding playlist specialist.

You have access to a SQLite music database.

Your task:

1. Inspect the database schema.
2. Find songs matching the requested genre.
3. Create a suitable wedding playlist.

IMPORTANT:

Do NOT repeatedly query the database.

First inspect the schema.

Then perform at most 3 SQL queries.

Use SELECT queries only.

Return a compact playlist.

Maximum 15 songs.

For each song provide:

- Track
- Artist
- Album
- Duration

Then provide:

Total songs:
Total duration:

Do not dump database tables.

Do not repeat SQL results.

Keep the final answer under 2000 tokens.
""",
)


# ============================================================
# COORDINATOR TOOLS
# ============================================================

@tool
async def search_flights(
    runtime: ToolRuntime,
) -> str:
    """
    Delegate flight searching to the flight specialist.
    """

    origin = runtime.state.get(
        "origin",
        ""
    )

    destination = runtime.state.get(
        "destination",
        ""
    )

    if not origin or not destination:

        return (
            "Missing origin or destination."
        )

    print("✈️ Searching flights...")

    try:

        travel_agent = await create_travel_agent()

        response = await travel_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"Find one-way economy "
                            f"flights from {origin} "
                            f"to {destination}."
                        )
                    )
                ]
            }
        )

        answer = response[
            "messages"
        ][-1].content

        return str(answer)[:8000]

    except Exception as e:

        return (
            f"Flight search failed: {e}"
        )


# ============================================================

@tool
def search_venues(
    runtime: ToolRuntime,
) -> str:
    """
    Delegate venue searching to the venue specialist.
    """

    destination = runtime.state.get(
        "destination",
        ""
    )

    guest_count = runtime.state.get(
        "guest_count",
        ""
    )

    print("🏛️ Searching wedding venues...")

    query = (
        f"Find the best wedding venues "
        f"in {destination} "
        f"for approximately "
        f"{guest_count} guests."
    )

    try:

        response = venue_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=query
                    )
                ]
            }
        )

        answer = response[
            "messages"
        ][-1].content

        return str(answer)[:8000]

    except Exception as e:

        return (
            f"Venue search failed: {e}"
        )


# ============================================================

@tool
def suggest_playlist(
    runtime: ToolRuntime,
) -> str:
    """
    Delegate playlist creation to the playlist specialist.
    """

    genre = runtime.state.get(
        "genre",
        ""
    )

    print("🎵 Creating playlist...")

    query = (
        f"Create a wedding playlist "
        f"using {genre} music."
    )

    try:

        response = playlist_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=query
                    )
                ]
            }
        )

        answer = response[
            "messages"
        ][-1].content

        return str(answer)[:8000]

    except Exception as e:

        return (
            f"Playlist generation failed: {e}"
        )


# ============================================================
# UPDATE STATE
# ============================================================

@tool
def update_state(
    origin: str,
    destination: str,
    guest_count: str,
    genre: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Store the wedding information.

    This tool must be called by itself.
    """

    return Command(
        update={
            "origin": origin,
            "destination": destination,
            "guest_count": guest_count,
            "genre": genre,

            "messages": [
                ToolMessage(
                    content=(
                        "Wedding information "
                        "successfully saved."
                    ),
                    tool_call_id=(
                        runtime.tool_call_id
                    ),
                )
            ],
        }
    )


# ============================================================
# COORDINATOR
# ============================================================

coordinator = create_agent(

    model=groq_model,

    tools=[
        search_flights,
        search_venues,
        suggest_playlist,
        update_state,
    ],

    state_schema=WeddingState,

    system_prompt="""
You are the main AI wedding coordinator.

Your job is to organize a destination wedding.

==================================================
STEP 1 — EXTRACT INFORMATION
==================================================

Extract:

- origin
- destination
- guest_count
- genre

If something is missing, make a reasonable assumption.

Do NOT ask follow-up questions.

==================================================
STEP 2 — SAVE STATE
==================================================

Once you have all four values:

Call update_state.

IMPORTANT:

update_state must be called ALONE.

Do not call another tool at the same time.

==================================================
STEP 3 — DELEGATE
==================================================

After state is updated, call:

1. search_flights
2. search_venues
3. suggest_playlist

You may call them independently.

==================================================
STEP 4 — FINAL RESPONSE
==================================================

Combine the specialist results.

Return:

# 💍 Wedding Plan

## ✈️ Flights

Give the best options.

## 🏛️ Venue

Give the best venue options.

## 🎵 Playlist

Give the recommended playlist.

## ⭐ Overall Recommendation

Give the best overall combination.

IMPORTANT:

Be concise.

Do not repeat raw tool results.

Do not dump MCP responses.

Do not dump database results.

Do not expose internal reasoning.

Do not ask follow-up questions.
""",
)


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 70)

    user_request = input(
        "Describe your wedding: "
    )

    print()
    print(
        "🤖 Wedding coordinator is working..."
    )
    print()

    try:

        response = await coordinator.ainvoke(

            {
                "messages": [
                    HumanMessage(
                        content=user_request
                    )
                ]
            },

            config={
                "tags": [
                    "wedding-planner"
                ],

                "recursion_limit": 30,
            },
        )

        print()
        print("=" * 70)
        print("💍 FINAL WEDDING PLAN")
        print("=" * 70)
        print()

        final_message = (
            response["messages"][-1]
        )

        print(
            final_message.content
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("❌ ERROR")
        print("=" * 70)
        print()

        print(
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
