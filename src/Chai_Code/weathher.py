from dotenv import load_dotenv
load_dotenv()

import requests

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# ============================================================
# TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    print(f"\n🔧 Tool Called: get_weather({city})")

    url = f"https://wttr.in/{city}?format=%c+%t"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            return response.text.strip()

        return f"Weather API returned status code {response.status_code}"

    except requests.RequestException as e:
        return f"Weather API error: {str(e)}"


# ============================================================
# LLM
# ============================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are a helpful AI assistant specialized in solving user queries.

You have access to tools.

Follow this workflow:

START
↓
PLAN
↓
ACTION
↓
OBSERVE
↓
OUTPUT

Rules:

1. START
Understand the user's request.

2. PLAN
Briefly state what needs to be done.
Do not reveal hidden chain-of-thought.

3. ACTION
If a tool is required, call the appropriate tool.
Never pretend that a tool was called.

4. OBSERVE
Use the actual result returned by the tool.

5. OUTPUT
Give the user the final answer clearly.

Available tools:

- get_weather(city)
  Use this when the user asks about current weather.

Important:
Do not invent tool results.
Do not expose hidden reasoning.
"""


# ============================================================
# TOOLS
# ============================================================

tools = [
    get_weather
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# CONTINUOUS CHAT
# ============================================================

print("\n======================================")
print("        🤖 AI AGENT STARTED")
print("======================================")
print("Type 0 to exit.\n")


while True:

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    query = input("You: ")

    # Exit
    if query.strip() == "0":
        print("\n👋 Goodbye!")
        break


    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    print("\n========== START ==========")
    print("Processing user query...")


    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    print("\n========== PLAN ==========")
    print("Determining what needs to be done...")


    # --------------------------------------------------------
    # FIRST LLM CALL
    # --------------------------------------------------------

    response = llm_with_tools.invoke([
        HumanMessage(
            content=f"""
{system_prompt}

User query:
{query}
"""
        )
    ])


    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    if response.tool_calls:

        print("\n========== ACTION ==========")

        tool_messages = []

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"Function: {tool_name}")
            print(f"Input: {tool_args}")


            # ------------------------------------------------
            # EXECUTE TOOL
            # ------------------------------------------------

            if tool_name == "get_weather":

                tool_result = get_weather.invoke(tool_args)

                print("\n========== OBSERVE ==========")
                print("Tool Result:", tool_result)


                # --------------------------------------------
                # Send result back to LLM
                # --------------------------------------------

                tool_message = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )

                tool_messages.append(tool_message)


        # ----------------------------------------------------
        # FINAL LLM CALL
        # ----------------------------------------------------

        final_response = llm_with_tools.invoke(
            [
                HumanMessage(
                    content=f"""
{system_prompt}

User query:
{query}
"""
                ),
                response,
                *tool_messages
            ]
        )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print("\n========== OUTPUT ==========")
        print(final_response.content)


    # --------------------------------------------------------
    # NO TOOL REQUIRED
    # --------------------------------------------------------

    else:

        print("\n========== OUTPUT ==========")
        print(response.content)


    print("\n--------------------------------------\n")