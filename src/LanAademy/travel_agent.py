from dotenv import load_dotenv
load_dotenv()

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


memory = InMemorySaver()


async def main():

    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            }
        }
    )

    tools = await client.get_tools()

    print("\nAvailable tools:")
    for tool in tools:
        print(f"- {tool.name}")

    agent = create_agent(
        model="mistral-small-latest",
        tools=tools,
        checkpointer=memory,
        system_prompt=(
            "You are a helpful travel agent. "
            "Search for flights when asked. "
            "Do not ask follow-up questions."
        ),
    )

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            config,
        )

        print("\nAgent:", response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())