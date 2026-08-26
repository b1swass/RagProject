from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph ,START,END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import List,Annotated

memory=InMemorySaver()

class ChatState(BaseModel):
    messages:Annotated[List,add_messages]


llm=init_chat_model(
    model="mistral-small-latest"

)
def chatBotNode(state:ChatState)->ChatState:
    res=llm.invoke(state.messages)
    state.messages=[res]
    return state

graph=StateGraph(ChatState)
graph.add_node('chatBot',chatBotNode)
graph.add_edge(START,"chatBot")
graph.add_edge('chatBot',END)
graph=graph.compile(checkpointer=memory)

while True:
    query=input("\n Ask AI:")
    if query.lower()=="quit":
        break
    res=graph.invoke({"messages":[{'role':"user",'content':query}]},
                     {"configurable":{'thread_id':"1"}})
    result=res['messages'][-1].content
    print("\n\nAI: ",result)

    
    