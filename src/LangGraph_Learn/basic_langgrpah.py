from pydantic import BaseModel
from langgraph.graph import StateGraph,START,END

class GreetState(BaseModel):
    message: str=""


graph = StateGraph(GreetState)

def Greet(state:GreetState):
    state.message="Hello ,How are you?"
    return state

graph.add_node('greet',Greet)
graph.add_edge(START,'greet')
graph.add_edge("greet",END)

FinalGraph=graph.compile()
res=FinalGraph.invoke({'message':"heelo i love ai"})

print(res)