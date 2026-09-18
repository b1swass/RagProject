from fastapi import FastAPI
from langserve import add_routes
from agent import agent

app=FastAPI(
    title="Google Search Agent"
)

@app.get("/home")
def weather():
    return {"status":"Very sunny and windy"}
add_routes(app,
           agent,
           path="/search-agent")