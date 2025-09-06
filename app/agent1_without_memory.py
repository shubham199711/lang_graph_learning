from langchain_core import messages
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAI

load_dotenv()


llm = GoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.0-flash",
)


class AgentState(TypedDict):
    messages: list[HumanMessage]


def process(state: AgentState) -> AgentState:
    """Process the messages"""
    llm_response = llm.invoke(state["messages"])
    state["messages"].append(HumanMessage(content=llm_response))
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()


user_input = input("Enter your message: ")
while user_input != "exit":
    result = app.invoke({"messages": [HumanMessage(content=user_input)]})
    print(result)
    user_input = input("Enter your message: ")
