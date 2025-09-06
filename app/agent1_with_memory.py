from langchain_core.messages import HumanMessage, AIMessage
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
    messages: list[HumanMessage | AIMessage]


def process(state: AgentState) -> AgentState:
    """Process the messages"""
    llm_response = llm.invoke(state["messages"])
    print("AI Response: ", llm_response)
    state["messages"].append(AIMessage(content=llm_response))
    print("Agent Current State: ", state["messages"])
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()


user_input = input("Enter your message: ")

agent_state = AgentState(messages=[])
if user_input != "exit":
    agent_state["messages"].append(HumanMessage(content=user_input))

while user_input != "exit":
    result = app.invoke(agent_state)
    user_input = input("Enter your message: ")
    if user_input != "exit":
        agent_state["messages"].append(HumanMessage(content=user_input))

with open("agent1_with_memory_state.json", "w") as f:
    for message in agent_state["messages"]:
        if isinstance(message, HumanMessage):
            f.write("HumanMessage: " + str(message.content) + "\n")
        elif isinstance(message, AIMessage):
            f.write("AIMessage: " + str(message.content) + "\n")
        f.write("\n")
    print("Agent State Saved to agent1_with_memory_state.json")
