from operator import mul
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """This is addition function which add two numbers int, int"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """This is multiply function which multiply two numbers int, int"""
    return a + b


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [add, multiply]

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.0-flash",
)
llm_with_tools = llm.bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    """This is used by chatbot to call model"""
    system_message = SystemMessage(
        content="You are an AI agent please answer the question and use tools when needed"
    )

    response = llm_with_tools.invoke([system_message] + state["messages"])

    print(f"AI : {response.content}")

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent", should_continue, {"continue": "tools", "end": END}
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


agent_state = {"messages": []}
user_input = input("What is your question: ")
agent_state["messages"].append(("user", user_input))
while user_input != "exit":
    result = app.invoke(agent_state)
    user_input = input("What is your question: ")
    agent_state["messages"].append(("user", user_input))
