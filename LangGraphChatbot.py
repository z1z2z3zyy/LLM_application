from typing import Annotated,TypedDict

from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import os
from langgraph.checkpoint.memory import MemorySaver


from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages:Annotated[list,add_messages]

def chatbot(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

def stream_graph_updates(user_input: str,config):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},config,stream_mode="values",):
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)
        event["messages"][-1].pretty_print()

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

if __name__=="__main__":
    load_dotenv(override=True)
    graph_builder=StateGraph(State)
    llm = ChatTongyi(
        streaming=True,
        model="qwen-plus"
    )
    tool = TavilySearch(max_results=2,tavily_api_key="")
    tools = [tool,multiply]
    llm_with_tools=llm.bind_tools(tools)
    tool_node = ToolNode(tools=[tool,multiply])
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("chatbot", END)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    graph = graph_builder.compile(checkpointer=memory)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input,config)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input,config)
            break



