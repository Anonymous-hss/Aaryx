from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import os

load_dotenv()

from app import graph

async def main():
    config = {"configurable": {"thread_id": "test_124"}}
    events = graph.astream_events(
        {"messages": [HumanMessage(content="What is weather of nagpur")]},
        version="v2",
        config=config
    )
    
    async for event in events:
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print("TYPE:", type(chunk.content), "CONTENT:", repr(chunk.content))
        elif event["event"] == "on_tool_end":
            print("TOOL END:", event["name"])
        elif event["event"] == "on_chat_model_end":
            print("MODEL END")

if __name__ == "__main__":
    asyncio.run(main())
