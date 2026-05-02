import asyncio
from app import graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

async def main():
    config = {"configurable": {"thread_id": "test_rag"}}
    print("Testing portfolio RAG via graph...")
    
    events = graph.astream_events(
        {"messages": [HumanMessage(content="What projects has Harshal Sawatkar built?")]},
        version="v2",
        config=config
    )
    
    async for event in events:
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # Print content of chunks as they arrive
            if chunk.content:
                print(chunk.content, end="", flush=True)
        elif event["event"] == "on_tool_end":
            print(f"\n[TOOL END: {event['name']}]")
            print(f"OUTPUT: {event['data']['output']}")

if __name__ == "__main__":
    asyncio.run(main())
