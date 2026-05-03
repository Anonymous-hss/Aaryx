import asyncio
from dotenv import load_dotenv
load_dotenv()
from app import llm_primary_with_tools
from langchain_core.messages import HumanMessage

async def main():
    print("Testing astream...")
    response = None
    async for chunk in llm_primary_with_tools.astream([HumanMessage(content="Hello")]):
        print(chunk.content, end="", flush=True)
        if response is None:
            response = chunk
        else:
            response += chunk
    print("\nDone")

if __name__ == "__main__":
    asyncio.run(main())
