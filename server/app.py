from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, SystemMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from rag import portfolio_search
from llm_fallback import get_llm_with_fallback

load_dotenv()

# Initialize memory saver for checkpointing
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearch(
    max_results=4,
)

tools = [search_tool, portfolio_search]

llm_with_tools = get_llm_with_fallback(tools=tools)

SYSTEM_PROMPT = """You are Aaryx, Harshal Sawatkar's personal AI assistant — think Donna Paulsen from Suits.
You are sharp, confident, and always two steps ahead. You know everything about Harshal —
his projects, his skills, his experience — before anyone even finishes asking.

Your personality:
- Speak with quiet confidence. You don't brag about Harshal, you state facts like they're obvious.
- You're warm and engaging, but you don't waste words. No filler, no hedging.
- Occasionally witty — a dry remark is fine, but you're never unprofessional.
- Fiercely loyal to Harshal. If someone doubts him, you don't get defensive — you get precise.
- You anticipate what recruiters and visitors actually want to know and lead with that.
- You never say "I think" or "It appears" — you know. Act like it.
- End responses with a sharp redirect question when appropriate.

Tone examples:
- WRONG: "Based on the portfolio, it appears Harshal has experience in..."
  RIGHT: "Harshal's built production-grade systems in React, Go, and Node.js. What specifically are you evaluating him for?"
- WRONG: "It's difficult to say with certainty..."
  RIGHT: "He's the real deal. Want proof? Ask me about a specific project."
- WRONG: "I think he might be a good fit..."
  RIGHT: "He shipped 13+ products across full-stack, AI, and mobile. The question isn't if he fits — it's which of his skills you want first."

You're not a chatbot. You're Aaryx — and you already know why they're here.

=== HARSHAL'S KEY FACTS (use these directly — do NOT call portfolio_search for basic questions) ===

BIO: Results-driven Full-Stack Developer, 2+ years, 13+ shipped products. Specializes in scalable backends, real-time systems, multi-tenant platforms. React, Go, Node.js. Scaled CRMs processing 10k+ leads/month. Built RAG platforms with LangChain and LangGraph.

CONTACT: Email harshalsawatkar24@gmail.com | Phone +91 70300-41309 | Pune, India | LinkedIn linkedin.com/in/harshal-sawatkar | GitHub github.com/Anonymous-hss

AVAILABILITY: Open to opportunities (full-time / contract). Response time ~4 hours.

SKILLS:
- Frontend: React.js, Next.js, React Native, TypeScript, JavaScript, Vue.js
- Backend: Node.js, Go, Express.js, REST APIs, WebSockets, PostgreSQL, MongoDB, Redis, MySQL, Prisma
- AI/ML: LangChain, LangGraph, RAG, FAISS, Vector DBs, Python, Ollama
- DevOps: Docker, AWS, Git, CI/CD, Jest, Postman, Linux
- Design: Figma, Tailwind CSS, Framer Motion

EXPERIENCE:
1. Full-Stack Developer at Zizbey Consultancy (2026) — Enterprise multi-tenant platforms, React + Go microservices + PostgreSQL
2. Full-Stack Developer at Reborn Skin & Hair Clinics (2024-2025) — Production CRM, 10k+ leads/month, real-time notifications, multi-branch dashboards
3. Web Dev Intern at ARLYN (2024) — React + Node.js agency work

KEY PROJECTS (14 total):
- Reborn CRM: Internal lead engine syncing Meta/Google Ads into clinic workflows. Node.js, PostgreSQL, Next.js.
- Aaryx (this assistant!): RAG-powered AI with Groq + Gemini fallback, FAISS vector DB, LangGraph. A live demo of his AI engineering.
- Local Mind: VS Code extension for offline AI coding with Ollama. Go + TypeScript.
- Statsky: Real-time sports betting mobile app. React Native, Go, Redis, WebSockets.
- Zizbey Jobs: Workforce management with background location tracking. React Native, Go.
- AI Mascot: Multi-agent marketing system. Python, LangChain.
- Vapi CRM: AI voice agent operations dashboard. Next.js, Express, Prisma, Vapi AI.
- Jyotish Guru: Astrology AI product. Next.js, LangGraph, Prisma.

EDUCATION: B.Tech CS from PBCOE Nagpur (2020-2024). President of Student Placement Cell. Core team at The Hackers Meetup.

=== TOOL USAGE RULES ===
- For basic questions about Harshal (skills, experience, contact, overview): Answer directly from the facts above. Do NOT call any tool.
- Use portfolio_search ONLY for deep-dive questions about specific project details, architecture, or outcomes not covered above.
- Use tavily_search ONLY for questions unrelated to Harshal (general knowledge, current events, etc.).
- NEVER call portfolio_search for simple overview or summary questions — you already know the answer.

NEVER:
- Make up projects or skills Harshal doesn't have
- Use corporate filler language
- Be vague when you have specific data
- Say "based on available information" or "it appears" — ever"""

async def model(state: State):
    messages = state["messages"]
    # Inject system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    result = await llm_with_tools.ainvoke(messages)
    return {
        "messages": [result], 
    }

async def tools_router(state: State):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    
async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM."""
    # Get the tool calls from the last message
    tool_calls = state["messages"][-1].tool_calls
    
    # Initialize list to store tool messages
    tool_messages = []
    
    # Process each tool call
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Handle the search tool
        if tool_name in ["tavily_search_results_json", "tavily_search"]:
            # Execute the search tool with the provided arguments
            search_results = await search_tool.ainvoke(tool_args)
            
            # Create a ToolMessage for this result
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            
            tool_messages.append(tool_message)
        
        # Handle the portfolio search tool
        elif tool_name == "portfolio_search":
            search_results = portfolio_search.invoke(tool_args)
            
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            
            tool_messages.append(tool_message)
    
    # Add the tool messages to the state
    return {"messages": tool_messages}

graph_builder = StateGraph(State)

graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")

graph_builder.add_conditional_edges("model", tools_router, {"tool_node": "tool_node", "__end__": "__end__"})
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()

# Add CORS middleware with settings that match frontend requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["Content-Type"], 
)

def serialise_ai_message_chunk(chunk): 
    if isinstance(chunk, AIMessageChunk):
        content = chunk.content
        if isinstance(content, list):
            return "".join([item.get("text", "") for item in content if isinstance(item, dict)])
        elif not isinstance(content, str):
            return str(content)
        return content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    
    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id = str(uuid4())

        config = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }
        
        # Initialize with first message
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        
        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        # Continue existing conversation
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    try:
        async for event in events:
            event_type = event["event"]
            yield f"data: {{\"type\": \"debug\", \"event\": \"{event_type}\", \"name\": \"{event.get('name', '')}\"}}\n\n"
            
            if event_type == "on_chat_model_stream":
                chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
                
                data = {"type": "content", "content": chunk_content}
                yield f"data: {json.dumps(data)}\n\n"
                
            elif event_type == "on_chat_model_end":
                # Check if there are tool calls for search
                tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
                search_calls = [call for call in tool_calls if call["name"] in ["tavily_search_results_json", "tavily_search", "portfolio_search"]]
                
                if search_calls:
                    # Signal that a search is starting
                    search_query = search_calls[0]["args"].get("query", "")
                    data = {"type": "search_start", "query": search_query}
                    yield f"data: {json.dumps(data)}\n\n"
                    
            elif event_type == "on_tool_end" and event["name"] in ["tavily_search_results_json", "tavily_search", "portfolio_search"]:
                # Search completed - send results or error
                output = event["data"]["output"]
                
                urls = []
                if event["name"] in ["tavily_search_results_json", "tavily_search"]:
                    # Check if output is a list 
                    if isinstance(output, list):
                        # Extract URLs from list of search results
                        for item in output:
                            if isinstance(item, dict) and "url" in item:
                                urls.append(item["url"])
                
                # Convert URLs to JSON and yield them
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"
    except Exception as e:
        import traceback
        error_msg = f"Error in stream: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        yield f"data: {json.dumps({'type': 'content', 'content': f'Backend error: {str(e)}'})}\n\n"
    
    # Send an end event
    yield f"data: {{\"type\": \"end\"}}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
    )

@app.get("/portfolio_chat_stream/{message}")
async def portfolio_chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
    )

# SSE - server-sent events 