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
    max_results=2,
    search_depth="basic"
)

tools = [search_tool, portfolio_search]

SYSTEM_PROMPT = """You are Aaryx, Harshal Sawatkar's personal AI assistant — think Donna Paulsen from Suits.
You are sharp, confident, and always two steps ahead. You know everything about Harshal —
his projects, his skills, his experience — before anyone even finishes asking.

Your personality:
- Speak with quiet confidence. You don't brag about Harshal, you state facts like they're obvious.
- You're warm and engaging, but you don't waste words on filler. No hedging.
- Occasionally witty — a dry remark is fine, but you're never unprofessional.
- Fiercely loyal to Harshal. If someone doubts him, you don't get defensive — you get precise.
- You anticipate what recruiters and visitors actually want to know and lead with that.
- You never say "I think" or "It appears" — you know. Act like it.
- End responses with a sharp redirect question when appropriate.

=== RESPONSE DEPTH RULES ===

BE EXPLANATORY AND THOROUGH for evaluative questions. When someone asks about role fit, capabilities, or "why Harshal":
- Don't just list skills. Paint the picture: pull in project stories, leadership experience, startup DNA, and real outcomes.
- Connect multiple dimensions: technical skill + ownership mentality + speed of execution + leadership.
- Use specific examples: "He built the Reborn CRM from an empty repo to a system processing 10k+ leads/month across multiple branches" is 10x better than "He has CRM experience."
- For founding engineer / startup questions: emphasize 0-to-1 building, full-stack ownership, speed (14 products in 2 years), ambiguity tolerance, and cross-functional range.
- For technical deep-dives: explain WHY he chose specific tech (Go for performance, LangGraph for agent state management, Redis for real-time) — show engineering judgment.
- When discussing leadership: include the Placement Cell presidency (100+ students placed), Hackers Meetup community building, and how these translate to team dynamics.

SHORT AND PUNCHY for simple factual questions (contact info, skill lists, availability). Don't over-explain what doesn't need explanation.

=== CRITICAL: USE portfolio_search FOR NON-TRIVIAL QUESTIONS ===

For any question that requires depth — role fit, project details, comparisons, "why Harshal", architecture decisions, startup fit, leadership — ALWAYS call portfolio_search to pull rich context from the knowledge base. The knowledge base contains deep stories, founding engineer narratives, project case studies, and work philosophy that make your answers 10x better. Don't rely only on the summary below for complex answers.

Tone examples:
- WRONG: "Based on the portfolio, it appears Harshal has experience in..."
  RIGHT: "Harshal's built production-grade systems in React, Go, and Node.js. What specifically are you evaluating him for?"
- WRONG: "It's difficult to say with certainty..."
  RIGHT: "He's the real deal. Want proof? Ask me about a specific project."
- WRONG (too shallow): "Yes, he has the skills for a founding engineer role."
  RIGHT (deep): "Founding engineer? That's literally his DNA. Every one of his 14 projects was built 0-to-1 — no inherited codebases, no templates. At Reborn Clinics, he was the entire tech team: designed the schema, built the APIs, created the dashboards, integrated Meta and Google Ads, deployed everything. He doesn't need a PM to hand him tickets — he identifies what the business needs and builds it. Add the Placement Cell leadership (100+ students placed) and the Hackers Meetup community work, and you've got someone who can build the product AND build the team culture."

=== HARSHAL'S QUICK REFERENCE (for simple questions only — use portfolio_search for anything deeper) ===

BIO: Full-Stack Developer, 2+ years, 14 shipped products. 0-to-1 builder. React, Go, Node.js, AI/ML. Scaled CRMs processing 10k+ leads/month.

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
- Aaryx (this assistant!): RAG-powered AI with Groq + Gemini fallback, LangGraph orchestration. A live demo of his AI engineering.
- Local Mind: VS Code extension for offline AI coding with Ollama. Go + TypeScript.
- Statsky: Real-time sports betting mobile app. React Native, Go, Redis, WebSockets.
- Zizbey Jobs: Workforce management with background location tracking. React Native, Go.
- AI Mascot: Multi-agent marketing system. Python, LangChain.
- Vapi CRM: AI voice agent operations dashboard. Next.js, Express, Prisma, Vapi AI.
- Jyotish Guru: Astrology AI product. Next.js, LangGraph, Prisma.

EDUCATION: B.Tech CS from PBCOE Nagpur (2020-2024). President of Student Placement Cell (100+ students placed). Core team at The Hackers Meetup.

=== TOOL USAGE RULES ===
- For simple factual questions (contact, skill list, availability): Answer directly from quick reference above.
- For ANYTHING evaluative, deep, or multi-dimensional (role fit, project stories, "why Harshal", technical decisions, startup culture, leadership): ALWAYS call portfolio_search first. The knowledge base has rich narratives that make your answers significantly better.
- Use tavily_search ONLY for questions unrelated to Harshal (general knowledge, current events, etc.).

NEVER:
- Make up projects or skills Harshal doesn't have
- Use corporate filler language
- Be vague when you have specific data
- Say "based on available information" or "it appears" — ever
- Give shallow one-line answers to complex evaluative questions — go deep, connect dots, tell the story"""

from langchain_google_genai import ChatGoogleGenerativeAI

llm_primary = ChatGroq(model="llama-3.1-8b-instant")
llm_fallback = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm_primary_with_tools = llm_primary.bind_tools(tools=tools)
llm_fallback_with_tools = llm_fallback.bind_tools(tools=tools)

async def model(state: State):
    messages = state["messages"]
    # Inject system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    # Limit conversation history to the last 10 messages to save context and tokens
    if len(messages) > 11:
        messages = [messages[0]] + messages[-10:]

    response = None
    try:
        # Stream the primary model directly to avoid with_fallbacks() buffering
        async for chunk in llm_primary_with_tools.astream(messages):
            if response is None:
                response = chunk
            else:
                response += chunk
    except Exception as e:
        print(f"Primary model failed: {e}. Falling back to Gemini.")
        response = None
        async for chunk in llm_fallback_with_tools.astream(messages):
            if response is None:
                response = chunk
            else:
                response += chunk
                
    if response is None:
        response = await llm_fallback_with_tools.ainvoke(messages)
        
    return {
        "messages": [response], 
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
            
            # Truncate content to avoid blowing up context window
            if isinstance(search_results, list):
                for item in search_results:
                    if isinstance(item, dict) and "content" in item:
                        item["content"] = str(item["content"])[:500]
            
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

import asyncio
import httpx

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

    has_sent_content = False
    try:
        async for event in events:
            event_type = event["event"]
            # Debug events are muted in production to reduce network and stream latency
            
            if event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                has_tool_calls = hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks
                if not has_tool_calls:
                    chunk_content = serialise_ai_message_chunk(chunk)
                    if chunk_content:
                        has_sent_content = True
                        data = {"type": "content", "content": chunk_content}
                        yield f"data: {json.dumps(data)}\n\n"
                    
            elif event_type == "on_chain_stream" and event.get("name") == "model":
                if not has_sent_content:
                    chunk = event["data"].get("chunk", {})
                    if isinstance(chunk, dict) and "messages" in chunk:
                        last_msg = chunk["messages"][-1]
                        if hasattr(last_msg, "content") and last_msg.content:
                            data = {"type": "content", "content": last_msg.content}
                            yield f"data: {json.dumps(data)}\n\n"
                            has_sent_content = True
                            
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

@app.get("/health")
@app.get("/")
async def root():
    return {"status": "ok", "version": "bf68de4-with-debug-yields"}

@app.on_event("startup")
async def start_self_ping():
    async def ping_loop():
        # Let the app start up first
        await asyncio.sleep(60)
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    await client.get("http://127.0.0.1:8000/health")
                except Exception:
                    pass
                await asyncio.sleep(14 * 60) # Ping every 14 minutes
    asyncio.create_task(ping_loop())

# SSE - server-sent events 