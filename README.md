# FastChat

FastChat is an intelligent AI assistant powered by Llama-3.1 via Groq and LangGraph, featuring web search capabilities via Tavily, and a modern frontend built with Next.js and Tailwind CSS.

## Project Structure

- `/client` - Next.js frontend
- `/server` - FastAPI backend

## Prerequisites

- Node.js 18+
- Python 3.10+
- Groq API Key
- Tavily API Key

## Getting Started

### 1. Backend Setup

1. Navigate to the server directory:
   ```bash
   cd server
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install fastapi uvicorn python-dotenv langchain-groq langchain-core langgraph langchain-tavily
   ```
4. Configure environment variables in `server/.env`:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
5. Run the server:
   ```bash
   uvicorn app:app --reload
   ```

### 2. Frontend Setup

1. Navigate to the client directory:
   ```bash
   cd client
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Configure environment variables in `client/.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
4. Run the development server:
   ```bash
   npm run dev
   ```

## Production Deployment

### Backend

The backend can be deployed to services like Render, Railway, or Heroku. Ensure you set the `GROQ_API_KEY` and `TAVILY_API_KEY` environment variables. 
Run command in production: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Frontend

The frontend is a Next.js application optimized for Vercel. 
1. Push your repository to GitHub.
2. Import the project in Vercel.
3. Set the `NEXT_PUBLIC_API_URL` environment variable to your deployed backend URL.
4. Deploy!
