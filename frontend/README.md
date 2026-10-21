# TARA Web App

The React + TypeScript frontend lives inside the TARA repository. It calls the local FastAPI adapter at /api; the adapter passes messages into TARABrain and uses Ollama for inference.

## Quick start (Windows)

From the repository root, run run_tara_web.bat, then open http://127.0.0.1:5173. The launcher opens separate terminal windows for the Python API and Vite server.

## Manual start

1. Install Python dependencies from the repository root: python -m pip install -r requirements.txt
2. Start Ollama and install the default model if needed: ollama pull llama3.2:3b
3. Terminal 1, repository root: python -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000
4. Terminal 2: cd frontend, then npm install, then npm run dev.
5. Open http://127.0.0.1:5173 and keep both terminals open.

## Configuration

TARA_OLLAMA_URL defaults to http://127.0.0.1:11434. TARA_OLLAMA_MODEL defaults to llama3.2:3b. Conversation history is saved in browser local storage. The Python brain processes prompts through its runtime memory/context. This first integration uses non-streaming inference.

## Stack

React 18 + TypeScript + Vite, Tailwind CSS 3, FastAPI + Pydantic, existing TARA TARABrain, and local Ollama inference.
