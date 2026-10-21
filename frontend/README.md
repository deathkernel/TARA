# TARA Browser Frontend

React + TypeScript + Vite frontend integrated into this repository. The UI
communicates with TARA's Python server and does not fake assistant responses.

## Run on Windows

From the repository root, install Node.js 20+, then:

```bat
cd frontend
npm install
npm test
npm run build
cd ..
python -m src.web_app
```

Open http://127.0.0.1:8000. The production UI is served by the same Python
server that routes chat through TARA and local Ollama.

For development, run `python -m src.web_app` in one terminal and
`cd frontend && npm run dev` in another. Vite proxies /api to localhost:8000.

Recent chat history is saved in this browser only; it is not account sync.
