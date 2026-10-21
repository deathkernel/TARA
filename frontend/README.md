# TARA Browser Frontend

React + TypeScript + Vite frontend integrated into this repository. It calls
TARA's local Python adapter and does not simulate assistant messages.

## Build and run
Requires Node.js 20+ and Python dependencies already installed from
`requirements.txt`. From the repository root:

```bat
cd frontend
npm install
npm test
npm run build
cd ..
python -m src.web_app
```

Open http://127.0.0.1:8000. The built UI is served by the same Python process
that routes chat through `TARABrain` and the configured local Ollama provider.

For frontend development, run the backend in one terminal:
`python -m src.web_app`
Then in another terminal run `cd frontend && npm run dev`; Vite proxies
`/api` to the Python server on port 8000.

Recent chats are stored in browser localStorage only; no server-side history or
account sync is implied.
