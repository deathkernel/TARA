# TARA Integrated Browser Frontend

The React + TypeScript interface lives in `frontend/` in this repository.
The Python server in `src/web_app.py` serves the built files from
`frontend/dist/` and sends chat turns through TARA's existing `TARABrain`
and local Ollama provider.

## Windows setup

From the repository root, ensure Ollama is running and `ollama list` includes
your model. Install Node.js 20+, then run:

```bat
cd frontend
npm install
npm test
npm run build
cd ..
python -m src.web_app
```

Open http://127.0.0.1:8000.

For active frontend development, run the Python server in one terminal and
`cd frontend && npm run dev` in another. Vite proxies `/api` requests to the
Python server at port 8000.

The browser stores up to 30 recent conversations in localStorage. These are not
synced to a server account. The frontend does not claim that model output can
automatically perform arbitrary shell or desktop actions.
