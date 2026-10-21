# TARA Web Frontend — Initial Scaffold

This directory contains a standalone, dependency-free web UI scaffold for TARA. The current main branch describes TARA as a Python reasoning/runtime project; a browser-facing HTTP API is not assumed here. Until an API adapter is implemented and verified, chat sends a clear local-demo response rather than pretending it called the model.

## Start

From the repository root:

```bash
python -m http.server 4173 --directory frontend
```

Open http://localhost:4173.

## Files

- `index.html` — accessible app shell
- `styles.css` — responsive dark workspace styling
- `app.js` — navigation, chat composer, local conversation UI, theme and settings interactions
- `README.md` — integration boundary and next steps

## Current limitations

This is a front-end scaffold, not a connected inference client. No credentials belong in browser code. The backend must expose a local, documented API (with validation, timeouts, bounded payloads, and origin protections) before the composer can call TARA safely. Keep tool execution and permission checks server-side.
