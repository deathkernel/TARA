# TARA Chat MVP — API Contract Proposal

**Status:** Proposed contract; not implemented and not evidence of current API support.  
**Goal:** Define the smallest server boundary needed for the web chat MVP without changing TARA's existing reasoning modules unnecessarily.

## Repository findings that affect this contract

- README documents the local chat entry command python -m src and says Ollama is the normal inference provider, with configured alternatives.
- requirements.txt currently lists Python packages but no FastAPI, Flask, Starlette, or other obvious HTTP server framework.
- src/agent.py is a cognitive loop, not a web server. Its docstring explicitly says it does not execute PC actions; results are supplied externally.
- src/integration.py, src/model_runtime.py, and src/groq_provider.py are relevant existing modules to inspect before adapting runtime calls.
- No frontend app/package manifest was found in the inspected repository tree.

Consequently, all endpoints below are proposals. They must be reconciled with the real runtime function signatures and tests before implementation.

## Design principle

Add a thin HTTP adapter that calls the supported runtime entry point. Do not duplicate reasoning, planning, memory, provider, or tool logic inside the frontend or a second chat engine. Keep provider secrets on the server. Never let model-generated text authorize tool execution.

## Proposed endpoints

### GET /api/health
Purpose: determine whether the HTTP adapter is alive and whether it can report a verified runtime/provider status.

Example response:
```json
{
  "status": "ok",
  "runtime": "available",
  "provider": {
    "configured": true,
    "reachable": true
  }
}
```

- Only return fields that can actually be checked.
- Do not expose API keys, full environment values, local file paths, or sensitive exception text.
- If provider reachability is not tested, omit reachable rather than guessing.

### POST /api/chat
Purpose: submit one user message to the supported TARA chat/inference path.

Proposed request:
```json
{
  "message": "Help me reason through this problem.",
  "conversation_id": null
}
```

Proposed response (non-streaming MVP):
```json
{
  "message": "A response produced by the configured TARA runtime.",
  "conversation_id": null
}
```

Contract notes:
- message must be a non-empty string after trimming; enforce a configured size limit.
- conversation_id remains optional and must not imply persistence. If no persistence implementation exists, omit it or return null.
- Use appropriate HTTP status codes for invalid input, unavailable provider, rate/size limits, and unexpected server errors.
- The server must map the request to the existing runtime through a tested adapter function. Do not invent a callable such as TARAEngine.chat() until source inspection proves it exists.
- Avoid logging secrets or full private prompts by default.

## Streaming decision

Streaming is **not assumed** for the first version. Start with a non-streaming response if that matches the runtime's current behavior. Adopt Server-Sent Events or another streaming transport only after confirming the provider/runtime yields incremental output and cancellation can be propagated safely.

If streaming is later added, specify event framing, completion/error events, disconnect behavior, timeout behavior, and request cancellation tests before exposing a Stop button.

## Conversation storage decision

Persistent conversation endpoints are deferred. The repository documents memory primitives, but that does not establish a user-facing conversation store, session ownership model, retention policy, or chat-history API.

Do not add GET /api/conversations or GET /api/conversations/{id} until a storage design exists with:
- ownership/session rules,
- schema and migration plan,
- retention/deletion behavior,
- concurrency and failure behavior,
- tests proving read/write behavior.

## Security and reliability requirements

- Keep provider keys server-side and load them through the project's supported configuration mechanism.
- Set request and response size limits and timeouts.
- Validate JSON and reject malformed or empty messages.
- Render response content safely in the browser; Markdown must not execute raw HTML.
- Treat external tool outputs and model outputs as untrusted content.
- Add CORS restrictions appropriate to the deployment; do not use wildcard origins with credentials.
- Do not expose stack traces or environment variables to clients.
- Tool execution remains governed by TARA's explicit authorization boundary, never by arbitrary text from a model response.

## Required implementation checks

1. Inspect src/integration.py, src/model_runtime.py, src/groq_provider.py, and the actual python -m src entry path.
2. Identify a callable path that accepts a user message and returns the user-facing assistant response.
3. Add unit tests around the adapter using mocked provider/runtime responses.
4. Add HTTP-level tests for valid input, invalid input, provider failure, timeout, and safe error payloads.
5. Run the existing pytest suite and the new adapter tests.
6. Update this contract with actual request/response shapes before declaring it implemented.

## Status checklist

- [x] Proposed minimal chat endpoint documented.
- [x] Health/status reporting constraints documented.
- [x] Streaming explicitly deferred pending evidence.
- [x] Persistence explicitly deferred pending a storage design.
- [ ] Runtime call path verified against source and tests.
- [ ] HTTP adapter implemented.
- [ ] HTTP adapter tests pass.
