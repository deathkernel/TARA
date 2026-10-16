# TARA Inference Service

FastAPI adapter that streams responses from the configured Groq model to the Next.js server. The browser never receives the provider API key. The service accepts inference requests only when a shared internal bearer token is configured.

## Run locally

From the repository root:

```bash
python -m venv .venv-inference
# Activate the environment, then:
python -m pip install -r services/inference/requirements.txt
export GROQ_API_KEY="your-groq-api-key"
export TARA_INFERENCE_API_TOKEN="a-long-random-secret"
export TARA_GROQ_MODEL="openai/gpt-oss-20b"
uvicorn services.inference.app:app --host 127.0.0.1 --port 8000
```

Configure the same `TARA_INFERENCE_API_TOKEN` and `TARA_INFERENCE_URL=http://127.0.0.1:8000` in the Next.js server environment. Never expose either secret through a `NEXT_PUBLIC_*` variable or send the provider key to the browser.

## Endpoints

- `GET /health/live` — process liveness, HTTP 200.
- `GET /health` — compatibility liveness endpoint, HTTP 200.
- `GET /health/ready` — HTTP 200 when the provider key and internal API token are configured; otherwise HTTP 503. This is configuration readiness, not a live upstream probe.
- `POST /v1/chat/stream` — authenticated server-to-server SSE stream.

## Safety and limits

- Provider URL is fixed to Groq's API; user input cannot choose a host or model.
- Request content is limited to 12,000 characters; completion tokens and daily process-local request/token budgets are configurable with `TARA_GROQ_MAX_COMPLETION_TOKENS`, `TARA_GROQ_MAX_DAILY_REQUESTS`, and `TARA_GROQ_MAX_DAILY_TOKENS`.
- Upstream response bodies and secrets are not forwarded in errors.
- Usage budgets are process-local, not shared across multiple workers or replicas. For multi-worker deployments, move quota accounting to a shared store before relying on the limits as account-wide hard caps.
- The endpoint does not persist prompts or responses. This first inference milestone does not yet add authenticated user sessions or database-backed chat history.

## Tests

```bash
python -m pytest services/inference/tests -q
```
