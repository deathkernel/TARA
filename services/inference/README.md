# TARA Inference Service

A small FastAPI HTTP shell for future model/inference integration. This milestone only implements process health; it intentionally does not import the existing cognitive brain, instantiate a provider, or claim that model inference is available.

## Run locally

From the repository root:

```bash
python -m venv .venv-inference
# Activate the environment, then:
python -m pip install -r services/inference/requirements.txt
uvicorn app:app --app-dir services/inference --host 127.0.0.1 --port 8000
```

- `GET /health/live` — process liveness, HTTP 200.
- `GET /health` — compatibility liveness endpoint, HTTP 200.
- `GET /health/ready` — HTTP 503 until the inference adapter is implemented.

## Tests

```bash
python -m pytest services/inference/tests -q
```

The next milestone will define the provider contract and connect this service to TARA's existing Python architecture. Until then, readiness intentionally remains false.
