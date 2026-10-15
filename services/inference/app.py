"""Health-only HTTP shell for TARA's future inference adapter.

This service deliberately does not import the full TARA brain or call an LLM yet.
The health endpoints make deployment/orchestration checks possible before inference
wiring is introduced in the next roadmap milestone.
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="TARA Inference Service",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    """Return success while the HTTP process can serve requests."""
    return {"status": "ok", "service": "tara-inference"}


@app.get("/health/ready", tags=["health"])
def readiness() -> JSONResponse:
    """Fail closed until the real inference adapter is wired in."""
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "service": "tara-inference",
            "reason": "inference adapter is not connected yet",
        },
    )


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Compatibility health route for simple container probes."""
    return liveness()
