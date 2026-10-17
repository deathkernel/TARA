# Containerized validation image. This is intentionally not advertised as the
# TARA chat runtime until a tested application entrypoint is available.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN groupadd --system tara && useradd --system --gid tara --home-dir /home/tara tara

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=tara:tara . .
USER tara

CMD ["python", "-m", "pytest", "-q"]
