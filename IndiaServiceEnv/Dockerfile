FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
RUN apt-get update && apt-get install -y curl \
  && rm -rf /var/lib/apt/lists/*
COPY . .
RUN uv sync --locked
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s \
  --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1
CMD ["uv", "run", "server"]
