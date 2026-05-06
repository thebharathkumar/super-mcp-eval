FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Node 20 is required at runtime so the harness can launch reference MCP servers
# such as @modelcontextprotocol/server-filesystem via npx.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY mcp_eval ./mcp_eval
RUN pip install -e ".[dev]"

COPY tests ./tests

EXPOSE 8501

CMD ["streamlit", "run", "mcp_eval/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
