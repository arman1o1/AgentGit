FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY agentgit.py ./
COPY agentgit_debugger.html ./
COPY tests/ ./tests/

EXPOSE 8765

CMD ["python3", "agentgit.py", "serve", "--port", "8765"]
