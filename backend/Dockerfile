# Backend image, deployed as a Render web service (render.yaml).
#
# Build from the repo root:  docker build -f backend/Dockerfile .
# .dockerignore limits the context to requirements.txt and backend/.
# Secrets (DATABASE_URL, SUPABASE_*, LOCAL_LLM_*, cloud keys) come from the
# service's environment at runtime; see DEPLOY.md.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app backend ./backend
USER app

# Online defaults; any of them can be overridden by the service's environment.
# Render sets PORT itself. MALLOC_ARENA_MAX keeps glibc from reserving an arena
# per thread, which matters on a 512 MB instance.
ENV PORT=10000 \
    MALLOC_ARENA_MAX=2 \
    AUTH_REQUIRED=true \
    STORAGE_BACKEND=supabase \
    STORAGE_CACHE_DIR=/tmp/sst_cache \
    LLM_MODE=local_first \
    LOCAL_LLM_EXECUTION_MODEL=qwen3.5-27b \
    LOCAL_LLM_KNOWLEDGE_MODEL=qwen3.5-27b \
    LOCAL_LLM_ROUTER_MODEL=qwen3.5-27b \
    LOCAL_LLM_THINKING=1

EXPOSE 10000

# One worker: the chat rate limiter and the local-LLM circuit breaker live in memory.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
