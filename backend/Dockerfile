# Backend image for a Hugging Face Docker Space.
#
# Build context is the Space repo that .github/workflows/deploy-backend.yml
# assembles: this file as ./Dockerfile, requirements.txt and backend/.
# Secrets (DATABASE_URL, SUPABASE_*, LOCAL_LLM_*, cloud keys) come from the
# Space settings at runtime; see DEPLOY.md.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app backend ./backend
USER app

# Online defaults; any of them can be overridden by a Space variable.
ENV PORT=7860 \
    AUTH_REQUIRED=true \
    STORAGE_BACKEND=supabase \
    STORAGE_CACHE_DIR=/tmp/sst_cache \
    LLM_MODE=local_first \
    LOCAL_LLM_EXECUTION_MODEL=qwen3.5-27b \
    LOCAL_LLM_KNOWLEDGE_MODEL=qwen3.5-27b \
    LOCAL_LLM_ROUTER_MODEL=qwen3.5-27b \
    LOCAL_LLM_THINKING=1

EXPOSE 7860

# One worker: the chat rate limiter and the local-LLM circuit breaker live in memory.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
