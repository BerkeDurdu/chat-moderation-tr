# Runs the moderation app the same way on Windows, macOS and Linux (CPU only).
#   docker build -t chat-moderation-tr .
#   docker run --rm -p 8501:8501 chat-moderation-tr      -> http://localhost:8501
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
WORKDIR /app

# CPU build of the pinned torch first (the default Linux wheel would pull CUDA), then the rest
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r requirements.txt

COPY src/ src/
COPY app.py download_models.py ./
COPY .streamlit/ .streamlit/

# the trained weights come from the public Hugging Face repo, so the image works offline afterwards
RUN python download_models.py

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
