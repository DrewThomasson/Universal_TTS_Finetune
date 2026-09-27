FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies for TTS and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install python dependencies
ARG UFT_CPU_TORCH=0
RUN if [ "$UFT_CPU_TORCH" = "1" ]; then \
      uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.11.0+cpu' 'torchaudio==2.11.0+cpu' 'torchcodec==0.11.1+cpu'; \
    fi
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy the application files
COPY . .

# Ensure models directory exists
RUN mkdir -p /app/models

# Expose the Gradio port
EXPOSE 7862

# Run the Gradio demo by default
CMD ["python", "web_gui.py", "--host", "0.0.0.0", "--port", "7862", "--out_path", "/app/finetune_models"]
