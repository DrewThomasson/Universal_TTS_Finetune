FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for TTS and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Ensure models directory exists
RUN mkdir -p /app/models

# Expose the Gradio port
EXPOSE 5003

# Run the Gradio demo by default
CMD ["python", "xtts_demo.py", "--port", "5003", "--out_path", "/app/finetune_models"]
