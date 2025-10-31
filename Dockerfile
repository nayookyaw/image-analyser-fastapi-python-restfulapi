
# Multi-stage image for small final size
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for Pillow/opencv if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# If using uv locally, can vendor a requirements.txt for container builds.
# For the setup, we install minimal runtime deps directly.
RUN pip install --no-cache-dir fastapi uvicorn pydantic Pillow onnxruntime yaml

EXPOSE 8000
CMD ["uvicorn", "image_analyser_backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
