FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision diffusers transformers accelerate einops runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
