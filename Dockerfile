FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt update && apt install -y \
    python3 python3-pip python-is-python3 git libgl1 unzip wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install -U flash-attn --no-build-isolation && \
    pip install git+https://github.com/huggingface/diffusers.git && \
    pip install fastapi uvicorn[standard] pillow

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]