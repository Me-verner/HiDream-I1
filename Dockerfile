FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ninja
RUN pip install -U flash-attn --no-build-isolation
RUN pip install runpod
COPY src /app/src
CMD ["python", "-u", "src/handler.py"]
