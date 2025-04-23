FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
WORKDIR /app
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ninja
RUN pip install -U flash-attn --no-build-isolation
RUN pip install runpod
# Copy the handler script
COPY src /app/src
# Start the serverless worker
CMD ["python", "-u", "src/handler.py"]
