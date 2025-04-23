FROM runpod/base:0.6.2-cuda12.1.0

WORKDIR /app

# Install your dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod

# Copy all your source code
COPY . .

# Start the RunPod handler (which you created in handler.py)
CMD ["python", "-u", "handler.py"]
