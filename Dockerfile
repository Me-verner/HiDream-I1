FROM runpod/base:0.6.3-cuda12.1.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod

COPY . .

CMD ["python3", "-u", "handler.py"]
