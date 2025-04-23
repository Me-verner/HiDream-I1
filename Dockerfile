FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod

COPY . .

CMD ["python3", "-u", "handler.py"]
