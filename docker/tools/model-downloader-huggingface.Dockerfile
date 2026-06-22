FROM python:3.11-slim

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir huggingface_hub==1.20.1
