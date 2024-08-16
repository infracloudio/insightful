FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir langchain_huggingface==0.0.3
