FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi uvicorn httpx

WORKDIR /app
COPY app/server.py /app/server.py

EXPOSE 8015
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8015", "--workers", "1"]
