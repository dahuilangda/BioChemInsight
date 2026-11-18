FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir fastapi==0.162.0 uvicorn==0.54.0 httpx==0.28.1

WORKDIR /app
COPY app/server.py /app/server.py

EXPOSE 8015
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8015", "--workers", "1"]
