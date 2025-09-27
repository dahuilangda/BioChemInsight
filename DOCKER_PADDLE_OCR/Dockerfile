FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        wget \
        git \
        ccache \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python -m pip install --upgrade pip

# Install PaddleOCR and runtime dependencies
RUN pip install --no-cache-dir \
        paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    pip install --no-cache-dir \
        paddleocr==3.2.0 \
        paddlenlp>=2.7.2 \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        PyMuPDF \
        numpy \
        opencv-python-headless \
        pillow 

WORKDIR /workspace

COPY app /workspace/app

EXPOSE 8010

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
