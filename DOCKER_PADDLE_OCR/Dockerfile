FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLEOCR_DEVICE=gpu \
    PADDLEOCR_LANG=auto \
    PADDLEOCR_RENDER_SCALE=1.3 \
    PADDLEOCR_USE_DOC_ORIENTATION=false \
    PADDLEOCR_USE_DOC_UNWARPING=false

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

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip

# Install PaddlePaddle + PaddleOCR runtime dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "paddleocr[doc-parser]==3.5.0" \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        python-docx \
        PyMuPDF \
        numpy \
        opencv-python-headless \
        pillow \
        redis \
        celery

WORKDIR /workspace

COPY app /workspace/app

EXPOSE 8010
