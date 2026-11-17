FROM docker.io/library/node:18-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/ui/package*.json ./
RUN npm ci

COPY frontend/ui/ ./
RUN npm run build

FROM docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG HF_ENDPOINT=https://hf-mirror.com
ARG HF_REPO="dahuilangda/BioChemInsight"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        bzip2 \
        libglib2.0-0 \
        libstdc++6 \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh -O /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm /tmp/mambaforge.sh

ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

ENV HF_ENDPOINT=${HF_ENDPOINT}

WORKDIR /app

RUN mamba install -c conda-forge \
        python=3.12 \
        libgcc-ng \
        libstdcxx-ng \
        numpy==1.26.4 \
        pandas \
        pillow \
        matplotlib \
        scikit-image \
        scipy \
        h5py \
        opencv=4.* \
        tqdm \
        rdkit \
        lxml \
        html5lib \
        beautifulsoup4 \
        jupyter \
        pytesseract \
        requests \
        transformers \
        huggingface_hub \
    -y && conda clean -afy

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 && \
    python -c "import torch, torchvision; assert torch.__version__ == '2.11.0+cu128', torch.__version__; assert torchvision.__version__ == '0.26.0+cu128', torchvision.__version__"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        PyMuPDF \
        PyPDF2 \
        openai \
        Levenshtein \
        mdutils \
        tabulate \
        python-multipart \
        fastapi \
        uvicorn \
        celery \
        redis \
        SmilesPE \
        qudida \
        albumentations==1.0.3 \
        timm==0.5.4 \
        ultralytics==8.4.199 \
        py2opsin && \
    python -c "import numpy, qudida, cv2; assert numpy.__version__ == '1.26.4', numpy.__version__; assert cv2.__version__.startswith('4.'), cv2.__version__"

# Download base model + MoE weights from HuggingFace
RUN mkdir -p /app/models /app/experiments/moe/production && \
    curl --fail --location --retry 3 --retry-delay 5 --show-error \
        -o /app/models/molnextr_best.pth \
        "${HF_ENDPOINT}/datasets/${HF_REPO}/resolve/main/molnextr_best.pth" && \
    for f in moe_encoder.pth moe_expert1.pth moe_expert2.pth moe_router.pt moe_confidence.pt moe_config.json; do \
        curl --fail --location --retry 3 --retry-delay 5 --show-error \
            -o /app/experiments/moe/production/$f \
            "${HF_ENDPOINT}/datasets/${HF_REPO}/resolve/main/moe/$f"; \
    done

COPY pipeline.py /app/pipeline.py
COPY structure_parser.py /app/structure_parser.py
COPY activity_parser.py /app/activity_parser.py
COPY constants.py /app/constants.py
COPY utils /app/utils
COPY model_skills /app/model_skills
COPY skills /app/skills
COPY services /app/services
COPY packages /app/packages
COPY data /app/data
COPY bin /app/bin
RUN python -c "from utils.MolNexTR import data_aug; print('albumentations data_aug import ok')" && \
    python -c "import ultralytics; print('detector runtime import ok')" && \
    python -c "import hashlib,sys; assert 'do_POST' in open('/app/docker/static-server.py').read() if __import__('os').path.exists('/app/docker/static-server.py') else True; print('static-server proxy ok')" 

COPY frontend/backend /app/frontend/backend

COPY --from=frontend-builder /frontend/dist /app/frontend/ui/dist

COPY frontend/ui/package*.json /app/frontend/ui/

RUN mkdir -p /app/frontend/backend/data /app/output

# PP-OCR venv for text-line OCR (grid reconstruction skill). paddle 2.6
# pinned: the 3.x CPU executor hits a PIR/oneDNN defect on these models;
# the venv isolates its numpy<2 from the main conda env.
RUN python -m venv /opt/ppocr_venv && \
    /opt/ppocr_venv/bin/pip install --no-cache-dir setuptools && \
    /opt/ppocr_venv/bin/pip install --no-cache-dir \
        "paddlepaddle==2.6.2" "paddleocr==2.7.3" "numpy<2" && \
    /opt/ppocr_venv/bin/python -c "import scipy.ndimage, skimage.morphology; import paddleocr, paddle; print('ppocr venv ok', paddle.__version__)"

ENV PYTHONPATH=/app

EXPOSE 8000 3000

COPY docker/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY docker/start-web.sh /app/start-web.sh
COPY docker/static-server.py /app/static-server.py
RUN chmod +x /app/docker-entrypoint.sh /app/start-web.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/start-web.sh"]
