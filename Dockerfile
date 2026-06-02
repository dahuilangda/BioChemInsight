FROM docker.io/library/node:18-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/ui/package*.json ./
RUN npm ci

COPY frontend/ui/ ./
RUN npm run build

FROM docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG ZENODO_HOST=
ARG HF_ENDPOINT=https://hf-mirror.com
ARG DECIMER_WEIGHTS_URL="https://zenodo.org/records/10663579/files/mask_rcnn_molecule.h5?download=1"
ARG MOLNEXTR_REPO="CYF200127/MolNexTR"
ARG MOLNEXTR_FILE="molnextr_best.pth"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        bzip2 \
        libglib2.0-0 \
        libstdc++6 \
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
        opencv \
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
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129 && \
    python -c "import torch, torchvision; assert torch.__version__ == '2.11.0+cu129', torch.__version__; assert torchvision.__version__ == '0.26.0+cu129', torchvision.__version__"

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
        timm==0.5.4 && \
    python -c "import numpy, qudida; assert numpy.__version__ == '1.26.4', numpy.__version__"

RUN mkdir -p /app/models && \
    if [ -n "$ZENODO_HOST" ] && [ "$ZENODO_HOST" != "zenodo.org" ]; then \
        echo "Downloading DECIMER weights with zenodo.org resolved to ${ZENODO_HOST}" && \
        curl --fail --location --retry 3 --retry-delay 5 --show-error --resolve "zenodo.org:443:${ZENODO_HOST}" \
            -o /tmp/mask_rcnn_molecule.h5 \
            "$DECIMER_WEIGHTS_URL"; \
    else \
        echo "Downloading DECIMER weights with normal DNS resolution" && \
        curl --fail --location --retry 3 --retry-delay 5 --show-error \
            -o /tmp/mask_rcnn_molecule.h5 \
            "$DECIMER_WEIGHTS_URL"; \
    fi

RUN curl --fail --location --retry 3 --retry-delay 5 --show-error \
        -o /app/models/${MOLNEXTR_FILE} \
        "${HF_ENDPOINT}/datasets/${MOLNEXTR_REPO}/resolve/main/${MOLNEXTR_FILE}"

COPY scripts/convert_decimer_weights.py /app/scripts/convert_decimer_weights.py
RUN python -c "import sys; sys.path.insert(0,'/app'); from scripts.convert_decimer_weights import convert_weights; convert_weights('/tmp/mask_rcnn_molecule.h5','/app/models/mask_rcnn_molecule.pth')" && \
    rm -f /tmp/mask_rcnn_molecule.h5

COPY pipeline.py /app/pipeline.py
COPY structure_parser.py /app/structure_parser.py
COPY activity_parser.py /app/activity_parser.py
COPY constants.py /app/constants.py
COPY utils /app/utils
COPY model_skills /app/model_skills
COPY data /app/data
COPY bin /app/bin
COPY scripts /app/scripts
RUN python -c "from utils.MolNexTR import data_aug; print('albumentations data_aug import ok')"

COPY frontend/backend /app/frontend/backend

COPY --from=frontend-builder /frontend/dist /app/frontend/ui/dist

COPY frontend/ui/package*.json /app/frontend/ui/

RUN mkdir -p /app/frontend/backend/data /app/output

ENV PYTHONPATH=/app

EXPOSE 8000 3000

COPY docker/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY docker/start-web.sh /app/start-web.sh
COPY docker/static-server.py /app/static-server.py
RUN chmod +x /app/docker-entrypoint.sh /app/start-web.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/start-web.sh"]
