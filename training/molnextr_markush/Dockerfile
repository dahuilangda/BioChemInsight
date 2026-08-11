FROM biocheminsight

ENV PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/workspace \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

WORKDIR /workspace

COPY training/molnextr_markush/requirements.txt /tmp/molnextr-markush-requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/molnextr-markush-requirements.txt

CMD ["python", "training/molnextr_markush/tools/train_fragment_attachment_expert.py", "--help"]
