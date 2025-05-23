# 使用 NVIDIA CUDA 11.8 cuDNN 开发版 Ubuntu20.04 镜像作为基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 设置非交互模式，避免安装过程中的提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并安装必要工具
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 安装 Mambaforge (包含 conda 和 mamba)
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh -O /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm /tmp/mambaforge.sh
ENV PATH=/opt/conda/bin:$PATH

# 升级 Python 到 3.10
RUN mamba install python=3.10 -y && conda clean -afy

# 设置工作目录
WORKDIR /app

# 安装 CUDA 工具及适用于 CUDA 11.8 的 PyTorch 等
RUN mamba install -c nvidia -c conda-forge cudatoolkit=11.8 -y
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装结构识别工具 decimer-segmentation 和 molscribe
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple decimer-segmentation molscribe

# 安装 OCR 和 AI 依赖
RUN mamba install -c conda-forge jupyter pytesseract transformers -y
RUN pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple paddleocr==2.8.1 PyMuPDF PyPDF2 openai
RUN pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
RUN pip install Levenshtein mdutils google-generativeai tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mamba install -c conda-forge numpy==1.23.5 -y

# 下载 DECIMER 模型权重
RUN wget -O /opt/conda/lib/python3.10/site-packages/decimer_segmentation/mask_rcnn_molecule.h5 \
    "https://zenodo.org/record/10663579/files/mask_rcnn_molecule.h5?download=1"

# 下载 PaddleOCR 模型
RUN mkdir -p /home/appuser/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer && \
    mkdir -p /home/appuser/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer && \
    mkdir -p /home/appuser/.paddleocr/whl/table/en_ppstructure_mobile_v2.0_SLANet_infer && \
    mkdir -p /home/appuser/.paddleocr/whl/layout/picodet_lcnet_x1_0_fgd_layout_infer && \
    mkdir -p /home/appuser/.paddleocr/whl/formula/rec_latex_ocr_infer && \
    wget -O /home/appuser/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar \
         https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && \
    wget -O /home/appuser/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/en_PP-OCRv4_rec_infer.tar \
         https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar && \
    wget -O /home/appuser/.paddleocr/whl/table/en_ppstructure_mobile_v2.0_SLANet_infer/en_ppstructure_mobile_v2.0_SLANet_infer.tar \
         https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/paddle3.0b2/en_ppstructure_mobile_v2.0_SLANet_infer.tar && \
    wget -O /home/appuser/.paddleocr/whl/layout/picodet_lcnet_x1_0_fgd_layout_infer/picodet_lcnet_x1_0_fgd_layout_infer.tar \
         https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar && \
    wget -O /home/appuser/.paddleocr/whl/formula/rec_latex_ocr_infer/rec_latex_ocr_infer.tar \
         https://paddleocr.bj.bcebos.com/contribution/rec_latex_ocr_infer.tar

# 下载 MolScribe 模型权重
RUN mkdir -p /app/models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir='/app/models', local_files_only=False)"

# 复制项目文件
# COPY models /app/models
COPY pipeline.py /app/pipeline.py
COPY structure_parser.py /app/structure_parser.py
COPY activity_parser.py /app/activity_parser.py
COPY constants.py /app/constants.py
COPY utils /app/utils
COPY data /app/data
COPY bin /app/bin

# 添加 UID 为 1000 的用户，确保挂载数据权限一致
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    chown -R 1000:1000 /app && \
    chown -R appuser:appuser /home/appuser/.paddleocr

# 切换为非 root 用户
USER appuser

# 设置默认执行命令
ENTRYPOINT ["python", "pipeline.py"]
