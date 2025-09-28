# 多阶段构建: Stage 1 - 构建前端
FROM node:18-alpine AS frontend-builder

WORKDIR /frontend

# 复制前端源代码
COPY frontend/ui/package*.json ./
RUN npm install

COPY frontend/ui/ ./
RUN npm run build

# Stage 2 - 主应用镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置非交互模式，避免安装过程中的提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并安装必要工具（包括 Node.js）
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
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
# 安装 Node.js 18.x
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# 安装 Mambaforge
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh -O /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm /tmp/mambaforge.sh
ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu

# 升级 Python 到 3.10
RUN mamba install python=3.10 -y && conda clean -afy

# 提供新的 libstdc++ 以满足 PyArrow 依赖的 GLIBCXX_3.4.32
RUN mamba install -c conda-forge libgcc-ng libstdcxx-ng -y && conda clean -afy

# 设置工作目录
WORKDIR /app

# 安装 CUDA 工具及适用于 CUDA 11.8 的 PyTorch 等
RUN mamba install -c nvidia -c conda-forge cudatoolkit=11.8 -y
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装结构识别工具 decimer-segmentation 和 molscribe
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple decimer-segmentation molscribe

# 安装 OCR 和 AI 依赖
RUN mamba install -c conda-forge jupyter pytesseract transformers -y
RUN pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple PyMuPDF PyPDF2 openai
RUN pip install Levenshtein mdutils google-generativeai tabulate python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mamba install -c conda-forge numpy==1.23.5 -y

# 安装 FastAPI 和 Web 服务依赖
RUN pip install fastapi uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 DECIMER 模型权重
RUN wget -O /opt/conda/lib/python3.10/site-packages/decimer_segmentation/mask_rcnn_molecule.h5 \
    "https://zenodo.org/record/10663579/files/mask_rcnn_molecule.h5?download=1"

# # 国内huggingface镜像
# ENV HF_ENDPOINT=https://hf-mirror.com

# 下载 MolScribe 模型权重
RUN mkdir -p /app/models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir='/app/models', local_files_only=False)"

# 下载 MolNexTR 模型权重
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('CYF200127/MolNexTR', 'molnextr_best.pth', repo_type='dataset', local_dir='/app/models', local_files_only=False)"

# 复制项目文件
COPY pipeline.py /app/pipeline.py
COPY structure_parser.py /app/structure_parser.py
COPY activity_parser.py /app/activity_parser.py
COPY constants.py /app/constants.py
COPY utils /app/utils
COPY data /app/data
COPY bin /app/bin

# 复制后端文件
COPY frontend/backend /app/frontend/backend

# 从构建阶段复制前端构建产物
COPY --from=frontend-builder /frontend/dist /app/frontend/ui/dist

# 安装前端服务工具
RUN npm install -g serve

# 复制前端 package.json (可选，用于文档)
COPY frontend/ui/package*.json /app/frontend/ui/

# Add UID 1000 user and grant permissions
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# 暴露端口
EXPOSE 8000 3000

# 创建启动脚本
RUN echo '#!/bin/bash\n\
# 启动后端 FastAPI 服务\n\
cd /app\n\
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000 &\n\
\n\
# 启动前端服务\n\
cd /app/frontend/ui\n\
npx serve -s dist -l 3000 &\n\
\n\
# 等待所有后台进程\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# 设置默认执行命令
ENTRYPOINT ["/app/start.sh"]
