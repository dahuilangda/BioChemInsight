# BioChemInsight 🧪

**BioChemInsight** 是一个强大的平台，可以自动从科学文献中提取化学结构及其对应的生物活性数据。通过利用深度学习进行图像识别和OCR，它简化了为化学信息学、机器学习和药物发现研究创建高质量结构化数据集的流程。

![logo](images/BioChemInsight.jpeg)

## 功能特性 🎉

  * **自动化数据提取** 🔍: 自动从 PDF 文档中识别并提取化合物结构和生物活性数据（例如 IC50, EC50, Ki）。
  * **先进识别核心** 🧠: 采用顶尖的 DECIMER Segmentation 模型进行图像分析，并使用 PaddleOCR 进行稳健的文本识别。
  * **dots_ocr OCR 引擎** 🆕: 若需显著提升 OCR 效果，可选择 `dots_ocr` 作为 OCR 引擎。配置方法请参考 `DOCKER_DOTS_OCR/README.md`。注意：在 RTX 5090 显卡上运行 `dots_ocr` 约需 30GB 显存。
  * **推荐视觉模型**: 视觉模型推荐使用 **GLM-V4.5** 或 **MiniCPM-V-4**，效果最佳。
  * **多种 SMILES 引擎** ⚙️: 支持在 **MolScribe**、**MolVec** 和 **MolNexTR** 之间无缝切换，将化学图谱转换为 SMILES 字符串。
  * **灵活的页面选择** 📄: 可处理特定的、非连续的页面（例如 "1-3, 5, 7-9, 12"），节省时间和计算资源。
  * **结构化数据输出** 🛠️: 将非结构化的文本和图像转换为可直接用于分析的格式，如 CSV 和 Excel。
  * **现代化 Web UI** 🌐: 基于 React 的前端界面配合 FastAPI 后端，提供直观的 PDF 处理、实时进度跟踪和交互式结果可视化。
  * **智能数据合并** 🔗: 基于化合物 ID 自动合并结构和生物活性数据，提供无缝的集成结果。


## 应用场景 🌟

  * **AI/ML 模型训练**: 为化学信息学和生物信息学领域的预测模型生成高质量的训练数据集。
  * **药物发现**: 加速构效关系（SAR）研究和先导化合物的优化过程。
  * **自动化文献挖掘**: 大幅减少从科学文章中手动整理数据所需的人力和时间。


## 工作流程 🚀

BioChemInsight 采用多阶段流水线将原始 PDF 转换为结构化数据：

1.  **PDF 预处理**: 将输入的 PDF 拆分为单个页面，然后将这些页面转换为高分辨率图像以供分析。
2.  **结构检测**: **DECIMER Segmentation** 扫描图像，以定位和分离化学结构图。
3.  **SMILES 转换**: 选定的识别引擎（**MolScribe**、**MolVec** 或 **MolNexTR**）将分离出的图谱转换为机器可读的 SMILES 字符串。
4.  **标识符识别**: 视觉模型（推荐：**GLM-V4.5**）识别与每个结构相关的化合物标识符（例如，“化合物 **1**”、“**2a**”）。
5.  **生物活性提取**: **PaddleOCR**（如已配置也可用 `dots_ocr`）从指定的实验页面提取文本，大型语言模型则辅助解析和标准化生物活性结果。
6.  **数据整合**: 所有提取的信息——化合物ID、SMILES 字符串和生物活性数据——被合并到结构化文件（CSV/Excel）中，以便下载和进行下游分析。


## 安装 🔧

#### 步骤 1: 克隆代码仓库

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
```

#### 步骤 2: 配置常量

项目需要一个 `constants.py` 文件来设置环境变量和路径。我们提供了一个模板文件。

```bash
# 重命名示例文件
mv constants_example.py constants.py
```

然后，编辑 `constants.py` 文件，设置您的 API 密钥、模型路径和其他必要配置。

#### 步骤 3: 创建并激活 Conda 环境

```bash
conda install -c conda-forge mamba
mamba create -n chem_ocr python=3.10
conda activate chem_ocr
```

#### 步骤 4: 安装依赖

首先，安装支持 CUDA 的 PyTorch。

```bash
# 安装 CUDA 工具和 PyTorch
mamba install -c conda-forge -c nvidia cuda-tools==11.8
mamba install pytorch==2.0.0 torchvision==0.15.0 torchio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

接下来，安装其余的 Python 包。

```bash
# 安装核心库和 OCR 工具 (使用镜像源以加快下载速度)
pip install decimer-segmentation molscribe -i https://pypi.tuna.tsinghua.edu.cn/simple
mamba install -c conda-forge jupyter pytesseract transformers
pip install paddleocr paddlepaddle-gpu PyMuPDF PyPDF2 fitz -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Web 服务依赖
pip install fastapi uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Node.js 和 npm (用于前端)
# 在 Ubuntu/Debian 上:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# 在 macOS 上 (使用 homebrew):
# brew install node
```


## 使用方法 📖

BioChemInsight 可以通过交互式网页界面或直接从命令行操作。

### 网页界面 🌐

基于 React 的现代化网页界面提供了直观的文档处理平台，具备实时进度跟踪功能。

#### 启动 Web 服务

**步骤 1: 启动后端 API 服务器**

在项目根目录下运行：

```bash
cd frontend/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**步骤 2: 启动前端开发服务器**

在新的终端中运行：

```bash
cd frontend/ui
npm install
npm run dev
```

**步骤 3: 访问界面**

在网页浏览器中打开 `http://localhost:5173` 访问界面。后端 API 地址为 `http://localhost:8000`。

#### 使用开发脚本快速启动

为了方便使用，您可以使用提供的启动脚本同时启动两个服务：

```bash
./start_dev.sh
```

该脚本将：
- 检查必需的依赖项
- 在需要时安装前端包
- 启动后端（端口 8000）和前端（端口 5173）服务
- 提供 Ctrl+C 便捷清理

#### 网页界面功能

1.  **PDF 上传**: 通过直观界面上传和管理 PDF 文件。
2.  **可视化页面选择**: 点击页面缩略图选择结构和活性提取页面。
3.  **分步处理流程**: 
    - **步骤 1**: 上传 PDF 并预览页面
    - **步骤 2**: 实时进度显示的化学结构提取
    - **步骤 3**: 基于结构约束化合物匹配的生物活性数据提取
    - **步骤 4**: 查看和下载合并结果
4.  **实时进度跟踪**: 详细状态更新的提取进度监控。
5.  **交互式结果**: 查看、编辑和下载结构化数据，集成化合物-活性匹配。
6.  **自动数据合并**: 基于化合物 ID 无缝结合结构和生物活性数据。


#### 网页界面功能

1.  **PDF 上传**: 上传您希望处理的 PDF 文件。
2.  **页面选择**: 通过点击缩略图或输入页面范围，直观地选择用于结构和实验数据提取的页面。
3.  **结构提取**: 点击 **"第一步: 提取结构"** 开始化学结构识别。
4.  **活性提取**: 输入实验方法的名称，选择相关页面，然后点击 **"第二步: 提取活性"**。
5.  **下载结果**: 在界面底部预览和下载结构化的数据表格。

### 命令行界面 (CLI)

对于批处理和自动化任务，推荐使用命令行界面。

#### 增强语法 (推荐)

新的语法支持灵活的、非连续的页面选择。

```bash
python pipeline.py data/sample.pdf \
    --structure-pages "242-267" \
    --assay-pages "270-272" \
    --assay-names "FRET EC50" \
    --engine molnextr \
    --output output
```

**灵活页面选择示例:**

  * **从非连续页面中提取结构:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-250,255,260-267" --engine molnextr --output output
    ```
  * **从分散的页面中提取多种实验数据:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-267" --assay-pages "30,35,270-272" --assay-names "IC50,FRET EC50" --engine molnextr --output output
    ```

#### 旧版语法 (仍然支持)

为了向后兼容，原始的起始/结束页面语法仍然可用。

```bash
python pipeline.py data/sample.pdf \
    --structure-start-page 242 \
    --structure-end-page 267 \
    --assay-start-page 270 \
    --assay-end-page 272 \
    --assay-names "FRET EC50" \
    --engine molnextr \
    --output output
```


## 输出 📂

平台将在指定的输出目录中生成以下结构化数据文件：

  * `structures.csv`: 包含检测到的化合物标识符及其对应的 SMILES 表示。
  * `assay_data.json`: 存储为每个指定实验提取的原始生物活性数据。
  * `merged.csv`: 一个合并文件，将化学结构与其相关的生物活性数据整合在一起。


## Docker 部署 🐳

在容器化环境中部署 BioChemInsight 以确保一致性和可移植性。

#### 步骤 1: 构建 Docker 镜像

```bash
docker build -t biocheminsight .
```

#### 步骤 2: 运行服务

**选项 A: 启动 Web 应用 (默认)**

运行此命令以启动 React 前端和 FastAPI 后端服务。

```bash
docker run --rm -d --gpus all \
    -p 3000:3000 -p 8000:8000 \
    -e http_proxy="" \
    -e https_proxy="" \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/constants.py:/app/constants.py \
    biocheminsight
```

启动后，访问：
- 前端界面：`http://localhost:3000`
- 后端 API：`http://localhost:8000`

**选项 B: 运行命令行流程**

如果您需要使用命令行执行批处理任务，请在 `docker run` 命令后指定 `python pipeline.py` 和相关参数来覆盖默认入口点。

```bash
docker run --rm --gpus all \
    -e http_proxy="" \
    -e https_proxy="" \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    --entrypoint python \
    biocheminsight \
    pipeline.py data/sample.pdf \
    --structure-pages "242-267" \
    --assay-pages "270-272" \
    --assay-names "FRET EC50" \
    --engine molnextr \
    --output output
```

**选项 C: 进入容器进行交互式会话**

如果需要在容器内进行调试或手动运行命令：

```bash
docker run --gpus all -it --rm \
  --entrypoint /bin/bash \
  -e http_proxy="" \
  -e https_proxy="" \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  --name biocheminsight_container \
  biocheminsight
```


## 路线图 📌

  * **并行处理**: 实现化合物-ID对的批量/并行识别，以显著加快大型文档的处理速度。