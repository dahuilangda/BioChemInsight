# BioChemInsight

BioChemInsight platform integrates chemical structure recognition with bioactivity analysis to automatically extract relationships between chemical structures and biological activities from literature. This platform efficiently generates structured data, significantly reducing the cost and complexity of data curation and accelerating the construction of high-quality datasets. This tool not only provides rich training data for machine learning and deep learning models but also lays the foundation for drug optimization and target screening research tasks.

## Features

- **Efficient Identification and Parsing**: Automatically extracts compound structures and biological activity data from PDF documents.
- **Deep Learning and OCR Technologies**: Utilizes DECIMER Segmentation models and PaddleOCR for image and text data processing.
- **Data Transformation**: Converts unstructured data into structured formats for further analysis and application.

## How It Works

BioChemInsight combines advanced image recognition and text extraction techniques to streamline the data extraction process. It includes two main modules:

### 1. **Chemical Structure Recognition**
   - Detects and extracts compound structure images from PDFs using image segmentation models.
   - Converts detected chemical structures into **SMILES** format and associates them with compound identifiers.

### 2. **Bioactivity Data Recognition**
   - Extracts bioactivity data such as IC50, EC50, Ki, and other experimental results from PDFs using OCR.
   - Enhances data extraction and transformation using advanced language models for consistency and accuracy.

## Workflow

1. **PDF Segmentation and Image Conversion**  
   Splits PDF documents into pages and converts them into image formats for processing.  
   
2. **Chemical Structure Detection and Conversion**  
   Locates compound images using **DECIMER Segmentation** and parses structures into **SMILES** format with **MolScribe** or **MolVec**.

3. **Compound Identifier Recognition**  
   Recognizes compound numbers using the **MiniCPB-V-2.6** model for robust detection and pairing.

4. **Bioactivity Data Extraction and Parsing**  
   Extracts bioactivity results using **PaddleOCR** and refines data with large language models.

5. **Data Integration**  
   Merges all extracted chemical and bioactivity data into structured formats such as CSV or Excel for downstream analysis.

## Installation

To set up **BioChemInsight**, follow the steps below:

### Step 1: Clone the Repository

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
```

### Step 2: Configure Constants

The project uses a configuration file named `constants.py` for environment-specific variables. A template file `constants_example.py` is provided in the repository. To configure your environment:

1. Rename `constants_example.py` to `constants.py`:
   ```bash
   mv constants_example.py constants.py
   ```

2. Open `constants.py` and update the values as per your environment:
   ```python
   GEMINI_MODEL_NAME = 'gemini-1.5-flash'  # GEMINI model name
   GEMINI_API_KEY = 'sk-xxxx'             # API key for the Gemini model
   LLM_MODEL_NAME = 'qwen'           # Secondary model name
   LLM_MODEL_URL = 'http://xxxx:8000/v1'  # URL for the secondary model
   LLM_MODEL_KEY = 'sk-xxxx'         # API key for the secondary model
   VISUAL_MODEL_URL = 'http://xxxx:8000/v1'  # URL for the visual model
   VISUAL_MODEL_KEY = 'sk-xxxx'           # API key for the visual model
   HTTP_PROXY = ''   # HTTP proxy (if needed)
   HTTPS_PROXY = ''  # HTTPS proxy (if needed)
   MOLVEC = '/path/to/BioChemInsight/bin/molvec-0.9.9-SNAPSHOT-jar-with-dependencies.jar'  # Path to MolVec JAR
   ```

3. Save the changes.

### Step 3: Create and Activate the Environment

```bash
conda install -c conda-forge mamba
mamba create -n chem_ocr python=3.10
conda activate chem_ocr
```

### Step 4: Install Dependencies

#### CUDA Tools and PyTorch
```bash
mamba install -c conda-forge -c nvidia cuda-tools==11.8
mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Additional Libraries
```bash
pip install decimer-segmentation molscribe -i https://pypi.tuna.tsinghua.edu.cn/simple
mamba install -c conda-forge jupyter pytesseract transformers
pip install paddleocr paddlepaddle-gpu PyMuPDF PyPDF2 fitz -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## Usage Example

Run the **BioChemInsight** pipeline to extract both chemical structures and bioactivity data:

```bash
python pipeline.py data/sample.pdf \
    --structure-start-page 242 \
    --structure-end-page 267 \
    --assay-start-page 270 \
    --assay-end-page 272 \
    --assay-name "FRET EC50" \
    --output output
```

### Flexible Options:
- Extract **only chemical structures**:
  ```bash
  python pipeline.py data/sample.pdf \
      --structure-start-page 242 \
      --structure-end-page 267 \
      --output output
  ```

- Extract **multiple assays** from different ranges:
  ```bash
  python pipeline.py data/sample.pdf \
      --assay-start-page 30 270 \
      --assay-end-page 40 272 \
      --assay-names "IC50,Ki" \
      --output output
  ```

- Merge structures and assays into a single file:
  After extracting structures and assays, the platform automatically merges the data into a consolidated CSV file.


## Output

The platform generates structured data files, including:
- **structures.csv**: Contains detected compound IDs and their SMILES representations.
- **assay_data.json**: Stores extracted bioactivity data for each assay.
- **merged.csv**: Combines chemical structures with bioactivity data into a single file.


## Applications

- **AI/ML Model Training**: Supplies high-quality training datasets for machine learning and deep learning tasks in cheminformatics and bioinformatics.
- **Drug Discovery**: Supports drug optimization and target screening by providing precise structure-activity relationships.
- **Literature Mining**: Automates the extraction of key data from scientific articles, reducing manual labor.


## Docker Deployment

This section describes how to deploy **BioChemInsight** using Docker for efficient operation in a GPU environment. The following instructions are based on building a Docker image with NVIDIA containers and Mambaforge environment, and configuring the Hugging Face mirror to [hf-mirror.com](https://hf-mirror.com) (for model and data downloads).


### Build Image

Make sure you are in the project root directory (where the Dockerfile is located), and execute the following command to build the Docker image (the image name can be customized, e.g., `biocheminsight`):

```bash
docker build -t biocheminsight .
```


### Run Container

When running the container, it is recommended to mount the data input and output directories, and use the `--gpus all` parameter to enable GPU support. For example:

```bash
docker run --rm --gpus all \
    --network host \
    -e http_proxy="" \
    -e https_proxy="" \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    biocheminsight data/sample.pdf \
    --structure-start-page 242 \
    --structure-end-page 267 \
    --assay-start-page 270 \
    --assay-end-page 272 \
    --assay-names "FRET EC50" \
    --output output
```


# Enter the Docker environment for testing
```bash
docker run --gpus all -it --rm \
  --entrypoint /bin/bash \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  --name biocheminsight_container \
  biocheminsight
```