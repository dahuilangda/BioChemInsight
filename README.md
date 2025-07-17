# BioChemInsight 🧪

BioChemInsight platform integrates chemical structure recognition with bioactivity analysis to automatically extract relationships between chemical structures and biological activities from literature. This platform efficiently generates structured data, significantly reducing the cost and complexity of data curation and accelerating the construction of high-quality datasets. The tool not only provides rich training data for machine learning and deep learning models but also lays the foundation for drug optimization and target screening research tasks.

## Features 🎉

* **Efficient Identification and Parsing** 🔍: Automatically extracts compound structures and biological activity data from PDF documents.
* **Deep Learning and OCR Technologies** 🧠: Utilizes DECIMER Segmentation models and PaddleOCR for image and text data processing.
* **Flexible Recognition Engines** ⚙️: Switch seamlessly between **MolScribe**, **MolVec**, and **MolNexTR** for SMILES generation.
* **Data Transformation** 🛠️: Converts unstructured data into structured formats for further analysis and application.

## How It Works 🔧

BioChemInsight combines advanced image recognition and text extraction techniques to streamline the data extraction process. It includes two main modules:

### 1. **Chemical Structure Recognition**

* Detects and extracts compound structure images from PDFs using image segmentation models.
* Converts detected chemical structures into **SMILES** format using the selected engine (MolScribe, MolVec, or MolNexTR) and associates them with compound identifiers.

### 2. **Bioactivity Data Recognition**

* Extracts bioactivity data such as IC50, EC50, Ki, and other experimental results from PDFs using OCR.
* Enhances data extraction and transformation using advanced language models for consistency and accuracy.

## Workflow 🚀

1. **PDF Segmentation and Image Conversion**
   Splits PDF documents into pages and converts them into image formats for processing.

2. **Chemical Structure Detection and Conversion**
   Locates compound images using **DECIMER Segmentation** and parses structures into **SMILES** format with your chosen engine (MolScribe, MolVec, or MolNexTR).

3. **Compound Identifier Recognition**
   Recognizes compound numbers using the **visual model** for robust detection and pairing.

4. **Bioactivity Data Extraction and Parsing**
   Extracts bioactivity results using **PaddleOCR** and refines data with large language models.

5. **Data Integration**
   Merges all extracted chemical and bioactivity data into structured formats such as CSV or Excel for downstream analysis.

## Installation 🔧

### Step 1: Clone the Repository

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
```

### Step 2: Configure Constants

The project uses a configuration file named `constants.py` for environment-specific variables. A template file `constants_example.py` is provided. Rename and configure:

```bash
mv constants_example.py constants.py
```

Edit `constants.py` to set your model names, API keys, and paths.

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

## Usage Example 📖

Run the **BioChemInsight** pipeline to extract both chemical structures and bioactivity data:

```bash
python pipeline.py data/sample.pdf \
    --structure-start-page 242 \
    --structure-end-page 267 \
    --assay-start-page 270 \
    --assay-end-page 272 \
    --assay-name "FRET EC50" \
    --engine molnextr \
    --output output
```

### Flexible Options:

* **Engine Switch**: `--engine {molscribe,molvec,molnextr}` to select recognition backend.
* Extract **only chemical structures**:

  ```bash
  python pipeline.py data/sample.pdf \
      --structure-start-page 242 \
      --structure-end-page 267 \
      --engine molnextr \
      --output output
  ```
* Extract **multiple assays**:

  ```bash
  python pipeline.py data/sample.pdf \
      --assay-start-page 30 270 \
      --assay-end-page 40 272 \
      --assay-names "IC50,Ki" \
      --engine molnextr \
      --output output
  ```

## Output 📂

The platform generates structured data files, including:

* **structures.csv**: Contains detected compound IDs and their SMILES representations.
* **assay\_data.json**: Stores extracted bioactivity data for each assay.
* **merged.csv**: Combines chemical structures with bioactivity data into a single file.

## Upcoming in Next Release 📌

* **Batch/Parallel Compound–ID Recognition**: Support for processing large batches in parallel to speed up SMILES matching.

## Applications 🌟

* **AI/ML Model Training**: Supplies high-quality training datasets for cheminformatics and bioinformatics.
* **Drug Discovery**: Supports structure–activity relationship studies for lead optimization.
* **Literature Mining**: Automates data extraction from scientific articles, reducing manual curation.

## Docker Deployment 🐳

### Build Image

```bash
docker build -t biocheminsight .
```

### Run Container

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
    --engine molnextr \
    --output output
```

### Enter the Docker Environment for Testing

```bash
docker run --gpus all -it --rm \
  --entrypoint /bin/bash \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  --name biocheminsight_container \
  biocheminsight
```
