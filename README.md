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

### Step 1: Create and Activate the Environment

```bash
mamba create -n chem_ocr python=3.10
conda activate chem_ocr
```

### Step 2: Install Dependencies

#### 2.1 CUDA Tools and PyTorch
```bash
# Install CUDA tools
mamba install -c conda-forge -c nvidia cuda-tools==11.8

# Install PyTorch with CUDA support
mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 2.2 Additional Libraries
```bash
# Install DECIMER and MolScribe for chemical structure processing
pip install decimer-segmentation molscribe -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install OCR and Transformer libraries
mamba install -c conda-forge jupyter pytesseract transformers
pip install paddleocr paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install PDF processing and utility libraries
pip install PyMuPDF PyPDF2 fitz -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install other dependencies
pip install fastapi uvicorn python-multipart openai -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seatable-api==2.6.11 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Levenshtein mdutils -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step 3: Verify Installation

Check that key libraries are correctly installed:

```bash
conda activate chem_ocr
python -c "import torch; print(torch.cuda.is_available())"  # Should return True for GPU support
```

---

## Usage Example

Run the **BioChemInsight** pipeline to extract both chemical structures and bioactivity data:

```bash
python pipeline.py data/sample.pdf \
    --structure-start-page 242 --structure-end-page 250 \
    --assay-start-page 270 --assay-end-page 272 \
    --assay-name "FRET EC50" \
    --output output
```

### Flexible Options:
- Extract **only chemical structures**:
  ```bash
  python pipeline.py data/sample.pdf \
      --structure-start-page 242 --structure-end-page 250 \
      --output output
  ```

- Extract **multiple assays** from different ranges:
  ```bash
  python pipeline.py data/sample.pdf \
      --assay-start-page 30 270 --assay-end-page 40 272 \
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