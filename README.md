# BioChemInsight 🧪

**BioChemInsight** is a powerful platform that automates the extraction of chemical structures and their corresponding bioactivity data from scientific literature. By leveraging deep learning for image recognition and OCR, it streamlines the creation of high-quality, structured datasets for cheminformatics, machine learning, and drug discovery research.

![logo](images/BioChemInsight.jpeg)

## Features 🎉

  * **Automated Data Extraction** 🔍: Automatically identifies and extracts compound structures and biological activity data (e.g., IC50, EC50, Ki) from PDF documents.
  * **Advanced Recognition Core** 🧠: Utilizes state-of-the-art DECIMER Segmentation models for image analysis and PaddleOCR for robust text recognition.
  * **dots_ocr as OCR Engine** 🆕: For significantly improved OCR performance, you can use `dots_ocr` as the OCR engine. Please refer to `DOCKER_DOTS_OCR/README.md` for setup and configuration. Note: Running `dots_ocr` on an RTX 5090 GPU requires approximately 30GB of VRAM.
  * **Recommended Visual Model**: For the visual model, it is recommended to use **GLM-V4.5** or **MiniCPM-V-4** for optimal results.
  * **Multiple SMILES Engines** ⚙️: Offers seamless switching between **MolScribe**, **MolVec**, and **MolNexTR** to convert chemical diagrams into SMILES strings.
  * **Flexible Page Selection** 📄: Process specific, non-continuous pages (e.g., "1-3, 5, 7-9, 12"), saving time and computational resources.
  * **Structured Data Output** 🛠️: Converts unstructured text and images into analysis-ready formats like CSV and Excel.
  * **Interactive Web UI** 🌐: A user-friendly Gradio-based web interface for easy PDF processing, page selection, and result visualization.


## Applications 🌟

  * **AI/ML Model Training**: Generate high-quality datasets for training predictive models in cheminformatics and bioinformatics.
  * **Drug Discovery**: Accelerate structure-activity relationship (SAR) studies and lead compound optimization.
  * **Automated Literature Mining**: Drastically reduce the manual effort and time required for curating data from scientific articles.


## Workflow 🚀

BioChemInsight employs a multi-stage pipeline to convert raw PDFs into structured data:

1.  **PDF Preprocessing**: The input PDF is split into individual pages, which are then converted into high-resolution images for analysis.
2.  **Structure Detection**: **DECIMER Segmentation** scans the images to locate and isolate chemical structure diagrams.
3.  **SMILES Conversion**: The selected recognition engine (**MolScribe**, **MolVec**, or **MolNexTR**) converts the isolated diagrams into machine-readable SMILES strings.
4.  **Identifier Recognition**: A visual model (recommended: **GLM-V4.5** or **MiniCPM-V-4**) recognizes the compound identifiers (e.g., "Compound **1**", "**2a**") associated with each structure.
5.  **Bioactivity Extraction**: **PaddleOCR** (or `dots_ocr` if configured) extracts text from specified assay pages, and large language models help parse and standardize the bioactivity results.
6.  **Data Integration**: All extracted information—compound IDs, SMILES strings, and bioactivity data—is merged into structured files (CSV/Excel) for download and downstream analysis.


## Installation 🔧

#### Step 1: Clone the Repository

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
```

#### Step 2: Configure Constants

The project requires a `constants.py` file for environment variables and paths. A template is provided.

```bash
# Rename the example file
mv constants_example.py constants.py
```

Then, edit `constants.py` to set your API keys, model paths, and other necessary configurations.

#### Step 3: Create and Activate the Conda Environment

```bash
conda install -c conda-forge mamba
mamba create -n chem_ocr python=3.10
conda activate chem_ocr
```

#### Step 4: Install Dependencies

First, install PyTorch with CUDA support.

```bash
# Install CUDA Tools and PyTorch
mamba install -c conda-forge -c nvidia cuda-tools==11.8
mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Next, install the remaining Python packages.

```bash
# Install core libraries and OCR tools (using a mirror for faster downloads)
pip install decimer-segmentation molscribe -i https://pypi.tuna.tsinghua.edu.cn/simple
mamba install -c conda-forge jupyter pytesseract transformers
pip install paddleocr paddlepaddle-gpu PyMuPDF PyPDF2 fitz -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage 📖

BioChemInsight can be operated via an interactive web interface or directly from the command line.

### Web Interface 🌐

The Gradio Web UI provides an easy-to-use graphical interface for processing documents.

#### Launch the Web Service

From the project root directory, run:

```bash
python app.py
```

The service will start on `http://0.0.0.0:7860` by default. Open this address in your web browser to access the interface.

#### Web Interface Features

1.  **PDF Upload**: Upload the PDF file you wish to process.
2.  **Page Selection**: Visually select pages for structure and assay extraction by clicking thumbnails or entering page ranges.
3.  **Structure Extraction**: Click **"Step 1: Extract Structures"** to begin chemical structure recognition.
4.  **Activity Extraction**: Enter the names of the assays, select the relevant pages, and click **"Step 2: Extract Activity"**.
5.  **Download Results**: Preview and download the structured data tables at the bottom of the interface.

### Command-Line Interface (CLI)

For batch processing and automation, the CLI is recommended.

#### Enhanced Syntax (Recommended)

The new syntax supports flexible, non-continuous page selections.

```bash
python pipeline.py data/sample.pdf \
    --structure-pages "242-267" \
    --assay-pages "270-272" \
    --assay-names "FRET EC50" \
    --engine molnextr \
    --output output
```

**Flexible Page Selection Examples:**

  * **Extract structures from non-continuous pages:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-250,255,260-267" --engine molnextr --output output
    ```
  * **Extract multiple assays from scattered pages:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-267" --assay-pages "30,35,270-272" --assay-names "IC50,FRET EC50" --engine molnextr --output output
    ```

#### Legacy Syntax (Still Supported)

For backward compatibility, the original start/end page syntax remains available.

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


## Output 📂

The platform generates the following structured data files in the specified output directory:

  * `structures.csv`: Contains the detected compound identifiers and their corresponding SMILES representations.
  * `assay_data.json`: Stores the raw extracted bioactivity data for each specified assay.
  * `merged.csv`: A combined file that merges chemical structures with their associated bioactivity data.


## Docker Deployment 🐳

Deploy BioChemInsight in a containerized environment for consistency and portability.

#### Step 1: Build the Docker Image

```bash
docker build -t biocheminsight .
```

#### Step 2: Run the Service

**Option A: Launch the Web App (Default)**

Run this command to start the Gradio interactive interface.

```bash
docker run --rm -d --gpus all \
    -p 7860:7860 \
    -e http_proxy="" \
    -e https_proxy="" \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/constants.py:/app/constants.py \
    biocheminsight
```

After launching, access the UI by visiting `http://localhost:7860` in your browser.

**Option B: Run the Command-Line Pipeline**

If you need to execute a batch job using the CLI, override the default entrypoint by specifying `python pipeline.py` and its arguments after the `docker run` command.

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

**Option C: Enter the Container for an Interactive Session**

To debug or run commands manually inside the container:

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


## Roadmap 📌

  * **Parallel Processing**: Implement batch/parallel recognition of Compound-ID pairs to significantly accelerate processing for large documents.