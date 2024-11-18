# BioChemInsight

BioChemInsight platform integrates chemical structure recognition with bioactivity analysis to automatically extract relationships between chemical structures and biological activities from literature. This platform efficiently generates structured data, significantly reducing the cost and complexity of data curation and accelerating the construction of high-quality datasets. This tool not only provides rich training data for machine learning and deep learning models but also lays the foundation for drug optimization and target screening research tasks.

## Features

- **Efficient Identification and Parsing**: Automatically extracts compound structures and biological activity data from PDF documents.
- **Deep Learning and OCR Technologies**: Utilizes DECIMER Segmentation models and PaddleOCR for image and text data processing.
- **Data Transformation**: Converts unstructured data into structured formats for further analysis and application.

## Implementation Principles

The BioChemInsight platform includes two main modules:

1. **Recognition of Compound Structures and Identifiers**:
    - Extracts compound structure images from PDF documents using image processing and visual models.
    - Parses chemical structures into SMILES format and identifies compound numbers.

2. **Recognition of Compound Identifiers and Activities**:
    - Extracts bioactivity data from documents using OCR technology.
    - Analyzes and converts data formats using large language models.

## Implementation Steps

1. **PDF Page Segmentation and Image Conversion**: Process PDF files using the PyPDF2 library, converting them into image formats.
2. **Chemical Structure Detection and Segmentation**: Locate compound images using the DECIMER Segmentation model.
3. **Parsing Chemical Structures into SMILES**: Parse chemical structures using the MolScribe model.
4. **Compound Identifier Recognition**: Identify compound numbers using the MiniCPB-V-2.6 model.
5. **Extraction and Parsing of Bioactivity Data**: Process and analyze bioactivity data using PaddleOCR and large language models.
6. **Data Integration and Table Generation**: Combine all data to generate the final CSV or Excel file.

## Installation

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
# Install dependencies
pip install -r requirements.txt
```

## Usage Example

Here is a simple example of how to run the BioChemInsight platform:

```bash
python extract_data.py --input_path "path/to/your/pdf"
```

For more detailed usage instructions and configuration options, please see the `docs` folder.