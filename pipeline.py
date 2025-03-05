import os
import sys
import argparse
import pandas as pd
import PyPDF2
from structure_parser import extract_structures_from_pdf
from activity_parser import extract_activity_data

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')

# Suppress other warnings
import warnings
warnings.filterwarnings("ignore")


def get_total_pages(pdf_file):
    """
    获取 PDF 文件的总页数。
    """
    with open(pdf_file, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        return len(pdf_reader.pages)


def extract_structures(pdf_file, structure_start_page, structure_end_page, output_dir):
    """
    从 PDF 文件中提取化学结构并保存为 CSV 文件。
    """
    structures = extract_structures_from_pdf(
        pdf_file=pdf_file,
        page_start=structure_start_page,
        page_end=structure_end_page,
        output=output_dir
    )
    structure_csv = os.path.join(output_dir, 'structures.csv')
    structures_df = pd.DataFrame(structures)
    structures_df.to_csv(structure_csv, index=False)
    print(f"Chemical structures saved to {structure_csv}")
    return structures_df


def extract_assay(pdf_file, assay_start_page, assay_end_page, assay_name, compound_id_list, output_dir, lang='en'):
    """
    提取指定活性数据，并保存为 JSON 文件。
    """
    assay_dict = extract_activity_data(
        pdf_file=pdf_file,
        assay_page_start=assay_start_page,
        assay_page_end=assay_end_page,
        assay_name=assay_name,
        compound_id_list=compound_id_list,
        output_dir=output_dir,
        lang=lang
    )
    assay_json = os.path.join(output_dir, f"{assay_name}_assay_data.json")
    with open(assay_json, 'w') as f:
        pd.DataFrame(assay_dict.items(), columns=['COMPOUND_ID', assay_name]).to_json(f, orient='records')
    print(f"Assay data saved to {assay_json}")
    return assay_dict


def merge_data(structures_df, assay_data_dicts, output_dir):
    """
    将提取的结构和活性数据合并成一个 CSV 文件。
    """
    for assay_name, assay_dict in assay_data_dicts.items():
        structures_df[assay_name] = structures_df['COMPOUND_ID'].map(assay_dict)

    merged_csv = os.path.join(output_dir, 'merged.csv')
    structures_df.to_csv(merged_csv, index=False)
    print(f"Merged data saved to {merged_csv}")
    return merged_csv


def load_structures(output_dir):
    """
    如果存在 structures.csv，则加载它。
    """
    structure_csv = os.path.join(output_dir, 'structures.csv')
    if os.path.exists(structure_csv):
        print(f"Loading existing structures from {structure_csv}")
        return pd.read_csv(structure_csv)
    else:
        print("No existing structures.csv found.")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract chemical structures and assay data from PDF files.')
    parser.add_argument('pdf_file', type=str, help='PDF file to process')
    parser.add_argument('--structure-start-page', type=int, help='Start page for structures (1-based index)', default=None)
    parser.add_argument('--structure-end-page', type=int, help='End page for structures (inclusive)', default=None)
    parser.add_argument('--assay-start-page', type=int, nargs='+', help='Start page(s) for assays (1-based index)', default=None)
    parser.add_argument('--assay-end-page', type=int, nargs='+', help='End page(s) for assays (inclusive)', default=None)
    parser.add_argument('--assay-names', type=str, help='Assay names to extract (comma-separated)', default='')
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    parser.add_argument('--lang', type=str, help='Language for text extraction', default='en')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # 获取 PDF 总页数
    total_pages = get_total_pages(args.pdf_file)

    # 如果未提供结构提取的页码，则默认从第一页到最后一页
    if args.structure_start_page is None:
        args.structure_start_page = 1
    if args.structure_end_page is None:
        args.structure_end_page = total_pages

    structures_df = extract_structures(
        pdf_file=args.pdf_file,
        structure_start_page=args.structure_start_page,
        structure_end_page=args.structure_end_page,
        output_dir=args.output
    )

    assay_data_dicts = {}

    # 如果提供了 assay 名称，则进行活性数据提取
    if args.assay_names:
        assay_names = args.assay_names.split(',')
        # 如果未提供 assay 页码，则为每个 assay 默认设置从第一页到最后一页
        if args.assay_start_page is None:
            args.assay_start_page = [1] * len(assay_names)
        if args.assay_end_page is None:
            args.assay_end_page = [total_pages] * len(assay_names)
        if len(args.assay_start_page) != len(args.assay_end_page):
            raise ValueError("Number of assay start pages and end pages must match.")
        if len(assay_names) != len(args.assay_start_page):
            raise ValueError("Number of assay names must match the number of assay page ranges.")

        compound_id_list = structures_df['COMPOUND_ID'].tolist() if structures_df is not None else None

        for assay_name, assay_start, assay_end in zip(assay_names, args.assay_start_page, args.assay_end_page):
            assay_data = extract_assay(
                pdf_file=args.pdf_file,
                assay_start_page=assay_start,
                assay_end_page=assay_end,
                assay_name=assay_name,
                compound_id_list=compound_id_list,
                output_dir=args.output,
                lang=args.lang
            )
            assay_data_dicts[assay_name] = assay_data

    # 如果同时提取了结构和 assay 数据，则合并数据
    if structures_df is not None and assay_data_dicts:
        merge_data(structures_df, assay_data_dicts, args.output)
    elif assay_data_dicts:
        print("Assay data extracted but no structures available to merge.")
    else:
        print("No data extracted. Please check your input parameters.")


if __name__ == '__main__':
    main()
