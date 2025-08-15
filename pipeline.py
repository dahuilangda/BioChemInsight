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


def extract_structures(pdf_file, structure_pages, output_dir, engine='molnextr'):
    """
    从 PDF 文件中提取化学结构并保存为 CSV 文件。
    支持不连续页面的解析。
    
    Args:
        pdf_file: PDF文件路径
        structure_pages: 页面列表，支持以下格式：
            - 单个页面: 5
            - 页面列表: [1, 3, 5, 7]
            - 页面范围（兼容性）: 如果是元组(start, end)则转换为范围
        output_dir: 输出目录
        engine: 结构识别引擎
    """
    # 处理不同的输入格式
    if isinstance(structure_pages, (int, tuple)):
        if isinstance(structure_pages, tuple) and len(structure_pages) == 2:
            # 兼容旧格式 (start, end)
            start, end = structure_pages
            page_list = list(range(start, end + 1))
        else:
            # 单个页面
            page_list = [structure_pages] if isinstance(structure_pages, int) else list(structure_pages)
    elif isinstance(structure_pages, list):
        page_list = structure_pages
    else:
        raise ValueError("structure_pages must be int, list, or tuple")
    
    print(f"Extracting structures from pages: {page_list}")
    
    all_structures = []
    
    # 将页面分组为连续的区间以优化处理
    def group_consecutive_pages(pages):
        pages = sorted(pages)
        groups = []
        current_group = [pages[0]]
        
        for i in range(1, len(pages)):
            if pages[i] == pages[i-1] + 1:  # 连续页面
                current_group.append(pages[i])
            else:  # 不连续，开始新组
                groups.append(current_group)
                current_group = [pages[i]]
        groups.append(current_group)
        return groups
    
    page_groups = group_consecutive_pages(page_list)
    print(f"Page groups for processing: {page_groups}")
    
    # 处理每组连续页面
    for group_idx, group in enumerate(page_groups):
        start_page = min(group)
        end_page = max(group)
        
        print(f"Processing group {group_idx + 1}: pages {start_page}-{end_page}")
        
        # 为每组创建子输出目录
        group_output_dir = os.path.join(output_dir, f"structures_group_{group_idx}")
        os.makedirs(group_output_dir, exist_ok=True)
        
        structures = extract_structures_from_pdf(
            pdf_file=pdf_file,
            page_start=start_page,
            page_end=end_page,
            output=group_output_dir,
            engine=engine
        )
        
        if structures:
            # 为每个结构添加页面信息
            for structure in structures:
                if isinstance(structure, dict):
                    structure['source_pages'] = list(group)
                    structure['group_id'] = group_idx
                all_structures.extend([structure] if not isinstance(structure, list) else structure)
    
    if all_structures:
        # 去重：如果有重复的SMILES，只保留一个
        seen_smiles = set()
        unique_structures = []
        for structure in all_structures:
            smiles = structure.get('SMILES', '') if isinstance(structure, dict) else str(structure)
            if smiles and smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique_structures.append(structure)
        
        if unique_structures:
            structures_df = pd.DataFrame(unique_structures)
            structure_csv = os.path.join(output_dir, 'structures.csv')
            structures_df.to_csv(structure_csv, index=False, encoding='utf-8-sig')
            print(f"Chemical structures saved to {structure_csv} ({len(structures_df)} unique structures)")
            return structures_df
    
    print("No structures were extracted")
    return None


def extract_assay(pdf_file, assay_pages, assay_name, compound_id_list, output_dir, lang='en'):
    """
    提取指定活性数据，并保存为 JSON 文件。
    支持不连续页面的解析。
    
    Args:
        pdf_file: PDF文件路径
        assay_pages: 页面列表，支持以下格式：
            - 单个页面: 5
            - 页面列表: [1, 3, 5, 7]  
            - 页面范围（兼容性）: 如果是元组(start, end)则转换为范围
        assay_name: 活性测试名称
        compound_id_list: 化合物ID列表
        output_dir: 输出目录
        lang: 语言
    """
    # 处理不同的输入格式
    if isinstance(assay_pages, (int, tuple)):
        if isinstance(assay_pages, tuple) and len(assay_pages) == 2:
            # 兼容旧格式 (start, end)
            start, end = assay_pages
            page_list = list(range(start, end + 1))
        else:
            # 单个页面
            page_list = [assay_pages] if isinstance(assay_pages, int) else list(assay_pages)
    elif isinstance(assay_pages, list):
        page_list = assay_pages
    else:
        raise ValueError("assay_pages must be int, list, or tuple")
    
    print(f"Extracting assay '{assay_name}' from pages: {page_list}")
    
    all_assay_data = {}
    
    # 将页面分组为连续的区间
    def group_consecutive_pages(pages):
        pages = sorted(pages)
        groups = []
        current_group = [pages[0]]
        
        for i in range(1, len(pages)):
            if pages[i] == pages[i-1] + 1:  # 连续页面
                current_group.append(pages[i])
            else:  # 不连续，开始新组
                groups.append(current_group)
                current_group = [pages[i]]
        groups.append(current_group)
        return groups
    
    page_groups = group_consecutive_pages(page_list)
    print(f"Assay page groups for processing: {page_groups}")
    
    # 处理每组连续页面
    for group_idx, group in enumerate(page_groups):
        start_page = min(group)
        end_page = max(group)
        
        print(f"Processing assay group {group_idx + 1}: pages {start_page}-{end_page}")
        
        assay_dict = extract_activity_data(
            pdf_file=pdf_file,
            assay_page_start=[start_page],
            assay_page_end=[end_page],
            assay_name=f"{assay_name}_Group_{group_idx}",
            compound_id_list=compound_id_list,
            output_dir=output_dir,
            lang=lang
        )
        
        if assay_dict:
            all_assay_data.update(assay_dict)
    
    print(f"Assay data extracted for {assay_name}: {len(all_assay_data)} compounds")

    # 保存JSON文件
    assay_json = os.path.join(output_dir, f"{assay_name}_assay_data.json")
    import json
    with open(assay_json, 'w', encoding='utf-8') as f:
        json.dump(all_assay_data, f, ensure_ascii=False, indent=2)
    print(f"Assay data saved to {assay_json}")
    
    return all_assay_data


def merge_data(structures_df, assay_data_dicts, output_dir):
    """
    将提取的结构和活性数据合并成一个 CSV 文件。
    """
    # COMPOUND_ID 转为字符串，防止匹配错误
    structures_df['COMPOUND_ID'] = structures_df['COMPOUND_ID'].astype(str)
    for assay_name, assay_dict in assay_data_dicts.items():
        structures_df[assay_name] = structures_df['COMPOUND_ID'].map(assay_dict)

    print(f'assay_data_dicts: {assay_data_dicts}')

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


def parse_pages_argument(pages_str):
    """
    解析页面参数字符串，支持以下格式：
    - "1-5": 页面范围
    - "1,3,5": 页面列表
    - "1-3,5,7-9": 混合格式
    """
    if not pages_str:
        return None
    
    pages = []
    parts = pages_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # 处理范围，如 "1-5"
            try:
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid page range format: {part}")
        else:
            # 处理单个页面
            try:
                pages.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")
    
    return sorted(list(set(pages)))  # 去重并排序


def main():
    parser = argparse.ArgumentParser(description='Extract chemical structures and assay data from PDF files.')
    parser.add_argument('pdf_file', type=str, help='PDF file to process')
    parser.add_argument('--structure-pages', type=str, help='Pages for structures (e.g., "1-5" or "1,3,5" or "1-3,5,7-9")', default=None)
    parser.add_argument('--assay-pages', type=str, help='Pages for assays (e.g., "1-5" or "1,3,5" or "1-3,5,7-9")', default=None)
    parser.add_argument('--assay-names', type=str, help='Assay names to extract (comma-separated)', default='')
    parser.add_argument('--engine', type=str, help='Engine for structure extraction (molscribe, molnextr, molvec)', default='molnextr')
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    parser.add_argument('--lang', type=str, help='Language for text extraction', default='en')
    
    # 保持向后兼容性的旧参数
    parser.add_argument('--structure-start-page', type=int, help='[DEPRECATED] Use --structure-pages instead', default=None)
    parser.add_argument('--structure-end-page', type=int, help='[DEPRECATED] Use --structure-pages instead', default=None)
    parser.add_argument('--assay-start-page', type=int, nargs='+', help='[DEPRECATED] Use --assay-pages instead', default=None)
    parser.add_argument('--assay-end-page', type=int, nargs='+', help='[DEPRECATED] Use --assay-pages instead', default=None)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # 获取 PDF 总页数
    total_pages = get_total_pages(args.pdf_file)
    print(f"PDF has {total_pages} pages total")

    structures_df = None

    # 处理结构提取
    if args.structure_pages:
        # 使用新的页面格式
        structure_pages = parse_pages_argument(args.structure_pages)
        print(f"Extracting structures from pages: {structure_pages}")
        structures_df = extract_structures(
            pdf_file=args.pdf_file,
            structure_pages=structure_pages,
            output_dir=args.output,
            engine=args.engine
        )
    elif args.structure_start_page is not None and args.structure_end_page is not None:
        # 向后兼容旧格式
        print("Using deprecated --structure-start-page and --structure-end-page. Please use --structure-pages instead.")
        structure_pages = (args.structure_start_page, args.structure_end_page)
        structures_df = extract_structures(
            pdf_file=args.pdf_file,
            structure_pages=structure_pages,
            output_dir=args.output,
            engine=args.engine
        )

    assay_data_dicts = {}

    # 处理活性数据提取
    if args.assay_names:
        assay_names = args.assay_names.split(',')
        assay_names = [name.strip() for name in assay_names]  # 去除空格
        print(f"Assay names to extract: {assay_names}")
        
        # 尝试读取当前目录下的 structures.csv 以获取化合物ID
        if structures_df is None:
            structures_df = load_structures(args.output)

        compound_id_list = structures_df['COMPOUND_ID'].tolist() if structures_df is not None else None
        if compound_id_list:
            print(f"Found {len(compound_id_list)} compound IDs for assay extraction")
        else:
            print("No compound IDs available - extracting assay data without structure matching")

        for assay_name in assay_names:
            print(f"Processing assay: {assay_name}")
            
            if args.assay_pages:
                # 使用新的页面格式
                assay_pages = parse_pages_argument(args.assay_pages)
                print(f"Extracting assay '{assay_name}' from pages: {assay_pages}")
            elif args.assay_start_page is not None and args.assay_end_page is not None:
                # 向后兼容旧格式
                print("Using deprecated --assay-start-page and --assay-end-page. Please use --assay-pages instead.")
                if len(args.assay_start_page) == 1 and len(args.assay_end_page) == 1:
                    assay_pages = (args.assay_start_page[0], args.assay_end_page[0])
                else:
                    # 多个范围的情况
                    assay_pages = []
                    for start, end in zip(args.assay_start_page, args.assay_end_page):
                        assay_pages.extend(range(start, end + 1))
            else:
                # 如果没有指定页面，使用所有页面
                print(f"No assay pages specified for {assay_name}, using all pages")
                assay_pages = list(range(1, total_pages + 1))
            
            assay_data = extract_assay(
                pdf_file=args.pdf_file,
                assay_pages=assay_pages,
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
    elif structures_df is not None:
        print("Structures extracted successfully.")
    else:
        print("No data extracted. Please check your input parameters.")


if __name__ == '__main__':
    main()
