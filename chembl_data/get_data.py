import os
import time
import requests
import logging
import pandas as pd
import xml.etree.ElementTree as ET
from Bio import Entrez
from impact_factor.core import Factor
from tqdm import tqdm

from seatable_api import Base

# 配置日志（INFO 级别输出关键信息）
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 设置 Entrez 邮箱（请替换为你自己的邮箱）
Entrez.email = "your_email@example.com"

def get_target_id_from_uniprot(uniprot_id):
    """
    根据 UniProt id 从 ChEMBL API 获取 target_id
    """
    url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
    params = {"target_components__accession": uniprot_id}
    logging.info(f"查询 target 的 UniProt ID: {uniprot_id}")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.error(f"查询 target_id 出错: {e}")
        return None

    targets = data.get("targets", [])
    if not targets:
        logging.warning("未找到匹配的 target")
        return None

    target_id = targets[0].get("target_chembl_id")
    logging.info(f"获得 target_id: {target_id}")
    return target_id

def download_chembl_activity(target_id, limit=1000):
    """
    从 ChEMBL API 下载指定 target_id 的活性数据
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    activities = []
    offset = 0
    logging.info(f"开始下载 target_id {target_id} 的活性数据")
    while True:
        params = {
            "target_chembl_id": target_id,
            "limit": limit,
            "offset": offset
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logging.error(f"下载活性数据时出错: {e}")
            break

        batch = data.get("activities", [])
        if not batch:
            logging.info("无更多活性数据")
            break

        activities.extend(batch)
        logging.info(f"下载 {len(batch)} 条记录, 累计 {len(activities)}")
        page_meta = data.get("page_meta", {})
        if page_meta.get("next"):
            offset += limit
            time.sleep(0.2)
        else:
            break
    return activities

def get_document_details(doc_id):
    """
    根据 document_chembl_id 查询文献详细信息，返回包含 pubmed_id、title、doi 等信息的字典
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/document/{doc_id}.json"
    logging.info(f"查询文献详情: {doc_id}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        logging.error(f"查询文献 {doc_id} 时出错: {e}")
        return None

def get_pubmed_metadata(pubmed_id):
    """
    利用 Entrez.efetch 查询 PubMed 元数据，
    返回包含 title、year、citation、影响因子、JCR 分区、pubmed_id 的字典。
    其中期刊影响因子和 JCR 分区利用 scholarscope.cn API（通过 impact_factor 包）获取。
    """
    logging.info(f"获取 PubMed 元数据: {pubmed_id}")
    try:
        handle = Entrez.efetch(db="pubmed", id=str(pubmed_id), retmode="xml")
        xml_data = handle.read()
        handle.close()
        root = ET.fromstring(xml_data)
        article = root.find("PubmedArticle")
        if article is None:
            return {}
        medline = article.find("MedlineCitation")
        article_elem = medline.find("Article")
        # 提取标题
        title_elem = article_elem.find("ArticleTitle")
        title = title_elem.text if title_elem is not None else ""
        # 提取年份
        journal = article_elem.find("Journal")
        year = ""
        if journal is not None:
            journal_issue = journal.find("JournalIssue")
            if journal_issue is not None:
                pub_date = journal_issue.find("PubDate")
                if pub_date is not None:
                    year_elem = pub_date.find("Year")
                    if year_elem is not None:
                        year = year_elem.text
                    else:
                        medline_date = pub_date.find("MedlineDate")
                        if medline_date is not None:
                            year = medline_date.text.split(" ")[0]
        # 提取期刊名称
        journal_title_elem = journal.find("Title") if journal is not None else None
        journal_title = journal_title_elem.text if journal_title_elem is not None else ""
        # 构造引用格式
        citation = f"{title}. {journal_title}. {year}." if year else f"{title}. {journal_title}."
        
        # 利用 scholarscope.cn API 获取期刊影响因子和 JCR 分区
        try:
            if journal_title:
                fa = Factor()
                row_data_list = fa.search(journal_title)
                if len(row_data_list) > 0:
                    row_data = row_data_list[0]
                    impact_factor_value = row_data.get("factor", "N/A")
                    jcr_partition_value = row_data.get("jcr", "N/A")
                else:
                    impact_factor_value = "N/A"
                    jcr_partition_value = "N/A"
            else:
                impact_factor_value = "N/A"
                jcr_partition_value = "N/A"
        except Exception as e:
            logging.error(f"获取期刊影响因子信息出错: {e}")
            impact_factor_value = "N/A"
            jcr_partition_value = "N/A"
            
        metadata = {
            "Title": title,
            "Year": year,
            "Citation": citation,
            "Impact Factor": impact_factor_value,
            "JCR Partition": jcr_partition_value,
            "PubMed ID": pubmed_id
        }
        logging.info(f"获取到的 PubMed 元数据: {metadata}")
        return metadata
    except Exception as e:
        logging.error(f"获取 PubMed 元数据出错: {e}")
        return {}

def download_pdf_from_pubmed(pubmed_id, save_dir):
    """
    根据 PubMed ID 查询 PMC 是否提供全文 PDF，并尝试下载 PDF 文件。
    流程：
      1. 利用 Entrez.elink 获取对应 PMC 文章的 ID
      2. 构造基础 PDF URL
      3. 请求该 URL（添加请求头模拟浏览器），并跟随重定向获得最终 PDF URL
      4. 下载 PDF 文件，若最终 URL 中包含 .pdf 文件名，则使用该名称保存，否则以 pubmed_id 命名
    """
    logging.info(f"开始下载 PubMed ID {pubmed_id} 对应的 PDF")
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=str(pubmed_id))
        record = Entrez.read(handle)
        handle.close()
    except Exception as e:
        logging.error(f"查询 PubMed ID {pubmed_id} 时出错: {e}")
        return None

    try:
        linksetdb = record[0].get("LinkSetDb", [])
        if not linksetdb:
            logging.warning(f"PubMed ID {pubmed_id} 没有对应的 PMC 文献")
            return None
        pmc_link = linksetdb[0]["Link"][0]
        pmc_id = pmc_link["Id"]
    except Exception as e:
        logging.error(f"解析 PMC 信息出错: {e}")
        return None

    base_pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
    logging.info(f"基础 PDF URL: {base_pdf_url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
    }
    try:
        response = requests.get(base_pdf_url, stream=True, headers=headers)
        final_url = response.url
        logging.info(f"最终 PDF URL: {final_url}")
        # 修改：检查响应状态码和最终 URL 是否以 .pdf 结尾
        if response.status_code == 200 and final_url.lower().endswith('.pdf'):
            filename = final_url.rstrip('/').split('/')[-1]
            if not filename.lower().endswith('.pdf'):
                filename = f"{pubmed_id}.pdf"
            file_path = os.path.join(save_dir, filename)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info(f"成功下载 PubMed ID {pubmed_id} 的 PDF，保存至 {file_path}")
            return file_path
        else:
            logging.warning(f"PubMed ID {pubmed_id} 的全文PDF不可获得，状态码: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"下载 PDF 时出错: {e}")
        return None

def process_documents_from_chembl(activities, target_id, pdf_dir="pdfs"):
    """
    从下载的 ChEMBL 活性数据中提取所有唯一的 document_chembl_id，
    依次查询文献详情、获取 PubMed 元数据、下载 PDF，
    返回包含 document_chembl_id、pubmed_id、title、doi、year、citation、影响因子、JCR 分区、PDF 文件路径 的记录列表
    PDF 将保存在 pdf_dir/target_id/ 目录下
    """
    pdf_save_dir = os.path.join(pdf_dir, target_id)
    if not os.path.exists(pdf_save_dir):
        os.makedirs(pdf_save_dir)
    
    # 从活性数据中提取 document_chembl_id（去重）
    doc_ids = {act.get("document_chembl_id") for act in activities if act.get("document_chembl_id")}
    logging.info(f"从活性数据中提取到 {len(doc_ids)} 个唯一的 document_chembl_id")
    
    records = []
    for doc_id in doc_ids:
        doc_details = get_document_details(doc_id)
        if not doc_details:
            continue
        pubmed_id = doc_details.get("pubmed_id")
        title = doc_details.get("title")
        doi = doc_details.get("doi")
        logging.info(f"处理文献 {doc_id}: PubMed ID: {pubmed_id}, 标题: {title}, DOI: {doi}")
        
        pdf_path = None
        metadata = {}
        if pubmed_id:
            pdf_path = download_pdf_from_pubmed(pubmed_id, pdf_save_dir)
            # 获取 PubMed 元数据，增加 title、year、citation、影响因子、JCR 分区
            metadata = get_pubmed_metadata(pubmed_id)
            time.sleep(0.3)  # 避免请求过快
        
        record = {
            "Document ChEMBL ID": doc_id,
            "PubMed ID": pubmed_id,
            "Title": title or metadata.get("Title", ""),
            "DOI": doi,
            "Year": metadata.get("Year", ""),
            "Citation": metadata.get("Citation", ""),
            "Impact Factor": metadata.get("Impact Factor", ""),
            "JCR Partition": metadata.get("JCR Partition", ""),
            "PDF File": pdf_path
        }
        logging.info(f"生成文献记录: {record}")
        records.append(record)
    return records

def save_compound_table(activities, doc_mapping, target_id, output_file="compound_table.xlsx"):
    """
    从活性数据中整理化合物信息表，
    包括：化合物 SMILES、实验名称、实验描述、实验值、实验值单位、document_chembl_id、pubmed_id 以及 target_id 等信息
    doc_mapping 是一个字典，键为 document_chembl_id，值为对应的 PubMed ID（从文献下载记录中获得）
    """
    compound_records = []
    for act in activities:
        molecule_id = act.get("molecule_chembl_id")
        smiles = act.get("canonical_smiles")
        assay_name = act.get("assay_chembl_id")  # 或其他字段，如 assay_description
        assay_desc = act.get("assay_description", "")
        activity_type = act.get("activity_type")
        standard_value = act.get("standard_value")
        standard_units = act.get("standard_units")
        doc_id = act.get("document_chembl_id")
        pubmed_id = doc_mapping.get(doc_id) if doc_id else None
        
        compound_record = {
            "Target ChEMBL ID": target_id,
            "Molecule ChEMBL ID": molecule_id,
            "Canonical SMILES": smiles,
            "Assay Name": assay_name,
            "Assay Description": assay_desc,
            "Activity Type": activity_type,
            "Standard Value": standard_value,
            "Standard Units": standard_units,
            "Document ChEMBL ID": doc_id,
            "PubMed ID": pubmed_id
        }
        logging.info(f"生成化合物记录: {compound_record}")
        compound_records.append(compound_record)
    
    df = pd.DataFrame(compound_records)
    df.to_excel(output_file, index=False)
    logging.info(f"化合物表已保存至 {output_file}")

def main():
    # 设置 target 的 UniProt ID，例如 "P03372"（Estrogen receptor alpha 的 UniProt ID）
    # uniprot_id = "P08908"
    
    api_token = '7f40a1db768b2956ba49e6ad0bd7f6a8f68f9379'
    base_url = 'http://172.31.8.20/'
    base = Base(api_token, base_url)
    base.auth()

    sql = 'select * from DATA where state != true limit 10000'
    rows = base.query(sql)
    for row in tqdm(rows):
        base.update_row('DATA', row['_id'], {'state': True})
        uniprot_id = row['UNIPROT']

        # 根据 UniProt ID 获取 target_id
        target_id = get_target_id_from_uniprot(uniprot_id)
        if not target_id:
            logging.error("无法获取 target_id，程序退出")
            return
        activities = download_chembl_activity(target_id)
        logging.info(f"共下载到 {len(activities)} 条活性记录")
        
        # 提取文献信息、获取 PubMed 元数据、下载 PDF
        documents = process_documents_from_chembl(activities, target_id, pdf_dir="pdfs")
        
        # 保存文献与 PDF 对应信息（包含 title、year、citation、影响因子、JCR 分区 等）
        doc_df = pd.DataFrame(documents)
        doc_output_file = f"{target_id}_document_pdf_mapping.xlsx"

        doc_df.to_excel(doc_output_file, index=False)

        info_dict = base.upload_local_file(
            file_path=doc_output_file,
            file_type='file',
            replace=True
        )

        base.update_row('DATA', row['_id'], {'DOCS': [info_dict]})

        logging.info(f"文献与 PDF 对应表已保存至 {doc_output_file}")
        
        # 构造文献映射字典，方便在化合物表中关联 PubMed ID
        doc_mapping = {record["Document ChEMBL ID"]: record["PubMed ID"] for record in documents}
        
        # 整理活性数据，生成化合物信息表
        save_compound_table(activities, doc_mapping, target_id, output_file=f"{target_id}_compound_table.xlsx")

        info_dict = base.upload_local_file(
            file_path=f"{target_id}_compound_table.xlsx",
            file_type='file',
            replace=True
        )

        base.update_row('DATA', row['_id'], {'CHEMBL': [info_dict]})
        logging.info(f"化合物表已上传至 SeaTable")

if __name__ == "__main__":
    main()
