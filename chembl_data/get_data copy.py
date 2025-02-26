import os
import time
import requests
import logging
import pandas as pd
from Bio import Entrez

# 配置日志（INFO 级别输出关键信息）
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 设置 Entrez 邮箱（请替换为你自己的邮箱）
Entrez.email = "your_email@example.com"

def get_target_id_from_name(target_name):
    """
    根据 target 名称从 ChEMBL API 获取 target_id
    """
    url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
    params = {"q": target_name}
    logging.info(f"查询 target 名称: {target_name}")
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
        content_type = response.headers.get("Content-Type", "")
        if response.status_code == 200 and "pdf" in content_type.lower():
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
            logging.warning(f"PubMed ID {pubmed_id} 的全文PDF不可获得，状态码: {response.status_code}, Content-Type: {content_type}")
            return None
    except Exception as e:
        logging.error(f"下载 PDF 时出错: {e}")
        return None

def process_documents_from_chembl(activities, pdf_dir="pdfs"):
    """
    从下载的 ChEMBL 活性数据中提取所有唯一的 document_chembl_id，
    并依次查询文献信息、下载 PDF，
    返回每个文献的详细信息（包含 document_chembl_id、pubmed_id、title、doi、PDF 文件路径）
    """
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
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
        logging.info(f"文献 {doc_id}：PubMed ID: {pubmed_id}, 标题: {title}, DOI: {doi}")
        pdf_path = None
        if pubmed_id:
            pdf_path = download_pdf_from_pubmed(pubmed_id, pdf_dir)
            time.sleep(0.3)  # 避免请求过快
        record = {
            "Document ChEMBL ID": doc_id,
            "PubMed ID": pubmed_id,
            "Title": title,
            "DOI": doi,
            "PDF File": pdf_path
        }
        records.append(record)
    return records

def main():
    # 设置 target 名称，例如 "Estrogen receptor"
    target_name = "Estrogen receptor"
    
    # 第一步：获取 target_id 并下载活性数据
    target_id = get_target_id_from_name(target_name)
    if not target_id:
        logging.error("无法获取 target_id，程序退出")
        return
    logging.info(f"开始下载 target {target_id} 的活性数据")
    activities = download_chembl_activity(target_id)
    logging.info(f"共下载到 {len(activities)} 条活性记录")
    
    # 第二步：从活性数据中提取所有唯一的 document_chembl_id，
    # 并依次查询文献详情、下载 PDF
    records = process_documents_from_chembl(activities, pdf_dir="pdfs")
    
    # 第三步：将文献详细信息与 PDF 文件路径整理成表格并保存为 Excel 文件
    df = pd.DataFrame(records)
    output_file = "document_pdf_mapping.xlsx"
    df.to_excel(output_file, index=False)
    logging.info(f"文献与 PDF 对应表已保存至 {output_file}")

if __name__ == "__main__":
    main()
