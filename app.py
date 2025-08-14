import os
import sys
import json
import pandas as pd
import gradio as gr
from pathlib import Path
import tempfile
import shutil
from typing import List

# 添加当前目录到Python路径，以便导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 抑制不必要的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# 导入项目模块
from pipeline import get_total_pages
from structure_parser import extract_structures_from_pdf
from activity_parser import extract_activity_data
import json

class BioChemInsightApp:
    """封装BioChemInsight应用的核心逻辑和UI"""
    def __init__(self):
        """初始化应用，创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.current_pdf_path = None

    def get_pdf_info(self, pdf_file):
        """当用户上传PDF后，获取基本信息"""
        if pdf_file is None:
            return "❌ 请先上传一个PDF文件", 0
        try:
            # 将上传的文件保存到临时目录，以便后续处理
            self.current_pdf_path = os.path.join(self.temp_dir, "uploaded.pdf")
            shutil.copy(pdf_file.name, self.current_pdf_path)
            total_pages = get_total_pages(self.current_pdf_path)
            info = f"✅ PDF上传成功，共 {total_pages} 页"
            return info, total_pages
        except Exception as e:
            return f"❌ PDF加载失败: {str(e)}", 0

    def parse_pages_input(self, pages_str):
        """解析页面输入字符串，支持 '1,3,5-7,10' 格式
        返回页面编号列表
        """
        if not pages_str or not pages_str.strip():
            return []
        
        pages = []
        parts = pages_str.strip().split(',')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if '-' in part:
                # 处理范围 "5-7"
                try:
                    start, end = map(int, part.split('-', 1))
                    pages.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                # 处理单个页面
                try:
                    pages.append(int(part))
                except ValueError:
                    continue
        
        # 去重并排序
        return sorted(list(set(pages)))
    
    def _generate_pdf_gallery(self, start_page, pages_per_view, total_pages):
        """生成可交互的PDF页面预览画廊"""
        if not self.current_pdf_path or not os.path.exists(self.current_pdf_path):
            return "<div class='center-placeholder'>请先上传有效的PDF文件</div>"
        
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(self.current_pdf_path)
            
            start_page = max(1, min(start_page, total_pages))
            end_page = min(start_page + pages_per_view - 1, total_pages)
            pages_to_show = list(range(start_page, end_page + 1))
            
            # 简化的HTML画廊，不使用JavaScript
            gallery_html = f"""
            <div class="gallery-wrapper">
                <div id="selection-info" class="selection-info-bar">
                    <span style="font-weight: 500;">提示: 请使用下方的页面选择工具来选择要处理的页面</span>
                </div>
                <div id="gallery-container" class="gallery-container">
            """
            
            for page_num in pages_to_show:
                page = doc[page_num - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_data = pix.tobytes("png")
                import base64
                img_base64 = base64.b64encode(img_data).decode()
                
                gallery_html += f"""
                <div class="page-item" data-page="{page_num}">
                    <img src='data:image/png;base64,{img_base64}' alt='Page {page_num}' />
                    <div class="page-label">Page {page_num}</div>
                </div>
                """
            
            gallery_html += """
                </div>
            </div>
            """
            doc.close()
            return gallery_html
        except Exception as e:
            return f"<div class='center-placeholder error'>生成预览失败: {str(e)}</div>"

    def update_gallery_view(self, page_input, total_pages):
        """根据用户输入更新画廊视图"""
        if not self.current_pdf_path:
            return "<div class='center-placeholder'>请先上传PDF文件</div>"
        try:
            start_page = int(page_input)
            pages_per_view = 12  # 每页显示12个预览
            return self._generate_pdf_gallery(start_page, pages_per_view, total_pages)
        except (ValueError, TypeError):
            return self._generate_pdf_gallery(1, 12, total_pages)
        except Exception as e:
            return f"<div class='center-placeholder error'>更新视图失败: {str(e)}</div>"

    def extract_structures_only(self, pdf_file, pages_input):
        """仅提取化学结构"""
        if not self.current_pdf_path:
            return "❌ 请先上传PDF文件", "", None
        
        try:
            # 解析页面输入
            page_nums = self.parse_pages_input(pages_input)
            if not page_nums:
                return "❌ 请输入要处理的页面，例如: 1,3,5-7,10", "", None
            
            output_dir = os.path.join(self.temp_dir, "structures_output")
            os.makedirs(output_dir, exist_ok=True)
            
            all_structures = []
            
            # 处理不连续页面：将页面分组为连续的区间
            page_nums.sort()
            groups = []
            current_group = [page_nums[0]]
            
            for i in range(1, len(page_nums)):
                if page_nums[i] == page_nums[i-1] + 1:  # 连续页面
                    current_group.append(page_nums[i])
                else:  # 不连续，开始新组
                    groups.append(current_group)
                    current_group = [page_nums[i]]
            groups.append(current_group)
            
            # 处理每组连续页面
            for group_idx, group in enumerate(groups):
                start_page = min(group)
                end_page = max(group)
                
                # 为每组创建子目录
                group_output_dir = os.path.join(output_dir, f"group_{group_idx}")
                os.makedirs(group_output_dir, exist_ok=True)
                
                structures = extract_structures_from_pdf(
                    pdf_file=self.current_pdf_path,
                    page_start=start_page,
                    page_end=end_page,
                    output=group_output_dir,
                    engine='molnextr'
                )
                
                if structures:
                    # 为每个结构添加页面信息
                    for structure in structures:
                        if isinstance(structure, dict):
                            structure['source_pages'] = list(group)
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
                    df = pd.DataFrame(unique_structures)
                    page_ranges = self._format_page_ranges(page_nums)
                    return f"✅ 成功从页面 {page_ranges} 提取 {len(df)} 个化学结构", df.to_html(classes="result-table", index=False), unique_structures
                else:
                    page_ranges = self._format_page_ranges(page_nums)
                    return f"⚠️ 在页面 {page_ranges} 未找到化学结构", "", None
            else:
                return "❌ 结构提取失败", "", None
        except Exception as e:
            return f"❌ 发生意外错误: {str(e)}", "", None

    def _format_page_ranges(self, page_nums):
        """将页面列表格式化为友好的范围显示"""
        if not page_nums:
            return ""
        
        page_nums.sort()
        ranges = []
        start = page_nums[0]
        end = page_nums[0]
        
        for i in range(1, len(page_nums)):
            if page_nums[i] == end + 1:
                end = page_nums[i]
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = page_nums[i]
        
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ", ".join(ranges)

    def extract_assay_only(self, pdf_file, pages_input, structures_data):
        """仅提取生物活性数据（依赖于已提取的结构）"""
        if not self.current_pdf_path:
            return "❌ 请先上传PDF文件", "", None
        
        # 解析页面输入
        page_nums = self.parse_pages_input(pages_input)
        if not page_nums:
            return "❌ 请输入要处理的页面，例如: 1,3,5-7,10", "", None
        
        if not structures_data:
            return "⚠️ 必须先成功提取化学结构", "", None
        
        try:
            output_dir = os.path.join(self.temp_dir, "assay_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建化合物ID列表
            if isinstance(structures_data[0], dict):
                compound_id_list = [row.get('COMPOUND_ID', row.get('SMILES', '')) for row in structures_data]
            else:
                compound_id_list = structures_data
            
            all_assay_data = {}
            
            # 处理不连续页面：将页面分组为连续的区间
            page_nums.sort()
            groups = []
            current_group = [page_nums[0]]
            
            for i in range(1, len(page_nums)):
                if page_nums[i] == page_nums[i-1] + 1:  # 连续页面
                    current_group.append(page_nums[i])
                else:  # 不连续，开始新组
                    groups.append(current_group)
                    current_group = [page_nums[i]]
            groups.append(current_group)
            
            # 处理每组连续页面
            for group_idx, group in enumerate(groups):
                start_page = min(group)
                end_page = max(group)
                
                assay_dict = extract_activity_data(
                    pdf_file=self.current_pdf_path,
                    assay_page_start=[start_page],
                    assay_page_end=[end_page], 
                    assay_name=f"Bioactivity_Assay_Group_{group_idx}",
                    compound_id_list=compound_id_list,
                    output_dir=output_dir,
                    lang='en'
                )
                
                if assay_dict:
                    all_assay_data.update(assay_dict)
            
            if all_assay_data:
                data = {"Bioactivity_Assay": all_assay_data}
                html = ""
                for name, items in data.items():
                    html += f"<h4>{name}</h4>"
                    if items:
                        df = pd.DataFrame(list(items.items()), columns=['Compound_ID', 'Activity'])
                        html += df.to_html(classes="result-table", index=False)
                
                page_ranges = self._format_page_ranges(page_nums)
                return f"✅ 成功从页面 {page_ranges} 提取活性数据，包含 {len(all_assay_data)} 个化合物", html, data
            else:
                page_ranges = self._format_page_ranges(page_nums)
                return f"⚠️ 在页面 {page_ranges} 未找到生物活性数据", "", None
        except Exception as e:
            return f"❌ 发生意外错误: {str(e)}", "", None

    def extract_both(self, pages_input):
        """同时提取结构和活性数据，并进行合并"""
        s_stat, s_html, s_data = self.extract_structures_only(None, pages_input)
        if not s_data:
            return s_stat, s_html, "结构提取失败，无法继续", "", "合并失败", None, None, None

        a_stat, a_html, a_data = self.extract_assay_only(None, pages_input, s_data)

        m_path, m_stat = None, "无需合并"
        if s_data and a_data:
            try:
                # 简单的数据合并逻辑
                merged_data = []
                for s_item in s_data:
                    compound_id = s_item.get('COMPOUND_ID', s_item.get('SMILES', ''))
                    merged_item = s_item.copy()
                    
                    # 查找对应的活性数据
                    for assay_name, assay_dict in a_data.items():
                        if compound_id in assay_dict:
                            merged_item[assay_name] = assay_dict[compound_id]
                    
                    merged_data.append(merged_item)
                
                if merged_data:
                    df = pd.DataFrame(merged_data)
                    path = os.path.join(self.temp_dir, "merged.csv")
                    df.to_csv(path, index=False, encoding='utf-8-sig')
                    m_path = path
                    m_stat = f"✅ 合并成功，生成 {len(df)} 条记录"
                else:
                    m_stat = "⚠️ 合并成功，但无匹配数据"
            except Exception as e:
                m_stat = f"❌ 合并时发生错误: {e}"

        return s_stat, s_html, a_stat, a_html, m_stat, s_data, a_data, m_path

    def download_file(self, data, file_type, filename):
        """根据数据生成可供下载的文件"""
        if not data: return None
        try:
            path = os.path.join(self.temp_dir, filename)
            if file_type == 'csv':
                pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8-sig')
            elif file_type == 'json':
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return path
        except Exception:
            return None

    def create_interface(self):
        """创建并返回Gradio界面"""
        css = """
        .center-placeholder { text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px; margin-top: 20px; }
        .error { color: red; }
        .gallery-wrapper { background: #f0f2f5; border-radius: 12px; padding: 16px; }
        .selection-info-bar { text-align: center; margin-bottom: 12px; padding: 8px; background: #e8f4fd; border-radius: 8px; }
        .clear-btn { margin-left: 15px; background: #ffc107; color: black; border: none; padding: 4px 10px; border-radius: 5px; cursor: pointer; }
        .gallery-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; max-height: 70vh; overflow-y: auto; padding: 5px; }
        .page-item { border: 2px solid #ddd; border-radius: 8px; cursor: pointer; transition: all 0.2s ease; position: relative; background: white; overflow: hidden; }
        .page-item:hover { border-color: #007bff; }
        .page-item.selected { border-color: #28a745; box-shadow: 0 0 10px rgba(40, 167, 69, 0.5); }
        .page-item .selection-check { position: absolute; top: 5px; right: 5px; width: 24px; height: 24px; border-radius: 50%; background: #28a745; color: white; display: none; align-items: center; justify-content: center; font-size: 16px; font-weight: bold; z-index: 2; }
        .page-item.selected .selection-check { display: flex; }
        .page-item img { width: 100%; height: auto; display: block; }
        .page-item .page-label { padding: 8px; font-size: 13px; font-weight: 500; color: #333; background: #f8f9fa; }
        .result-table { width: 100%; border-collapse: collapse; }
        .result-table th, .result-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .result-table th { background-color: #f2f2f2; }
        """
        with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="BioChemInsight", css=css) as interface:
            # JavaScript函数定义 - 放在最顶部确保优先加载
            gr.Markdown("<h1>🧬 BioChemInsight: 智能生物化学文献数据提取</h1>")
            
            # 状态变量
            total_pages, current_page, structures_data, assay_data, merged_path = (
                gr.State(0), gr.State(1), gr.State(None), gr.State(None), gr.State(None)
            )

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="上传PDF文件", file_types=[".pdf"])
                    pdf_info = gr.Textbox(label="文档信息", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("<h4>页面导航</h4>")
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ 上一页")
                            next_btn = gr.Button("➡️ 下一页")
                        page_input = gr.Number(label="跳转到", value=1, precision=0)
                        go_btn = gr.Button("跳转", variant="primary")

                    with gr.Group():
                        gr.Markdown("<h4>页面选择</h4>")
                        pages_input = gr.Textbox(
                            label="要处理的页面", 
                            placeholder="例如: 1,3,5-7,10 (支持单页面、范围、混合)",
                            info="输入页面编号，支持逗号分隔和连字符范围"
                        )
                    
                    with gr.Group():
                        gr.Markdown("<h4>提取操作</h4>")
                        struct_btn = gr.Button("🧪 仅提取结构")
                        assay_btn = gr.Button("📊 仅提取活性")
                        both_btn = gr.Button("🚀 全部提取并合并", variant="primary")

                with gr.Column(scale=3):
                    gallery = gr.HTML("<div class='center-placeholder'>上传PDF后，此处将显示页面预览</div>")

            with gr.Tabs():
                with gr.TabItem("化学结构"):
                    struct_stat = gr.Textbox(label="状态", interactive=False)
                    struct_disp = gr.HTML()
                    struct_dl_btn = gr.Button("下载结构 (CSV)", visible=False)
                with gr.TabItem("生物活性"):
                    assay_stat = gr.Textbox(label="状态", interactive=False)
                    assay_disp = gr.HTML()
                    assay_dl_btn = gr.Button("下载活性 (JSON)", visible=False)
                with gr.TabItem("合并结果"):
                    merged_stat = gr.Textbox(label="状态", interactive=False)
                    merged_dl_btn = gr.Button("下载合并数据 (CSV)", visible=False)

            # 隐藏的文件组件用于触发下载
            dl_struct, dl_assay, dl_merged = gr.File(visible=False), gr.File(visible=False), gr.File(visible=False)

            # --- 事件处理逻辑 ---
            def on_upload(pdf):
                info, pages = self.get_pdf_info(pdf)
                gallery_html = self.update_gallery_view(1, pages)
                return info, pages, 1, gallery_html, None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            pdf_input.upload(on_upload, [pdf_input], [pdf_info, total_pages, current_page, gallery, structures_data, assay_data, merged_path, struct_dl_btn, assay_dl_btn, merged_dl_btn])

            def handle_nav(page, pages):
                html = self.update_gallery_view(page, pages)
                return page, html
            go_btn.click(handle_nav, [page_input, total_pages], [current_page, gallery])
            prev_btn.click(lambda c, t: handle_nav(max(1, c - 12), t), [current_page, total_pages], [current_page, gallery])
            next_btn.click(lambda c, t: handle_nav(min(t, c + 12), t), [current_page, total_pages], [current_page, gallery])

            def run_struct_extract(pdf, pages_input):
                stat, html, data = self.extract_structures_only(pdf, pages_input)
                return stat, html, data, gr.update(visible=bool(data))
            struct_btn.click(run_struct_extract, inputs=[pdf_input, pages_input], outputs=[struct_stat, struct_disp, structures_data, struct_dl_btn])

            def run_assay_extract(pdf, pages_input, s_data):
                stat, html, data = self.extract_assay_only(pdf, pages_input, s_data)
                return stat, html, data, gr.update(visible=bool(data))
            assay_btn.click(run_assay_extract, inputs=[pdf_input, pages_input, structures_data], outputs=[assay_stat, assay_disp, assay_data, assay_dl_btn])

            def run_both_extract(pages_input):
                s_stat, s_html, a_stat, a_html, m_stat, s_data, a_data, m_path = self.extract_both(pages_input)
                return s_stat, s_html, a_stat, a_html, m_stat, s_data, a_data, m_path, gr.update(visible=bool(s_data)), gr.update(visible=bool(a_data)), gr.update(visible=bool(m_path))
            both_btn.click(run_both_extract, inputs=[pages_input], outputs=[struct_stat, struct_disp, assay_stat, assay_disp, merged_stat, structures_data, assay_data, merged_path, struct_dl_btn, assay_dl_btn, merged_dl_btn])

            struct_dl_btn.click(lambda d: self.download_file(d, 'csv', 'structures.csv'), [structures_data], [dl_struct])
            assay_dl_btn.click(lambda d: self.download_file(d, 'json', 'assay_data.json'), [assay_data], [dl_assay])
            merged_dl_btn.click(lambda p: p, [merged_path], [dl_merged])
            
        return interface

    def __del__(self):
        """在应用关闭时清理临时目录"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"清理临时文件失败: {e}")

def main():
    """主函数，启动Gradio应用"""
    app = BioChemInsightApp()
    interface = app.create_interface()
    interface.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)

if __name__ == "__main__":
    main()
