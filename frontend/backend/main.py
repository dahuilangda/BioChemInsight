from __future__ import annotations

import asyncio
import base64
import tempfile
import io
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import rdDepictor
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdchem
    from rdkit.Geometry import Point3D
except ImportError:  # pragma: no cover - optional dependency
    Chem = None
    Draw = None
    rdDepictor = None
    AllChem = None
    rdchem = None
    Point3D = None

from pipeline import extract_assay, extract_structures

from .pdf_manager import PDFManager
from .schemas import (
    AssayResultResponse,
    AssayTaskRequest,
    MergeResultResponse,
    MergeTaskRequest,
    StructureTaskRequest,
    StructuresResultResponse,
    TaskStatusResponse,
    UpdateStructuresRequest,
    UploadPDFResponse,
)
from .task_manager import TaskManager, Task

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
PDF_STORAGE = DATA_ROOT / "pdfs"
TASK_OUTPUT_ROOT = DATA_ROOT / "tasks"

for path in (DATA_ROOT, PDF_STORAGE, TASK_OUTPUT_ROOT):
    path.mkdir(parents=True, exist_ok=True)

pdf_manager = PDFManager(PDF_STORAGE)
task_manager = TaskManager()

app = FastAPI(title="BioChemInsight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RenderSmilesRequest(BaseModel):
    smiles: str
    width: int | None = None
    height: int | None = None


class RenderSmilesResponse(BaseModel):
    smiles: str
    image: str


class EditorAtom(BaseModel):
    id: int
    element: str
    x: float
    y: float


class EditorBond(BaseModel):
    a1: int
    a2: int
    order: int = 1


class MoleculeGraph(BaseModel):
    atoms: List[EditorAtom]
    bonds: List[EditorBond]


class ParseSmilesResponse(MoleculeGraph):
    smiles: str


class BuildMoleculeRequest(MoleculeGraph):
    pass


class BuildMoleculeResponse(BaseModel):
    smiles: str
    image: str


class RenderSmilesRequest(BaseModel):
    smiles: str
    width: int | None = None
    height: int | None = None


class RenderSmilesResponse(BaseModel):
    smiles: str
    image: str


def render_smiles_to_image(smiles: str, width: int = 280, height: int = 220) -> str:
    if Chem is None or Draw is None:
        raise HTTPException(status_code=503, detail="RDKit 未安装，无法生成结构图像")

    normalized = (smiles or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="SMILES 不能为空")

    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        raise HTTPException(status_code=400, detail="无法解析提供的 SMILES")

    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:  # pragma: no cover - fallback when coords exist
        pass

    drawer = Draw.MolDraw2DCairo(width or 280, height or 220)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def smiles_to_graph(smiles: str) -> MoleculeGraph:
    if Chem is None:
        raise HTTPException(status_code=503, detail="RDKit 未安装，无法解析结构")

    normalized = (smiles or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="SMILES 不能为空")

    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        raise HTTPException(status_code=400, detail="无法解析提供的 SMILES")

    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()

    atoms: List[EditorAtom] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(
            EditorAtom(
                id=atom.GetIdx(),
                element=atom.GetSymbol(),
                x=float(pos.x),
                y=float(pos.y),
            ),
        )

    bonds: List[EditorBond] = []
    for bond in mol.GetBonds():
        order = int(round(bond.GetBondTypeAsDouble()))
        order = max(1, min(order, 3))
        bonds.append(
            EditorBond(
                a1=bond.GetBeginAtomIdx(),
                a2=bond.GetEndAtomIdx(),
                order=order,
            ),
        )

    return MoleculeGraph(atoms=atoms, bonds=bonds)


def graph_to_mol(graph: MoleculeGraph) -> Chem.Mol:
    if Chem is None or rdchem is None or AllChem is None:
        raise HTTPException(status_code=503, detail="RDKit 未安装，无法构建结构")

    if not graph.atoms:
        raise HTTPException(status_code=400, detail="请至少提供一个原子")

    rw_mol = Chem.RWMol()
    id_map: Dict[int, int] = {}

    for atom_data in graph.atoms:
        try:
            atom = Chem.Atom(atom_data.element)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=f"无法识别元素 {atom_data.element}") from exc
        new_idx = rw_mol.AddAtom(atom)
        id_map[atom_data.id] = new_idx

    bond_type_map = {
        1: rdchem.BondType.SINGLE,
        2: rdchem.BondType.DOUBLE,
        3: rdchem.BondType.TRIPLE,
    }

    for bond in graph.bonds:
        if bond.a1 not in id_map or bond.a2 not in id_map:
            raise HTTPException(status_code=400, detail="键连接了未知的原子")
        order = max(1, min(int(bond.order or 1), 3))
        try:
            rw_mol.AddBond(id_map[bond.a1], id_map[bond.a2], bond_type_map[order])
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail="无法创建化学键") from exc

    mol = rw_mol.GetMol()

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"结构不合法: {exc}") from exc

    conf = Chem.Conformer(len(graph.atoms))
    for atom_data in graph.atoms:
        idx = id_map[atom_data.id]
        if Point3D is None:
            raise HTTPException(status_code=503, detail="RDKit 缺少坐标支持")
        conf.SetAtomPosition(idx, Point3D(float(atom_data.x), float(atom_data.y), 0.0))

    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    rdDepictor.Compute2DCoords(mol)
    return mol


@app.post("/api/chem/render", response_model=RenderSmilesResponse)
async def render_smiles_endpoint(payload: RenderSmilesRequest) -> RenderSmilesResponse:
    image = render_smiles_to_image(payload.smiles, payload.width or 280, payload.height or 220)
    return RenderSmilesResponse(smiles=payload.smiles.strip(), image=image)


@app.post("/api/chem/parse", response_model=ParseSmilesResponse)
async def parse_smiles_endpoint(payload: RenderSmilesRequest) -> ParseSmilesResponse:
    graph = smiles_to_graph(payload.smiles)
    return ParseSmilesResponse(smiles=payload.smiles.strip(), atoms=graph.atoms, bonds=graph.bonds)


@app.post("/api/chem/build", response_model=BuildMoleculeResponse)
async def build_molecule_endpoint(payload: BuildMoleculeRequest) -> BuildMoleculeResponse:
    mol = graph_to_mol(payload)
    smiles = Chem.MolToSmiles(mol)
    image = render_smiles_to_image(smiles, 280, 220)
    return BuildMoleculeResponse(smiles=smiles, image=image)

@app.post("/api/chem/render", response_model=RenderSmilesResponse)
async def render_smiles_endpoint(payload: RenderSmilesRequest) -> RenderSmilesResponse:
    image = render_smiles_to_image(payload.smiles, payload.width or 280, payload.height or 220)
    return RenderSmilesResponse(smiles=payload.smiles.strip(), image=image)


def parse_pages_input(pages_str: Optional[str], explicit_pages: Optional[List[int]]) -> List[int]:
    if explicit_pages:
        return sorted({p for p in explicit_pages if isinstance(p, int) and p > 0})
    if not pages_str:
        raise ValueError("No pages specified")
    pages = set()
    for part in pages_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start_s, end_s = part.split('-', 1)
                start, end = int(start_s), int(end_s)
                if start > end:
                    start, end = end, start
                pages.update(range(start, end + 1))
            except ValueError as exc:
                raise ValueError(f"Invalid page range '{part}'") from exc
        else:
            try:
                pages.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid page number '{part}'") from exc
    if not pages:
        raise ValueError("No valid pages provided")
    return sorted(pages)


def ensure_within_root(target: Path, root: Path) -> None:
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access to path is not permitted") from exc


def _normalize_artifact_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    alt = (base_dir / candidate).resolve()
    if alt.exists():
        return str(alt)
    return value


def _get_task_or_404(task_id: str) -> Task:
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _stringify(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


async def render_pdf_page(pdf_path: Path, page_num: int, zoom: float = 2.0, max_width: Optional[int] = None) -> str:
    if page_num < 1:
        raise HTTPException(status_code=400, detail="Page numbers are 1-based")
    try:
        with fitz.open(pdf_path) as doc:
            if page_num > doc.page_count:
                raise HTTPException(status_code=404, detail="Page not found")
            page = doc[page_num - 1]
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to render page: {exc}") from exc

    if max_width and pix.width > max_width:
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                width, height = img.size
                if width > max_width:
                    ratio = max_width / float(width)
                    new_height = int(height * ratio)
                    resample = getattr(Image, "Resampling", Image)
                    resized = img.resize((max_width, new_height), resample.LANCZOS)
                    buffer = io.BytesIO()
                    resized.save(buffer, format="PNG", optimize=True)
                    img_bytes = buffer.getvalue()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Failed to resize page: {exc}") from exc

    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return encoded


async def launch_structure_task(task_id: str, pdf_id: str, pages: List[int], engine: str) -> None:
    task_manager.update(task_id, status="running", progress=0.05, message="Preparing extraction")
    pdf_doc = pdf_manager.ensure_pdf(pdf_id)

    output_dir = TASK_OUTPUT_ROOT / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()

    def task_runner() -> Optional[pd.DataFrame]:
        return extract_structures(
            pdf_file=str(pdf_doc.stored_path),
            structure_pages=pages,
            output_dir=str(output_dir),
            engine=engine,
        )

    try:
        df = await loop.run_in_executor(None, task_runner)
        task_manager.update(task_id, progress=0.85, message="Post-processing results")
        csv_path = output_dir / "structures.csv"

        if df is None or df.empty:
            empty_df = pd.DataFrame()
            empty_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            task_manager.update(
                task_id,
                status="completed",
                progress=1.0,
                message="No structures found for selected pages",
                data=[],
                result_path=str(csv_path),
            )
            return

        df = df.fillna("")
        records_raw = df.to_dict(orient="records")
        records: List[dict] = []
        for item in records_raw:
            normalized = {}
            for key, value in item.items():
                if isinstance(value, str):
                    lower_key = key.lower()
                    if "file" in lower_key or "path" in lower_key:
                        normalized[key] = _normalize_artifact_path(value, output_dir)
                    else:
                        normalized[key] = value
                else:
                    normalized[key] = value
            records.append(normalized)
        pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
        task_manager.update(
            task_id,
            status="completed",
            progress=1.0,
            message=f"Extracted {len(records)} structures",
            data=records,
            result_path=str(csv_path),
        )
    except Exception as exc:
        task_manager.update(
            task_id,
            status="failed",
            progress=1.0,
            message="Extraction failed",
            error=str(exc),
        )


async def launch_assay_task(
    task_id: str,
    pdf_id: str,
    pages: List[int],
    assay_names: List[str],
    lang: str,
    ocr_engine: str,
    structure_task_id: Optional[str] = None,
) -> None:
    task_manager.update(task_id, status="running", progress=0.05, message="Preparing assay extraction")
    pdf_doc = pdf_manager.ensure_pdf(pdf_id)

    output_dir = TASK_OUTPUT_ROOT / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取化合物列表（如果提供了结构任务ID）
    compound_id_list = None
    if structure_task_id:
        try:
            structure_task = _get_task_or_404(structure_task_id)
            if structure_task.status == "completed" and structure_task.type == "structure_extraction":
                structure_csv_path = Path(structure_task.result_path)
                if structure_csv_path.exists():
                    structures_df = pd.read_csv(structure_csv_path)
                    compound_id_list = structures_df['COMPOUND_ID'].astype(str).tolist()
                    print(f"Using {len(compound_id_list)} compounds from structure task for matching")
                    task_manager.update(
                        task_id, 
                        progress=0.08, 
                        message=f"Using {len(compound_id_list)} compounds from structure task for matching"
                    )
                else:
                    task_manager.update(task_id, progress=0.08, message="Structure file not found, extracting all compounds")
            else:
                task_manager.update(task_id, progress=0.08, message="Structure task not completed, extracting all compounds")
        except Exception as e:
            print(f"Error loading structure data: {e}")
            task_manager.update(task_id, progress=0.08, message="Error loading structure data, extracting all compounds")

    loop = asyncio.get_running_loop()

    def task_runner(compound_list: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        for idx, assay_name in enumerate(assay_names, start=1):
            sub_dir = output_dir / assay_name.replace(" ", "_")
            sub_dir.mkdir(parents=True, exist_ok=True)
            
            # 添加调试信息
            if compound_list:
                print(f"Using compound list for {assay_name}: {compound_list[:10]}{'...' if len(compound_list) > 10 else ''}")
            else:
                print(f"No compound list provided for {assay_name}, extracting all compounds")
            
            data = extract_assay(
                pdf_file=str(pdf_doc.stored_path),
                assay_pages=pages,
                assay_name=assay_name,
                compound_id_list=compound_list,  # 使用传入的化合物列表
                output_dir=str(sub_dir),
                lang=lang,
                ocr_engine=ocr_engine,
            )
            results[assay_name] = data or {}
            progress = 0.1 + (idx / max(len(assay_names), 1)) * 0.7
            task_manager.update(
                task_id,
                progress=min(progress, 0.85),
                message=f"Processed assay {idx}/{len(assay_names)}: {assay_name}",
            )
        return results

    try:
        raw_results = await loop.run_in_executor(None, lambda: task_runner(compound_id_list))
        task_manager.update(task_id, progress=0.9, message="Compiling assay results")
        csv_path = output_dir / "assays.csv"

        record_map: Dict[str, Dict[str, object]] = {}
        for assay_name, assay_dict in raw_results.items():
            for compound_id, value in (assay_dict or {}).items():
                compound_key = str(compound_id)
                record = record_map.setdefault(compound_key, {"COMPOUND_ID": compound_key})
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        record[f"{assay_name}_{inner_key}"] = _stringify(inner_value)
                else:
                    record[assay_name] = _stringify(value)

        records = list(record_map.values())
        if records:
            pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(csv_path, index=False, encoding="utf-8-sig")

        task_manager.update(
            task_id,
            status="completed",
            progress=1.0,
            message=f"Extracted assay data for {len(records)} compounds",
            data=records,
            result_path=str(csv_path),
        )
    except Exception as exc:
        task_manager.update(
            task_id,
            status="failed",
            progress=1.0,
            message="Assay extraction failed",
            error=str(exc),
        )


async def launch_merge_task(
    task_id: str,
    structure_task_id: str,
    assay_task_ids: List[str],
) -> None:
    task_manager.update(task_id, status="running", progress=0.1, message="Preparing data merge")
    
    # 获取结构任务数据
    structure_task = _get_task_or_404(structure_task_id)
    if structure_task.status != "completed":
        raise ValueError("Structure task must be completed")
    if structure_task.type != "structure_extraction":
        raise ValueError("Invalid structure task type")
    
    output_dir = TASK_OUTPUT_ROOT / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()

    try:
        # 加载结构数据
        structure_csv_path = Path(structure_task.result_path)
        structures_df = pd.read_csv(structure_csv_path)
        structures_df['COMPOUND_ID'] = structures_df['COMPOUND_ID'].astype(str)
        compound_id_list = structures_df['COMPOUND_ID'].tolist()
        
        task_manager.update(task_id, progress=0.2, message=f"Loaded {len(structures_df)} structures, extracting assays with structure matching")
        
        # 按照 pipeline.py 的逻辑：使用结构中的化合物ID重新提取活性数据
        def task_runner() -> Dict[str, Dict[str, object]]:
            assay_data_dicts = {}
            
            for idx, assay_task_id in enumerate(assay_task_ids):
                assay_task = _get_task_or_404(assay_task_id)
                if assay_task.status != "completed":
                    raise ValueError(f"Assay task {assay_task_id} must be completed")
                if assay_task.type != "assay_extraction":
                    raise ValueError(f"Invalid assay task type: {assay_task_id}")
                
                # 获取原始任务参数
                params = assay_task.params
                pdf_id = assay_task.pdf_id
                pages = params.get("pages", [])
                assay_names = params.get("assay_names", [])
                lang = params.get("lang", "en")
                ocr_engine = params.get("ocr_engine", "dots_ocr")
                
                # 获取PDF文件路径
                pdf_doc = pdf_manager.ensure_pdf(pdf_id)
                
                # 为每个assay重新提取数据，这次传入compound_id_list
                for assay_name in assay_names:
                    print(f"Re-extracting assay '{assay_name}' with compound list: {compound_id_list}")
                    
                    # 使用pipeline中的extract_assay函数，传入compound_id_list
                    assay_data = extract_assay(
                        pdf_file=str(pdf_doc.stored_path),
                        assay_pages=pages,
                        assay_name=assay_name,
                        compound_id_list=compound_id_list,  # 关键：传入结构中的化合物列表
                        output_dir=str(output_dir),
                        lang=lang,
                        ocr_engine=ocr_engine
                    )
                    
                    if assay_data:
                        assay_data_dicts[assay_name] = assay_data
                
                progress = 0.2 + (idx + 1) / len(assay_task_ids) * 0.6
                task_manager.update(
                    task_id, 
                    progress=progress, 
                    message=f"Re-extracted assay data {idx + 1}/{len(assay_task_ids)} with structure matching"
                )
            
            return assay_data_dicts
        
        # 在后台线程执行重新提取
        assay_data_dicts = await loop.run_in_executor(None, task_runner)
        
        task_manager.update(task_id, progress=0.9, message="Merging structure and assay data")
        
        # 使用pipeline.py中的merge_data函数进行合并
        from pipeline import merge_data
        merged_csv_path = merge_data(structures_df, assay_data_dicts, str(output_dir))
        
        # 读取合并后的数据
        merged_df = pd.read_csv(merged_csv_path)
        records = merged_df.to_dict('records')
        
        task_manager.update(
            task_id,
            status="completed",
            progress=1.0,
            message=f"Merged data for {len(records)} compounds with structures",
            data=records,
            result_path=merged_csv_path,
        )
    except Exception as exc:
        task_manager.update(
            task_id,
            status="failed",
            progress=1.0,
            message="Data merge failed",
            error=str(exc),
        )


@app.post("/api/pdfs", response_model=UploadPDFResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadPDFResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        pdf_doc = pdf_manager.register(tmp_path, filename=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadPDFResponse(pdf_id=pdf_doc.id, filename=pdf_doc.filename, total_pages=pdf_doc.total_pages)


@app.get("/api/pdfs/{pdf_id}", response_model=UploadPDFResponse)
async def get_pdf(pdf_id: str) -> UploadPDFResponse:
    pdf_doc = pdf_manager.ensure_pdf(pdf_id)
    return UploadPDFResponse(pdf_id=pdf_doc.id, filename=pdf_doc.filename, total_pages=pdf_doc.total_pages)


@app.get("/api/pdfs/{pdf_id}/pages/{page_num}")
async def get_pdf_page(pdf_id: str, page_num: int, zoom: float = 2.0, max_width: Optional[int] = None) -> dict:
    pdf_doc = pdf_manager.ensure_pdf(pdf_id)
    encoded = await render_pdf_page(pdf_doc.stored_path, page_num, zoom, max_width)
    return {"page": page_num, "image": encoded}


@app.post("/api/tasks/structures", response_model=TaskStatusResponse)
async def queue_structure_task(payload: StructureTaskRequest) -> TaskStatusResponse:
    try:
        pages = parse_pages_input(payload.pages, payload.page_numbers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    pdf_manager.ensure_pdf(payload.pdf_id)

    task = task_manager.create("structure_extraction", pdf_id=payload.pdf_id, params={"pages": pages, "engine": payload.engine})
    asyncio.create_task(launch_structure_task(task.id, payload.pdf_id, pages, payload.engine))
    return TaskStatusResponse(**task.to_dict())


@app.post("/api/tasks/assays", response_model=TaskStatusResponse)
async def queue_assay_task(payload: AssayTaskRequest) -> TaskStatusResponse:
    try:
        pages = parse_pages_input(payload.pages, payload.page_numbers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    pdf_manager.ensure_pdf(payload.pdf_id)

    task = task_manager.create(
        "bioactivity_extraction",
        pdf_id=payload.pdf_id,
        params={
            "pages": pages,
            "assay_names": payload.assay_names,
            "lang": payload.lang,
            "ocr_engine": payload.ocr_engine,
            "structure_task_id": payload.structure_task_id,
        },
    )
    asyncio.create_task(launch_assay_task(task.id, payload.pdf_id, pages, payload.assay_names, payload.lang, payload.ocr_engine, payload.structure_task_id))
    return TaskStatusResponse(**task.to_dict())


@app.post("/api/tasks/merge", response_model=TaskStatusResponse)
async def queue_merge_task(payload: MergeTaskRequest) -> TaskStatusResponse:
    task = task_manager.create(
        "data_merge",
        params={
            "structure_task_id": payload.structure_task_id,
            "assay_task_ids": payload.assay_task_ids,
        },
    )
    asyncio.create_task(launch_merge_task(task.id, payload.structure_task_id, payload.assay_task_ids))
    return TaskStatusResponse(**task.to_dict())


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    task = _get_task_or_404(task_id)
    return TaskStatusResponse(**task.to_dict())


@app.get("/api/tasks/{task_id}/structures", response_model=StructuresResultResponse)
async def get_task_structures(task_id: str) -> StructuresResultResponse:
    task = _get_task_or_404(task_id)
    if task.type != "structure_extraction":
        raise HTTPException(status_code=400, detail="Task does not contain structure data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    records = task.data or []
    return StructuresResultResponse(task=TaskStatusResponse(**task.to_dict()), records=records)


@app.get("/api/tasks/{task_id}/assays", response_model=AssayResultResponse)
async def get_task_assays(task_id: str) -> AssayResultResponse:
    task = _get_task_or_404(task_id)
    if task.type != "bioactivity_extraction":
        raise HTTPException(status_code=400, detail="Task does not contain assay data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    records = task.data or []
    return AssayResultResponse(task=TaskStatusResponse(**task.to_dict()), records=records)


@app.get("/api/tasks/{task_id}/merged", response_model=MergeResultResponse)
async def get_task_merged(task_id: str) -> MergeResultResponse:
    task = _get_task_or_404(task_id)
    if task.type != "data_merge":
        raise HTTPException(status_code=400, detail="Task does not contain merged data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    records = task.data or []
    return MergeResultResponse(task=TaskStatusResponse(**task.to_dict()), records=records)


@app.put("/api/tasks/{task_id}/structures", response_model=StructuresResultResponse)
async def update_task_structures(task_id: str, payload: UpdateStructuresRequest) -> StructuresResultResponse:
    task = _get_task_or_404(task_id)
    if task.type != "structure_extraction":
        raise HTTPException(status_code=400, detail="Task does not contain structure data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")

    records = payload.records
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="Records must be a list")

    csv_path = Path(task.result_path or (TASK_OUTPUT_ROOT / task_id / "structures.csv"))
    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    task_manager.update(task_id, data=records, result_path=str(csv_path), message="Structures updated")
    updated = _get_task_or_404(task_id)
    return StructuresResultResponse(task=TaskStatusResponse(**updated.to_dict()), records=records)


@app.get("/api/tasks/{task_id}/download")
async def download_task_artifact(task_id: str) -> FileResponse:
    task = _get_task_or_404(task_id)
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    if not task.result_path:
        raise HTTPException(status_code=404, detail="No artifact available")

    csv_path = Path(task.result_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    return FileResponse(csv_path, filename=csv_path.name, media_type="text/csv")


@app.get("/api/artifacts")
async def get_artifact(path: str) -> dict:
    artifact_path = Path(path).expanduser().resolve()
    ensure_within_root(artifact_path, DATA_ROOT.resolve())

    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        with artifact_path.open("rb") as fh:
            content = fh.read()
    except OSError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    encoded = base64.b64encode(content).decode("utf-8")
    suffix = artifact_path.suffix.lower()
    mime = "image/png" if suffix in {".png", ".jpg", ".jpeg"} else "application/octet-stream"
    return {"path": str(artifact_path), "content": encoded, "mime_type": mime}
