from __future__ import annotations

import asyncio
import base64
import hashlib
import tempfile
import io
import os
import sys
from collections import OrderedDict
from datetime import datetime, time

try:  # optional user runtime configuration
    import constants as project_constants
except ImportError:  # pragma: no cover
    project_constants = None

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    str(getattr(project_constants, "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,garbage_collection_threshold:0.8"))
    if project_constants
    else "max_split_size_mb:64,garbage_collection_threshold:0.8",
)

import torch
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
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

from pipeline import (
    auto_detect_assay_names,
    auto_detect_assay_pages,
    auto_detect_structure_pages,
    extract_assays,
    extract_structures,
)

from .pdf_manager import PDFManager
from .schemas import (
    AutoDetectTaskRequest,
    AssayResultResponse,
    AssayTaskRequest,
    FullPipelineRequest,
    MergeResultResponse,
    MergeTaskRequest,
    StructureTaskRequest,
    StructuresResultResponse,
    TaskListResponse,
    TaskStatusResponse,
    UpdateStructuresRequest,
    UploadPDFResponse,
)
from .work_queue import cancel_queued_task, enqueue_task, get_queue_positions
from .task_manager import Task, create_task_manager

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
PDF_STORAGE = DATA_ROOT / "pdfs"
TASK_OUTPUT_ROOT = DATA_ROOT / "tasks"

for path in (DATA_ROOT, PDF_STORAGE, TASK_OUTPUT_ROOT):
    path.mkdir(parents=True, exist_ok=True)

pdf_manager = PDFManager(PDF_STORAGE)
task_manager = create_task_manager()

# 限制并发任务数量，避免系统资源耗尽
def _int_setting(name: str, default: int) -> int:
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return int(env_value)
    return int(getattr(project_constants, name, default)) if project_constants else default


MAX_CONCURRENT_TASKS = _int_setting("MAX_CONCURRENT_TASKS", 2)
STRUCTURE_TASK_CONCURRENCY = _int_setting("STRUCTURE_TASK_CONCURRENCY", 2)
MOLNEXTR_POSTPROCESS_WORKERS = max(1, _int_setting("MOLNEXTR_POSTPROCESS_WORKERS", 1))
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
structure_task_semaphore = asyncio.Semaphore(max(1, STRUCTURE_TASK_CONCURRENCY))

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
    molblock: str | None = None


class RenderSmilesResponse(BaseModel):
    smiles: str
    image: str


class RenderSmilesBatchItem(BaseModel):
    key: str
    smiles: str = ""
    width: int | None = None
    height: int | None = None
    molblock: str | None = None


class RenderSmilesBatchRequest(BaseModel):
    items: List[RenderSmilesBatchItem]


class RenderSmilesBatchResult(BaseModel):
    key: str
    smiles: str
    image: str = ""
    error: str | None = None


class RenderSmilesBatchResponse(BaseModel):
    results: List[RenderSmilesBatchResult]


class TaskCanceled(Exception):
    pass


def _raise_if_task_canceled(task_id: str) -> None:
    task = task_manager.get(task_id)
    if task is not None and task.status == "canceled":
        raise TaskCanceled()


def _mark_task_canceled(task_id: str) -> None:
    task_manager.update(
        task_id,
        status="canceled",
        progress=1.0,
        message="Canceled",
        error=None,
    )


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
    molblock: str | None = None


class RenderSmilesResponse(BaseModel):
    smiles: str
    image: str


class RenderSmilesBatchItem(BaseModel):
    key: str
    smiles: str = ""
    width: int | None = None
    height: int | None = None
    molblock: str | None = None


class RenderSmilesBatchRequest(BaseModel):
    items: List[RenderSmilesBatchItem]


class RenderSmilesBatchResult(BaseModel):
    key: str
    smiles: str
    image: str = ""
    error: str | None = None


class RenderSmilesBatchResponse(BaseModel):
    results: List[RenderSmilesBatchResult]


_RENDER_IMAGE_CACHE: "OrderedDict[str, str]" = OrderedDict()
_RENDER_IMAGE_CACHE_MAX = 512


def _render_cache_key(smiles: str, width: int, height: int, molblock: str | None = None) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(width).encode("utf-8"))
    hasher.update(b"x")
    hasher.update(str(height).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update((smiles or "").strip().encode("utf-8"))
    hasher.update(b"\0")
    hasher.update((molblock or "").strip().encode("utf-8"))
    return hasher.hexdigest()


def _mol_from_molblock(molblock: str) -> Chem.Mol:
    def normalize_molblock(value: str) -> str:
        import re
        normalized = value.strip().lstrip("\ufeff")
        if (normalized.startswith('"') and normalized.endswith('"')) or (
            normalized.startswith("'") and normalized.endswith("'")
        ):
            normalized = normalized[1:-1].strip()
        normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\t", " ")
        normalized = "".join(ch for ch in normalized if ch >= " " or ch == "\n")
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        if "$MOL" in normalized:
            _, normalized = normalized.split("$MOL", 1)
            normalized = normalized.lstrip("\n")
        if "$$$$" in normalized:
            normalized = normalized.split("$$$$", 1)[0].rstrip()
        if "M  END" in normalized:
            normalized = normalized.split("M  END", 1)[0] + "M  END"
        if "M  V30 END CTAB" in normalized and "M  END" not in normalized:
            normalized = normalized.rstrip() + "\nM  END"
        if "nan" in normalized.lower() or "inf" in normalized.lower() or "1.#" in normalized.lower():
            normalized = re.sub(r'(?<!\S)[+-]?(?:nan|inf)(?!\S)', "0.0000", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r'(?<!\S)[+-]?1\.#(?:IND|INF)(?!\S)', "0.0000", normalized, flags=re.IGNORECASE)
        lines = normalized.splitlines()
        version_idx = None
        for idx, line in enumerate(lines):
            if "V2000" in line or "V3000" in line:
                version_idx = idx
                break
        if version_idx is None:
            counts_pattern = re.compile(r"^\\s*\\d+\\s+\\d+\\s+\\d+\\s+\\d+\\s+\\d+\\s+\\d+")
            for idx, line in enumerate(lines):
                if counts_pattern.match(line) and "." not in line:
                    if any("M  V30" in entry for entry in lines):
                        lines[idx] = f"{line.rstrip()} V3000"
                    else:
                        lines[idx] = f"{line.rstrip()} V2000"
                    version_idx = idx
                    break
        if version_idx is not None:
            header = lines[max(0, version_idx - 3):version_idx]
            while len(header) < 3:
                header.insert(0, "")
            lines = header + lines[version_idx:]
            normalized = "\n".join(lines)
        elif any("M  V30" in line for line in lines):
            header = ["", "", ""]
            counts_line = "  0  0  0  0  0  0            999 V3000"
            normalized = "\n".join(header + [counts_line] + lines)
        return normalized

    normalized = normalize_molblock(molblock)
    mol = Chem.MolFromMolBlock(normalized, sanitize=False, removeHs=False, strictParsing=False)
    if mol is None:
        raise ValueError("Invalid molblock")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)
    return mol


def render_smiles_to_image(smiles: str, width: int = 280, height: int = 220, molblock: str | None = None) -> str:
    if Chem is None or Draw is None:
        raise HTTPException(status_code=503, detail="RDKit 未安装，无法生成结构图像")

    normalized_smiles = (smiles or "").strip()
    normalized_molblock = (molblock or "").strip()
    normalized_width = int(width or 280)
    normalized_height = int(height or 220)
    cache_key = _render_cache_key(normalized_smiles, normalized_width, normalized_height, normalized_molblock)
    cached = _RENDER_IMAGE_CACHE.get(cache_key)
    if cached:
        _RENDER_IMAGE_CACHE.move_to_end(cache_key)
        return cached

    if normalized_molblock:
        try:
            mol = _mol_from_molblock(normalized_molblock)
        except Exception:
            raise HTTPException(status_code=400, detail="无法解析提供的 Molfile")
    else:
        if not normalized_smiles:
            raise HTTPException(status_code=400, detail="SMILES 不能为空")
        mol = Chem.MolFromSmiles(normalized_smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="无法解析提供的 SMILES")
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:  # pragma: no cover - coordinates may already exist
            pass

    drawer = Draw.MolDraw2DCairo(normalized_width, normalized_height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    image = f"data:image/png;base64,{encoded}"
    _RENDER_IMAGE_CACHE[cache_key] = image
    if len(_RENDER_IMAGE_CACHE) > _RENDER_IMAGE_CACHE_MAX:
        _RENDER_IMAGE_CACHE.popitem(last=False)
    return image


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
    image = render_smiles_to_image(payload.smiles, payload.width or 280, payload.height or 220, payload.molblock)
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
    image = render_smiles_to_image(payload.smiles, payload.width or 280, payload.height or 220, payload.molblock)
    return RenderSmilesResponse(smiles=payload.smiles.strip(), image=image)


@app.post("/api/chem/render-batch", response_model=RenderSmilesBatchResponse)
async def render_smiles_batch_endpoint(payload: RenderSmilesBatchRequest) -> RenderSmilesBatchResponse:
    results: List[RenderSmilesBatchResult] = []
    # Keep the endpoint bounded so a table scroll cannot create an oversized
    # single request. The frontend sends small viewport batches.
    for item in (payload.items or [])[:32]:
        try:
            image = render_smiles_to_image(item.smiles, item.width or 220, item.height or 170, item.molblock)
            results.append(RenderSmilesBatchResult(key=item.key, smiles=item.smiles.strip(), image=image))
        except HTTPException as exc:
            results.append(
                RenderSmilesBatchResult(
                    key=item.key,
                    smiles=item.smiles.strip(),
                    image="",
                    error=str(exc.detail),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive per-row isolation
            results.append(
                RenderSmilesBatchResult(
                    key=item.key,
                    smiles=item.smiles.strip(),
                    image="",
                    error=str(exc),
                )
            )
    return RenderSmilesBatchResponse(results=results)


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


def _normalize_records(records_raw: List[dict], base_dir: Path) -> List[dict]:
    records: List[dict] = []
    for item in records_raw:
        normalized = {}
        for key, value in item.items():
            if isinstance(value, str):
                lower_key = key.lower()
                if "file" in lower_key or "path" in lower_key:
                    normalized[key] = _normalize_artifact_path(value, base_dir)
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def _load_csv_records(csv_path: Path, base_dir: Path) -> List[dict]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path).fillna("")
    return _normalize_records(df.to_dict(orient="records"), base_dir)


def _get_filtered_structures_csv_path(task: Task) -> Path:
    if task.result_path:
        return Path(task.result_path).with_name("filtered_structures.csv")
    return TASK_OUTPUT_ROOT / task.task_id / "filtered_structures.csv"


def _ensure_usable_identifier(value: str, label: str) -> str:
    if not value or value.strip() in {"undefined", "null"}:
        raise HTTPException(status_code=400, detail=f"{label} is missing")
    return value


def _get_task_or_404(task_id: str) -> Task:
    task_id = _ensure_usable_identifier(task_id, "Task ID")
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


def _request_partition_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        first = forwarded_for.split(",", 1)[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enqueue_work_item(task: Task, task_name: str, partition_id: str, args: list, kwargs: Optional[dict] = None) -> None:
    metadata = dict(task.metadata or {})
    metadata["queue_partition"] = partition_id
    task_manager.update(task.id, metadata=metadata)
    enqueue_task(task.id, task_name, partition_id, args=args, kwargs=kwargs or {})


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


def _assay_names_from_detection_diagnostics(diagnostics: List[dict]) -> List[str]:
    names: List[str] = []
    seen = set()
    for item in diagnostics or []:
        if not isinstance(item, dict):
            continue
        candidates = item.get("detected_assay_names")
        if not isinstance(candidates, list):
            page_decision = item.get("llm_page_detection")
            if isinstance(page_decision, dict):
                candidates = page_decision.get("assay_names")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            name = str(candidate or "").strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
    return names


async def launch_auto_detect_task(
    task_id: str,
    pdf_id: str,
    assay_names: List[str],
    detect_structure_pages: bool,
    detect_assay_pages: bool,
    detect_assay_names: bool,
) -> None:
    async with task_semaphore:
        pdf_doc = pdf_manager.ensure_pdf(pdf_id)
        loop = asyncio.get_running_loop()
        selected_assay_names = [name.strip() for name in (assay_names or []) if name and name.strip()]
        detected_structure_pages: List[int] = []
        detected_assay_pages: List[int] = []
        detected_assay_names: List[str] = []

        try:
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.04, message="Preparing automatic page detection")

            if detect_structure_pages:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.12, message="Detecting structure pages")
                detected_structure_pages, structure_diagnostics = await loop.run_in_executor(
                    None,
                    lambda: auto_detect_structure_pages(str(pdf_doc.stored_path)),
                )
                _raise_if_task_canceled(task_id)
                task = _get_task_or_404(task_id)
                params = dict(task.params or {})
                params.update(
                    {
                        "detected_structure_pages": detected_structure_pages,
                        "structure_detection_diagnostics_preview": structure_diagnostics[:20],
                    }
                )
                task_manager.update(
                    task_id,
                    params=params,
                    progress=0.45,
                    message=f"Detected {len(detected_structure_pages)} structure page{'s' if len(detected_structure_pages) != 1 else ''}",
                )

            if detect_assay_pages:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.52, message="Detecting bioactivity pages")

                def assay_detection_progress(event: dict) -> None:
                    _raise_if_task_canceled(task_id)
                    stage = str((event or {}).get("stage") or "")
                    current = int((event or {}).get("current") or 0)
                    total = int((event or {}).get("total") or 0)
                    message = str((event or {}).get("message") or "Detecting bioactivity pages")
                    if total > 0:
                        fraction = max(0.0, min(1.0, current / total))
                    else:
                        fraction = 0.0
                    if stage == "ocr":
                        progress = 0.52 + fraction * 0.16
                    elif stage == "llm":
                        progress = 0.68 + fraction * 0.10
                    else:
                        progress = 0.52
                    task_manager.update(task_id, progress=min(progress, 0.78), message=message)

                detected_assay_pages, assay_diagnostics = await loop.run_in_executor(
                    None,
                    lambda: auto_detect_assay_pages(
                        str(pdf_doc.stored_path),
                        assay_names=selected_assay_names,
                        progress_callback=assay_detection_progress,
                    ),
                )
                _raise_if_task_canceled(task_id)
                detected_assay_names = _assay_names_from_detection_diagnostics(assay_diagnostics)
                task = _get_task_or_404(task_id)
                params = dict(task.params or {})
                params.update(
                    {
                        "detected_assay_pages": detected_assay_pages,
                        "assay_detection_diagnostics_preview": assay_diagnostics[:20],
                    }
                )
                if detected_assay_names:
                    params["detected_assay_names"] = detected_assay_names
                task_manager.update(
                    task_id,
                    params=params,
                    progress=0.78,
                    message=f"Detected {len(detected_assay_pages)} bioactivity page{'s' if len(detected_assay_pages) != 1 else ''}",
                )

            if detect_assay_names:
                _raise_if_task_canceled(task_id)
                if selected_assay_names:
                    detected_assay_names = selected_assay_names
                elif not detected_assay_names:
                    task_manager.update(task_id, progress=0.82, message="Detecting assay names")
                    detected_assay_names = await loop.run_in_executor(
                        None,
                        lambda: auto_detect_assay_names(
                            str(pdf_doc.stored_path),
                            assay_pages=detected_assay_pages if detected_assay_pages else None,
                        ),
                    )
                _raise_if_task_canceled(task_id)
                task = _get_task_or_404(task_id)
                params = dict(task.params or {})
                params["detected_assay_names"] = detected_assay_names
                task_manager.update(
                    task_id,
                    params=params,
                    progress=0.92,
                    message=(
                        f"Detected {len(detected_assay_names)} assay name{'s' if len(detected_assay_names) != 1 else ''}"
                        if detected_assay_names
                        else "No assay names detected"
                    ),
                )

            _raise_if_task_canceled(task_id)
            task = _get_task_or_404(task_id)
            params = dict(task.params or {})
            task_manager.update(
                task_id,
                status="completed",
                progress=1.0,
                message="Automatic detection plan is ready",
                params=params,
                data=[],
            )
        except TaskCanceled:
            _mark_task_canceled(task_id)
        except Exception as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Automatic detection failed",
                error=str(exc),
            )


async def launch_structure_task(
    task_id: str,
    pdf_id: str,
    pages: List[int],
    engine: str,
    structure_filter_strictness: str,
    auto_detect_pages: bool = False,
) -> None:
    # 使用信号量限制并发任务数量
    async with task_semaphore, structure_task_semaphore:
        try:
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.05, message="Preparing extraction")
            pdf_doc = pdf_manager.ensure_pdf(pdf_id)

            output_dir = TASK_OUTPUT_ROOT / task_id
            output_dir.mkdir(parents=True, exist_ok=True)

            loop = asyncio.get_running_loop()
            selected_pages = list(pages or [])

            if auto_detect_pages:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.08, message="Auto-detecting structure pages")
                selected_pages, detection_diagnostics = await loop.run_in_executor(
                    None,
                    lambda: auto_detect_structure_pages(str(pdf_doc.stored_path)),
                )
                _raise_if_task_canceled(task_id)
                task = _get_task_or_404(task_id)
                next_params = dict(task.params or {})
                next_params["detected_pages"] = selected_pages
                next_params["structure_detection_diagnostics_preview"] = detection_diagnostics[:20]
                task_manager.update(task_id, params=next_params, message=f"Auto-detected {len(selected_pages)} structure pages")

            def progress_callback(current_page, total_pages, message):
                _raise_if_task_canceled(task_id)
                if total_pages > 0:
                    progress = 0.05 + (current_page / total_pages) * 0.8  # Scale progress
                    task_manager.update(task_id, progress=min(progress, 0.85), message=message)

            def task_runner() -> Optional[pd.DataFrame]:
                return extract_structures(
                    pdf_file=str(pdf_doc.stored_path),
                    structure_pages=selected_pages,
                    output_dir=str(output_dir),
                    engine=engine,
                    structure_filter_strictness=structure_filter_strictness,
                    progress_callback=progress_callback,
                )

            df = await loop.run_in_executor(None, task_runner)
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, progress=0.85, message="Post-processing results")
            csv_path = output_dir / "structures.csv"
            filtered_csv_path = output_dir / "filtered_structures.csv"

            if df is None or df.empty:
                empty_df = pd.DataFrame()
                empty_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                filtered_records = _load_csv_records(filtered_csv_path, output_dir)
                task_manager.update(
                    task_id,
                    status="completed",
                    progress=1.0,
                    message=(
                        f"No accepted structures found for selected pages (strictness: {structure_filter_strictness})"
                        + (
                            f"; filtered {len(filtered_records)} candidate{'s' if len(filtered_records) != 1 else ''}"
                            if filtered_records
                            else ""
                        )
                    ),
                    data=[],
                    result_path=str(csv_path),
                )
                return

            df = df.fillna("")
            records = _normalize_records(df.to_dict(orient="records"), output_dir)
            filtered_records = _load_csv_records(filtered_csv_path, output_dir)
            pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
            if filtered_records:
                pd.DataFrame(filtered_records).to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")
            task_manager.update(
                task_id,
                status="completed",
                progress=1.0,
                message=(
                    f"Extracted {len(records)} structures (strictness: {structure_filter_strictness})"
                    + (f"; filtered {len(filtered_records)} candidates" if filtered_records else "")
                ),
                data=records,
                result_path=str(csv_path),
            )
        except TaskCanceled:
            _mark_task_canceled(task_id)
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
    structure_task_id: Optional[str] = None,
    auto_detect_pages: bool = False,
    auto_detect_assay_names_flag: bool = False,
) -> None:
    # 使用信号量限制并发任务数量
    async with task_semaphore:
        try:
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.05, message="Preparing assay extraction")
            pdf_doc = pdf_manager.ensure_pdf(pdf_id)

            output_dir = TASK_OUTPUT_ROOT / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            selected_pages = list(pages or [])
            selected_assay_names = list(assay_names or [])

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

            if auto_detect_pages:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.06, message="Auto-detecting assay pages")

                def assay_task_detection_progress(event: dict) -> None:
                    _raise_if_task_canceled(task_id)
                    stage = str((event or {}).get("stage") or "")
                    current = int((event or {}).get("current") or 0)
                    total = int((event or {}).get("total") or 0)
                    message = str((event or {}).get("message") or "Auto-detecting assay pages")
                    fraction = max(0.0, min(1.0, current / total)) if total > 0 else 0.0
                    if stage == "ocr":
                        progress = 0.06 + fraction * 0.025
                    elif stage == "llm":
                        progress = 0.085 + fraction * 0.015
                    else:
                        progress = 0.06
                    task_manager.update(task_id, progress=min(progress, 0.10), message=message)

                selected_pages, detection_diagnostics = await loop.run_in_executor(
                    None,
                    lambda: auto_detect_assay_pages(
                        str(pdf_doc.stored_path),
                        assay_names=selected_assay_names,
                        progress_callback=assay_task_detection_progress,
                    ),
                )
                _raise_if_task_canceled(task_id)
                if not selected_pages:
                    total_pages = get_total_pages(str(pdf_doc.stored_path))
                    selected_pages = list(range(1, total_pages + 1))
                task = _get_task_or_404(task_id)
                next_params = dict(task.params or {})
                next_params["detected_pages"] = selected_pages
                next_params["assay_detection_diagnostics_preview"] = detection_diagnostics[:20]
                task_manager.update(task_id, params=next_params, message=f"Auto-detected {len(selected_pages)} assay pages")

            if auto_detect_assay_names_flag and not selected_assay_names:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.08, message="Auto-detecting assay names")
                selected_assay_names = await loop.run_in_executor(
                    None,
                    lambda: auto_detect_assay_names(str(pdf_doc.stored_path), assay_pages=selected_pages),
                )
                _raise_if_task_canceled(task_id)
                task = _get_task_or_404(task_id)
                next_params = dict(task.params or {})
                next_params["detected_assay_names"] = selected_assay_names
                task_manager.update(
                    task_id,
                    params=next_params,
                    message=(
                        f"Auto-detected {len(selected_assay_names)} assay name{'s' if len(selected_assay_names) != 1 else ''}"
                        if selected_assay_names
                        else "No assay names auto-detected"
                    ),
                )

            def task_runner(compound_list: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
                def progress_callback(current_group, total_groups, message):
                    _raise_if_task_canceled(task_id)
                    if total_groups > 0:
                        group_progress = current_group / total_groups
                        current_progress = 0.1 + group_progress * 0.75
                        task_manager.update(task_id, progress=min(current_progress, 0.85), message=message)

                if compound_list:
                    print(f"Using compound list for shared assay extraction: {compound_list[:10]}{'...' if len(compound_list) > 10 else ''}")
                else:
                    print("No compound list provided for shared assay extraction, extracting all compounds")

                results = extract_assays(
                    pdf_file=str(pdf_doc.stored_path),
                    assay_pages=selected_pages,
                    assay_names=selected_assay_names,
                    compound_id_list=compound_list,
                    output_dir=str(output_dir),
                    lang=lang,
                    progress_callback=progress_callback,
                )
                _raise_if_task_canceled(task_id)
                task_manager.update(
                    task_id,
                    progress=0.85,
                    message=f"Finished processing {len(selected_assay_names)} assay(s)",
                )
                return results

            if not selected_assay_names:
                raise ValueError("No assay names available for extraction. Provide assay names or enable auto detection.")
            raw_results = await loop.run_in_executor(None, lambda: task_runner(compound_id_list))
            _raise_if_task_canceled(task_id)
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
        except TaskCanceled:
            _mark_task_canceled(task_id)
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
    # 使用信号量限制并发任务数量
    async with task_semaphore:
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
                    if assay_task.type != "bioactivity_extraction":
                        raise ValueError(f"Invalid assay task type: {assay_task_id}")
                    
                    # 获取原始任务参数
                    params = assay_task.params
                    pdf_id = assay_task.pdf_id
                    pages = params.get("pages", [])
                    assay_names = params.get("assay_names", [])
                    lang = params.get("lang", "en")
                    
                    # 获取PDF文件路径
                    pdf_doc = pdf_manager.ensure_pdf(pdf_id)
                    
                    if assay_names:
                        print(f"Re-extracting assays with shared OCR/chunk path: {assay_names}")
                        batch_results = extract_assays(
                            pdf_file=str(pdf_doc.stored_path),
                            assay_pages=pages,
                            assay_names=assay_names,
                            compound_id_list=compound_id_list,
                            output_dir=str(output_dir),
                            lang=lang,
                        )
                        for assay_name, assay_data in (batch_results or {}).items():
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
    pdf_id = _ensure_usable_identifier(pdf_id, "PDF ID")
    pdf_doc = pdf_manager.ensure_pdf(pdf_id)
    return UploadPDFResponse(pdf_id=pdf_doc.id, filename=pdf_doc.filename, total_pages=pdf_doc.total_pages)


@app.get("/api/pdfs/{pdf_id}/pages/{page_num}")
async def get_pdf_page(pdf_id: str, page_num: int, zoom: float = 2.0, max_width: Optional[int] = None) -> dict:
    pdf_id = _ensure_usable_identifier(pdf_id, "PDF ID")
    pdf_doc = pdf_manager.ensure_pdf(pdf_id)
    encoded = await render_pdf_page(pdf_doc.stored_path, page_num, zoom, max_width)
    return {"page": page_num, "image": encoded}


@app.post("/api/tasks/auto-detect", response_model=TaskStatusResponse)
async def queue_auto_detect_task(payload: AutoDetectTaskRequest, request: Request) -> TaskStatusResponse:
    if not (payload.detect_structure_pages or payload.detect_assay_pages or payload.detect_assay_names):
        raise HTTPException(status_code=400, detail="Enable at least one automatic detection target")

    pdf_id = _ensure_usable_identifier(payload.pdf_id, "PDF ID")
    pdf_manager.ensure_pdf(pdf_id)

    task = task_manager.create(
        "auto_detect_plan",
        pdf_id=pdf_id,
        params={
            "assay_names": payload.assay_names,
            "detect_structure_pages": payload.detect_structure_pages,
            "detect_assay_pages": payload.detect_assay_pages,
            "detect_assay_names": payload.detect_assay_names,
        },
    )
    _enqueue_work_item(
        task,
        "auto_detect_plan",
        _request_partition_id(request),
        args=[
            pdf_id,
            payload.assay_names,
            payload.detect_structure_pages,
            payload.detect_assay_pages,
            payload.detect_assay_names,
        ],
    )
    queued = _get_task_or_404(task.id)
    return TaskStatusResponse(**queued.to_dict())


@app.post("/api/tasks/structures", response_model=TaskStatusResponse)
async def queue_structure_task(payload: StructureTaskRequest, request: Request) -> TaskStatusResponse:
    if payload.auto_detect_pages:
        pages = []
    else:
        try:
            pages = parse_pages_input(payload.pages, payload.page_numbers)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    pdf_id = _ensure_usable_identifier(payload.pdf_id, "PDF ID")
    pdf_manager.ensure_pdf(pdf_id)

    task = task_manager.create(
        "structure_extraction",
        pdf_id=pdf_id,
        params={
            "pages": pages,
            "engine": payload.engine,
            "structure_filter_strictness": payload.structure_filter_strictness,
            "auto_detect_pages": payload.auto_detect_pages,
        },
    )
    _enqueue_work_item(
        task,
        "structure_extraction",
        _request_partition_id(request),
        args=[
            pdf_id,
            pages,
            payload.engine,
            payload.structure_filter_strictness,
            payload.auto_detect_pages,
        ],
    )
    queued = _get_task_or_404(task.id)
    return TaskStatusResponse(**queued.to_dict())


@app.post("/api/tasks/assays", response_model=TaskStatusResponse)
async def queue_assay_task(payload: AssayTaskRequest, request: Request) -> TaskStatusResponse:
    if payload.auto_detect_pages:
        pages = []
    else:
        try:
            pages = parse_pages_input(payload.pages, payload.page_numbers)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    pdf_id = _ensure_usable_identifier(payload.pdf_id, "PDF ID")
    pdf_manager.ensure_pdf(pdf_id)

    task = task_manager.create(
        "bioactivity_extraction",
        pdf_id=pdf_id,
        params={
            "pages": pages,
            "assay_names": payload.assay_names,
            "auto_detect_pages": payload.auto_detect_pages,
            "auto_detect_assay_names": payload.auto_detect_assay_names,
            "lang": payload.lang,
            "structure_task_id": payload.structure_task_id,
        },
        metadata={
            "structure_task_id": payload.structure_task_id,
        } if payload.structure_task_id else {}
    )
    _enqueue_work_item(
        task,
        "bioactivity_extraction",
        _request_partition_id(request),
        args=[
            pdf_id,
            pages,
            payload.assay_names,
            payload.lang,
            payload.structure_task_id,
            payload.auto_detect_pages,
            payload.auto_detect_assay_names,
        ],
    )
    queued = _get_task_or_404(task.id)
    return TaskStatusResponse(**queued.to_dict())


class ReparseStructureRequest(BaseModel):
    pdf_id: str
    page_num: int
    segment_idx: int
    engine: str = "molnextr"
    segment_file: Optional[str] = None


@app.post("/api/structures/reparse", response_model=dict)
async def reparse_structure(payload: ReparseStructureRequest) -> dict:
    """Re-parse a specific structure segment using a different engine"""
    pdf_doc = pdf_manager.ensure_pdf(payload.pdf_id)
    
    if not payload.segment_file or not os.path.exists(payload.segment_file):
        raise HTTPException(status_code=400, detail="Segment file not found")
    
    # Import the required engine
    molblock = ''
    if payload.engine == 'molscribe':
        from molscribe import MolScribe
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir="./models")
        model = MolScribe(ckpt_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        result = model.predict_image_file(payload.segment_file, return_atoms_bonds=True, return_confidence=True)
        if isinstance(result, dict):
            smiles = result.get('smiles') or ''
            molblock = (
                result.get('predicted_molfile')
                or result.get('molfile')
                or result.get('molblock')
                or result.get('molfile_v3')
                or result.get('molfileV3')
                or ''
            )
        else:
            smiles = result or ''
    elif payload.engine == 'molvec':
        from rdkit import Chem
        if Chem is None:
            raise HTTPException(status_code=503, detail="RDKit not available for molvec engine")
        cmd = f'java -jar {MOLVEC} -f {payload.segment_file} -o {payload.segment_file}.sdf'
        os.popen(cmd).read()
        try:
            sdf = Chem.SDMolSupplier(f'{payload.segment_file}.sdf')
            if len(sdf) != 0 and sdf[0] is not None:
                smiles = Chem.MolToSmiles(sdf[0])
                molblock = Chem.MolToMolBlock(sdf[0])
            else:
                smiles = ''
        except Exception as e:
            print(f"Error reading SDF: {e}")
            smiles = ''
    elif payload.engine == 'molnextr':
        from utils.MolNexTR import molnextr
        BASE_ = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            '/app/models/molnextr_best.pth',  # Docker environment
            f'{BASE_}/models/molnextr_best.pth',     # 本地相对路径
        ]
        
        ckpt_path = None
        for path in possible_paths:
            if os.path.exists(path):
                ckpt_path = path
                break
        
        # 如果本地没有找到，尝试下载
        if ckpt_path is None:
            try:
                from huggingface_hub import hf_hub_download
                print('正在下载 MolNexTR 模型，这可能需要几分钟...')
                ckpt_path = hf_hub_download('CYF200127/MolNexTR', 'molnextr_best.pth', 
                                          repo_type='dataset', local_dir="./models")
                print(f'模型下载完成: {ckpt_path}')
            except Exception as e:
                print(f'模型下载失败: {e}')
                print('请手动下载模型文件或使用其他引擎 (molscribe/molvec)')
                raise FileNotFoundError(f'MolNexTR model not found. Please download it first or use another engine. Error: {e}')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Loading MolNexTR model from: {ckpt_path}')
        model = molnextr(ckpt_path, device, postprocess_workers=MOLNEXTR_POSTPROCESS_WORKERS)
        result = model.predict_final_results(payload.segment_file, return_atoms_bonds=True, return_confidence=True)
        if isinstance(result, dict):
            smiles = result.get('predicted_smiles') or ''
            molblock = (
                result.get('predicted_molfile')
                or result.get('molfile')
                or result.get('molblock')
                or result.get('molfile_v3')
                or result.get('molfileV3')
                or ''
            )
        else:
            smiles = result or ''
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported engine: {payload.engine}")
    
    return {"smiles": smiles, "molblock": molblock}


@app.post("/api/tasks/merge", response_model=TaskStatusResponse)
async def queue_merge_task(payload: MergeTaskRequest, request: Request) -> TaskStatusResponse:
    task = task_manager.create(
        "data_merge",
        params={
            "structure_task_id": payload.structure_task_id,
            "assay_task_ids": payload.assay_task_ids,
        },
    )
    _enqueue_work_item(
        task,
        "data_merge",
        _request_partition_id(request),
        args=[payload.structure_task_id, payload.assay_task_ids],
    )
    queued = _get_task_or_404(task.id)
    return TaskStatusResponse(**queued.to_dict())


async def launch_full_pipeline_task(
    task_id: str,
    pdf_id: str,
    structure_filter_strictness: str = "strict",
    lang: str = "en",
) -> None:
    async with task_semaphore, structure_task_semaphore:
        pdf_doc = pdf_manager.ensure_pdf(pdf_id)
        loop = asyncio.get_running_loop()
        output_dir = TASK_OUTPUT_ROOT / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Phase 1: Auto-detect (0.00 - 0.15)
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.02, message="Auto-detecting structure pages")

            detected_structure_pages, _structure_diagnostics = await loop.run_in_executor(
                None,
                lambda: auto_detect_structure_pages(
                    str(pdf_doc.stored_path),
                ),
            )
            _raise_if_task_canceled(task_id)
            task_manager.update(
                task_id,
                progress=0.07,
                message=f"Detected {len(detected_structure_pages)} structure page{'s' if len(detected_structure_pages) != 1 else ''}",
            )

            def assay_detection_progress(event: dict) -> None:
                _raise_if_task_canceled(task_id)
                stage = str((event or {}).get("stage") or "")
                current = int((event or {}).get("current") or 0)
                total = int((event or {}).get("total") or 0)
                fraction = max(0.0, min(1.0, current / total)) if total > 0 else 0.0
                if stage == "ocr":
                    progress = 0.07 + fraction * 0.03
                elif stage == "llm":
                    progress = 0.10 + fraction * 0.02
                else:
                    progress = 0.07
                task_manager.update(task_id, progress=min(progress, 0.12), message=str((event or {}).get("message") or "Detecting bioactivity pages"))

            detected_assay_pages, assay_diagnostics = await loop.run_in_executor(
                None,
                lambda: auto_detect_assay_pages(
                    str(pdf_doc.stored_path),
                    progress_callback=assay_detection_progress,
                ),
            )
            _raise_if_task_canceled(task_id)
            detected_assay_names = _assay_names_from_detection_diagnostics(assay_diagnostics)

            if not detected_assay_names:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.12, message="Detecting assay names")
                detected_assay_names = await loop.run_in_executor(
                    None,
                    lambda: auto_detect_assay_names(
                        str(pdf_doc.stored_path),
                        assay_pages=detected_assay_pages if detected_assay_pages else None,
                    ),
                )

            _raise_if_task_canceled(task_id)
            task_manager.update(
                task_id,
                progress=0.15,
                message=(
                    f"Detected {len(detected_structure_pages)} structure pages, "
                    f"{len(detected_assay_pages)} assay pages, "
                    f"{len(detected_assay_names)} assay names"
                ),
                params={
                    "detected_structure_pages": detected_structure_pages,
                    "detected_assay_pages": detected_assay_pages,
                    "detected_assay_names": detected_assay_names,
                },
            )

            # Phase 2: Structure extraction (0.15 - 0.55)
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, progress=0.16, message="Extracting structures")

            def structure_progress_callback(current_page, total_pages, message):
                _raise_if_task_canceled(task_id)
                if total_pages > 0:
                    progress = 0.16 + (current_page / total_pages) * 0.38
                    task_manager.update(task_id, progress=min(progress, 0.54), message=message)

            def structure_runner():
                return extract_structures(
                    pdf_file=str(pdf_doc.stored_path),
                    structure_pages=detected_structure_pages,
                    output_dir=str(output_dir),
                    engine="molnextr",
                    structure_filter_strictness=structure_filter_strictness,
                    progress_callback=structure_progress_callback,
                )

            structures_df = await loop.run_in_executor(None, structure_runner)
            _raise_if_task_canceled(task_id)

            structure_records = []
            compound_id_list = []
            if structures_df is not None and not structures_df.empty:
                structures_df = structures_df.fillna("")
                structure_records = _normalize_records(structures_df.to_dict(orient="records"), output_dir)
                compound_id_list = [str(r.get("COMPOUND_ID", "")) for r in structure_records if r.get("COMPOUND_ID")]
                csv_path = output_dir / "structures.csv"
                pd.DataFrame(structure_records).to_csv(csv_path, index=False, encoding="utf-8-sig")
                filtered_csv_path = output_dir / "filtered_structures.csv"
                filtered_records = _load_csv_records(filtered_csv_path, output_dir)
                if filtered_records:
                    pd.DataFrame(filtered_records).to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")

            _raise_if_task_canceled(task_id)
            task = _get_task_or_404(task_id)
            params = dict(task.params or {})
            params["structure_task_id"] = task_id
            task_manager.update(
                task_id,
                progress=0.55,
                message=f"Extracted {len(structure_records)} structures",
                params=params,
            )

            # Phase 3: Assay extraction (0.55 - 0.90)
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, progress=0.56, message="Extracting bioactivity data")

            if not detected_assay_pages:
                total_pages = len(fitz.open(str(pdf_doc.stored_path)))
                detected_assay_pages = list(range(1, total_pages + 1))

            def assay_progress_callback(current_group, total_groups, message):
                _raise_if_task_canceled(task_id)
                if total_groups > 0:
                    progress = 0.56 + (current_group / total_groups) * 0.33
                    task_manager.update(task_id, progress=min(progress, 0.89), message=message)

            def assay_runner():
                return extract_assays(
                    pdf_file=str(pdf_doc.stored_path),
                    assay_pages=detected_assay_pages,
                    assay_names=detected_assay_names,
                    compound_id_list=compound_id_list if compound_id_list else None,
                    output_dir=str(output_dir),
                    lang=lang,
                    progress_callback=assay_progress_callback,
                )

            assay_results = await loop.run_in_executor(None, assay_runner)
            _raise_if_task_canceled(task_id)

            assay_records = []
            if assay_results:
                record_map: Dict[str, Dict[str, object]] = {}
                for assay_name, assay_data in assay_results.items():
                    if isinstance(assay_data, dict) and "records" in assay_data:
                        for record in assay_data.get("records") or []:
                            if isinstance(record, dict):
                                compound_key = str(record.get("COMPOUND_ID") or "").strip()
                                if not compound_key:
                                    continue
                                merged_record = record_map.setdefault(compound_key, {"COMPOUND_ID": compound_key})
                                for key, value in record.items():
                                    if key != "COMPOUND_ID":
                                        merged_record[key] = _stringify(value)
                        continue
                    for compound_id, value in (assay_data or {}).items():
                        compound_key = str(compound_id)
                        record = record_map.setdefault(compound_key, {"COMPOUND_ID": compound_key})
                        if isinstance(value, dict):
                            for inner_key, inner_value in value.items():
                                record[f"{assay_name}_{inner_key}"] = _stringify(inner_value)
                        else:
                            record[assay_name] = _stringify(value)
                assay_records = list(record_map.values())
                pd.DataFrame(assay_records).to_csv(output_dir / "assays.csv", index=False, encoding="utf-8-sig")

            _raise_if_task_canceled(task_id)
            task_manager.update(
                task_id,
                progress=0.92,
                message=f"Extracted {len(assay_records)} assay records",
            )

            # Phase 4: Done (0.92 - 1.0)
            _raise_if_task_canceled(task_id)
            task_manager.update(
                task_id,
                status="completed",
                progress=1.0,
                message=(
                    f"Pipeline complete: {len(structure_records)} structures, "
                    f"{len(assay_records)} assay records"
                ),
                data=structure_records,
                result_path=str(output_dir / "structures.csv"),
                params={
                    "detected_structure_pages": detected_structure_pages,
                    "detected_assay_pages": detected_assay_pages,
                    "detected_assay_names": detected_assay_names,
                    "structure_records_count": len(structure_records),
                    "assay_records_count": len(assay_records),
                    "structure_records": structure_records,
                    "assay_records": assay_records,
                },
            )
        except TaskCanceled:
            _mark_task_canceled(task_id)
        except Exception as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Full pipeline failed",
                error=str(exc),
            )


@app.post("/api/tasks/full-pipeline", response_model=TaskStatusResponse)
async def queue_full_pipeline_task(payload: FullPipelineRequest, request: Request) -> TaskStatusResponse:
    pdf_id = _ensure_usable_identifier(payload.pdf_id, "PDF ID")
    pdf_manager.ensure_pdf(pdf_id)
    task = task_manager.create(
        "full_pipeline",
        pdf_id=pdf_id,
        params={
            "structure_filter_strictness": payload.structure_filter_strictness,
            "lang": payload.lang,
        },
    )
    _enqueue_work_item(
        task,
        "full_pipeline",
        _request_partition_id(request),
        args=[pdf_id, payload.structure_filter_strictness, payload.lang],
    )
    queued = _get_task_or_404(task.id)
    return TaskStatusResponse(**queued.to_dict())


@app.get("/api/tasks", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=5, le=100),
    limit: Optional[int] = Query(default=None, ge=1, le=200),
    search: str = Query(default=""),
    status: str = Query(default="all"),
    task_type: str = Query(default="all"),
    date_from: str = Query(default=""),
    date_to: str = Query(default=""),
    sort_by: str = Query(default="updated_at"),
    sort_dir: str = Query(default="desc"),
) -> TaskListResponse:
    if limit is not None:
        page = 1
        page_size = max(1, min(int(limit or 20), 200))

    tasks = sorted(task_manager.list(), key=lambda item: item.created_at, reverse=True)
    queue_positions = get_queue_positions()
    running_count = sum(1 for task in tasks if task.status == "running")
    pending_count = sum(1 for task in tasks if task.status == "pending")

    normalized_search = (search or "").strip().lower()
    normalized_status = (status or "all").strip().lower()
    normalized_type = (task_type or "all").strip()
    if normalized_search:
        tasks = [
            task
            for task in tasks
            if normalized_search
            in " ".join(
                [
                    task.id,
                    task.type,
                    task.status,
                    task.message or "",
                    task.pdf_id or "",
                    task.error or "",
                ]
            ).lower()
        ]
    if normalized_status != "all":
        tasks = [task for task in tasks if task.status == normalized_status]
    if normalized_type != "all":
        tasks = [task for task in tasks if task.type == normalized_type]

    date_start = None
    date_end = None
    if date_from:
        try:
            date_start = datetime.combine(datetime.fromisoformat(date_from).date(), time.min)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from")
    if date_to:
        try:
            date_end = datetime.combine(datetime.fromisoformat(date_to).date(), time.max)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to")
    if date_start is not None:
        tasks = [task for task in tasks if task.updated_at >= date_start]
    if date_end is not None:
        tasks = [task for task in tasks if task.updated_at <= date_end]

    allowed_sort_fields = {"task_id", "created_at", "updated_at", "type", "status", "progress", "queue_position"}
    normalized_sort_by = sort_by if sort_by in allowed_sort_fields else "updated_at"
    normalized_sort_dir = "asc" if str(sort_dir).lower() == "asc" else "desc"

    def sort_value(task: Task) -> Any:
        if normalized_sort_by == "task_id":
            return task.id
        if normalized_sort_by == "queue_position":
            return queue_positions.get(task.id) or 10**9
        value = getattr(task, normalized_sort_by)
        if hasattr(value, "timestamp"):
            return value.timestamp()
        return value

    tasks = sorted(tasks, key=sort_value, reverse=normalized_sort_dir == "desc")

    total_count = len(tasks)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)
    start = (page - 1) * page_size
    page_tasks = tasks[start : start + page_size]

    payload = []
    for task in page_tasks:
        item = task.to_dict()
        item["queue_position"] = queue_positions.get(task.id)
        payload.append(TaskStatusResponse(**item))
    return TaskListResponse(
        tasks=payload,
        running_count=running_count,
        pending_count=pending_count,
        max_concurrent_tasks=MAX_CONCURRENT_TASKS,
        structure_task_concurrency=max(1, STRUCTURE_TASK_CONCURRENCY),
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        search=search or "",
        status_filter=normalized_status,
        type_filter=normalized_type,
        date_from=date_from or "",
        date_to=date_to or "",
        sort_by=normalized_sort_by,
        sort_dir=normalized_sort_dir,
    )


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    task = _get_task_or_404(task_id)
    return TaskStatusResponse(**task.to_dict())


@app.post("/api/tasks/{task_id}/cancel", response_model=TaskStatusResponse)
async def cancel_task(task_id: str) -> TaskStatusResponse:
    task = _get_task_or_404(task_id)
    if task.status in {"completed", "failed", "canceled"}:
        return TaskStatusResponse(**task.to_dict())

    if task.status == "pending":
        cancel_queued_task(task_id)
    updated = task_manager.update(task_id, status="canceled", progress=1.0, message="Cancel requested" if task.status == "running" else "Canceled", error=None)
    return TaskStatusResponse(**(updated or task).to_dict())


@app.get("/api/tasks/{task_id}/structures", response_model=StructuresResultResponse)
async def get_task_structures(task_id: str) -> StructuresResultResponse:
    task = _get_task_or_404(task_id)
    if task.type != "structure_extraction":
        raise HTTPException(status_code=400, detail="Task does not contain structure data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    output_dir = Path(task.result_path).parent if task.result_path else (TASK_OUTPUT_ROOT / task.task_id)
    records = _normalize_records(list(task.data or []), output_dir)
    filtered_records = _load_csv_records(_get_filtered_structures_csv_path(task), output_dir)
    return StructuresResultResponse(
        task=TaskStatusResponse(**task.to_dict()),
        records=records,
        filtered_records=filtered_records,
    )


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
    filtered_csv_path = _get_filtered_structures_csv_path(task)
    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    task_manager.update(task_id, data=records, result_path=str(csv_path), message="Structures updated")
    updated = _get_task_or_404(task_id)
    filtered_records = _load_csv_records(filtered_csv_path, csv_path.parent)
    return StructuresResultResponse(
        task=TaskStatusResponse(**updated.to_dict()),
        records=records,
        filtered_records=filtered_records,
    )


def merge_structure_activity_data(structure_task_id: str, activity_data: List[Dict[str, Any]], output_dir: Path) -> Path:
    """
    合并结构数据和活性数据，类似 pipeline.py 的 merge_data 函数
    """
    import pandas as pd
    
    # 加载结构数据
    try:
        structure_task = task_manager.get(structure_task_id)
        if not structure_task or structure_task.status != "completed" or not structure_task.result_path:
            print(f"Structure task {structure_task_id} not completed or no result")
            return None
            
        structure_csv_path = Path(structure_task.result_path)
        if not structure_csv_path.exists():
            print(f"Structure CSV not found: {structure_csv_path}")
            return None
            
        structures_df = pd.read_csv(structure_csv_path)
        print(f"Loaded {len(structures_df)} structures from {structure_csv_path}")
        
    except Exception as e:
        print(f"Error loading structure data: {e}")
        return None
    
    # 将活性数据转换为字典格式 {compound_id: assay_value}
    assay_data_dict = {}
    for record in activity_data:
        if 'COMPOUND_ID' in record and record['COMPOUND_ID']:
            compound_id = str(record['COMPOUND_ID'])
            # 收集所有非 COMPOUND_ID 的字段作为活性数据
            assay_values = {k: v for k, v in record.items() if k != 'COMPOUND_ID'}
            if assay_values:
                assay_data_dict[compound_id] = assay_values
    
    print(f"Activity data for {len(assay_data_dict)} compounds")
    
    # COMPOUND_ID 转为字符串，防止匹配错误
    structures_df['COMPOUND_ID'] = structures_df['COMPOUND_ID'].astype(str)
    
    # 为每个活性数据字段创建新列
    all_assay_fields = set()
    for assay_values in assay_data_dict.values():
        all_assay_fields.update(assay_values.keys())
    
    # 初始化所有活性列为 NaN
    for field in all_assay_fields:
        structures_df[field] = None
    
    # 填充活性数据
    for compound_id, assay_values in assay_data_dict.items():
        mask = structures_df['COMPOUND_ID'] == compound_id
        for field, value in assay_values.items():
            structures_df.loc[mask, field] = value
    
    # 保存合并后的数据
    merged_csv_path = output_dir / "merged.csv"
    structures_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    print(f"Merged data saved to {merged_csv_path}")
    
    return merged_csv_path


@app.get("/api/tasks/{task_id}/download")
async def download_task_artifact(task_id: str) -> FileResponse:
    task = _get_task_or_404(task_id)
    
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    if not task.result_path:
        raise HTTPException(status_code=404, detail="No artifact available")

    # 对于活性提取任务，尝试生成合并的 CSV
    if task.type == "bioactivity_extraction" and hasattr(task, 'metadata') and task.metadata and 'structure_task_id' in task.metadata:
        structure_task_id = task.metadata['structure_task_id']
        
        # 创建合并输出目录
        task_output_dir = Path(task.result_path).parent
        merged_csv_path = merge_structure_activity_data(structure_task_id, task.data or [], task_output_dir)
        
        if merged_csv_path and merged_csv_path.exists():
            return FileResponse(merged_csv_path, filename="merged_results.csv", media_type="text/csv")

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
