from __future__ import annotations

import asyncio
import shutil
import base64
import hashlib
import logging
import json
import tempfile
import io
import os
import re
import sys
import threading
import time
import zipfile
from collections import OrderedDict
from datetime import datetime

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
for _ek, _ev in [
    ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
    ("OPENBLAS_NUM_THREADS", "4"), ("NUMEXPR_NUM_THREADS", "4"),
    ("TOKENIZERS_PARALLELISM", "false"),
    ("TF_NUM_INTRAOP_THREADS", "2"), ("TF_NUM_INTEROP_THREADS", "2"),
]:
    os.environ.setdefault(_ek, _ev)

import torch
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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

from activity_parser import ASSAY_FAILURE_SCOPES, AssayOCRPageError, merge_assay_results
from pipeline import (
    auto_detect_assay_names,
    auto_detect_assay_pages,
    auto_detect_structure_pages_cached,
    auto_detect_assay_pages_cached,
    extract_assays,
    extract_structures,
    get_total_pages,
    load_assay_name_sources,
    verify_assay_names_for_pages,
)
from utils.compound_id_utils import assay_subject_ids, is_assay_subject_structure
from utils.model_harness import append_jsonl
from utils.structure_recognition import StructureRecognizer
from utils.molecule_2d_layout import mol_from_smiles_coordgen, smiles_molblock_consistent, optimize_2d_layout
from utils.pose_refinement import recover_source_coordinates

from .pdf_manager import PDFManager
from .schemas import (
    AnnotationItemsResponse,
    AnnotationUpsertRequest,
    AnnotationsResponse,
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
from .render_utils import apply_atom_render_labels, get_atom_render_label
from .work_queue import cancel_queued_task, enqueue_task, get_queue_positions
from .task_manager import Task, create_task_manager

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
PDF_STORAGE = DATA_ROOT / "pdfs"
TASK_OUTPUT_ROOT = DATA_ROOT / "tasks"
DEFAULT_OCR_LANG = str(getattr(project_constants, "PADDLEOCR_LANG", "auto") or "auto")

for path in (DATA_ROOT, PDF_STORAGE, TASK_OUTPUT_ROOT):
    path.mkdir(parents=True, exist_ok=True)

pdf_manager = PDFManager(PDF_STORAGE)
task_manager = create_task_manager()

def _int_setting(name: str, default: int) -> int:
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return int(env_value)
    return int(getattr(project_constants, name, default)) if project_constants else default


def _float_setting(name: str, default: float) -> float:
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return float(env_value)
    return float(getattr(project_constants, name, default)) if project_constants else default


MAX_CONCURRENT_TASKS = _int_setting("MAX_CONCURRENT_TASKS", 3)
STRUCTURE_TASK_CONCURRENCY = _int_setting("STRUCTURE_TASK_CONCURRENCY", 3)
TASK_STEP_HEARTBEAT_SECONDS = max(1.0, _float_setting("TASK_STEP_HEARTBEAT_SECONDS", 10.0))
TASK_STEP_TIMEOUT_SECONDS = max(0.0, _float_setting("TASK_STEP_TIMEOUT_SECONDS", 0.0))
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
structure_task_semaphore = asyncio.Semaphore(max(1, STRUCTURE_TASK_CONCURRENCY))

app = FastAPI(title="BioChemInsight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    # "*" origins cannot be combined with credentials per the CORS spec; the
    # UI is served from the same origin (port 3000), so same-origin requests
    # need no CORS at all and explicit origins cover proxied deployments.
    allow_origins=str(getattr(project_constants, 'API_ALLOWED_ORIGINS', '') or '').split(',') if str(getattr(project_constants, 'API_ALLOWED_ORIGINS', '') or '').strip() else [],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Accept-Ranges", "Content-Length", "Content-Range", "Content-Disposition"],
)


class RenderSmilesRequest(BaseModel):
    smiles: str
    width: int | None = None
    height: int | None = None
    molblock: str | None = None


class RenderSmilesResponse(BaseModel):
    smiles: str
    image: str


class JsmeMolblockRequest(BaseModel):
    molblock: str
    smiles: str = ""


class JsmeMolblockResponse(BaseModel):
    molblock: str
    changed: bool = False


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


class TaskStepTimeout(Exception):
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


async def _run_interruptible_step(
    task_id: str,
    label: str,
    fn,
    *,
    timeout_seconds: Optional[float] = None,
    heartbeat_seconds: Optional[float] = None,
):
    heartbeat = max(1.0, float(heartbeat_seconds or TASK_STEP_HEARTBEAT_SECONDS))
    timeout = TASK_STEP_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds or 0)
    loop = asyncio.get_running_loop()
    outcome = loop.create_future()

    def publish(result):
        if not outcome.done():
            outcome.set_result(result)

    def target():
        try:
            result = (True, fn())
        except BaseException as exc:
            result = (False, exc)
        try:
            loop.call_soon_threadsafe(publish, result)
        except RuntimeError:
            if not loop.is_closed():
                raise

    _raise_if_task_canceled(task_id)
    started_at_ms = time.time_ns() // 1_000_000
    started = time.monotonic()
    threading.Thread(target=target, name=f"task-step-{task_id[:8]}-{label[:24]}", daemon=True).start()
    last_heartbeat = 0.0
    status = "failed"

    try:
        while True:
            _raise_if_task_canceled(task_id)
            elapsed = time.monotonic() - started
            if timeout > 0 and elapsed >= timeout:
                raise TaskStepTimeout(f"{label} exceeded {int(timeout)}s")
            if elapsed - last_heartbeat >= heartbeat:
                task = task_manager.get(task_id)
                if task is not None and task.status == "running":
                    task_manager.update(task_id, message=task.message or label)
                last_heartbeat = elapsed
            wait = min(heartbeat, 2.0)
            if timeout > 0:
                wait = min(wait, timeout - elapsed)
            done, _ = await asyncio.wait({outcome}, timeout=wait)
            if done:
                ok, payload = outcome.result()
                if ok:
                    status = "completed"
                    return payload
                raise payload
    except (TaskCanceled, asyncio.CancelledError):
        status = "canceled"
        raise
    except TaskStepTimeout:
        status = "timed_out"
        raise
    finally:
        outcome.cancel()
        append_jsonl(TASK_OUTPUT_ROOT / task_id / "task_steps.jsonl", {
            "step": label,
            "status": status,
            "started_at_ms": started_at_ms,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        })


async def _run_parallel_steps(*steps):
    tasks = [asyncio.create_task(step) for step in steps]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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


_RENDER_IMAGE_CACHE: "OrderedDict[str, str]" = OrderedDict()
_RENDER_IMAGE_CACHE_MAX = 512


def _render_cache_key(smiles: str, width: int, height: int, molblock: str | None = None) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"bounded-source-pose-v1")
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


def _prepare_mol_for_render(mol: Chem.Mol) -> Chem.Mol:
    """Make a detached copy before applying render-time label overrides."""
    return Chem.Mol(mol)


def prepare_molblock_for_jsme(molblock: str, smiles: str = "") -> tuple[str, bool]:
    normalized_molblock = (molblock or "").strip()
    if not normalized_molblock and not smiles:
        return "", False
    try:
        if normalized_molblock:
            mol = _mol_from_molblock(normalized_molblock)
            if smiles and not smiles_molblock_consistent(smiles, normalized_molblock):
                from utils.pose_refinement import transfer_source_pose
                mol = transfer_source_pose(smiles, mol) or mol_from_smiles_coordgen(smiles)
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return "", False
            rdDepictor.Compute2DCoords(mol)
        prepared = Chem.Mol(mol)
        optimize_2d_layout(prepared)
        try:
            Chem.Kekulize(prepared, clearAromaticFlags=True)
        except Exception:
            return normalized_molblock, False
        prepared_molblock = Chem.MolToMolBlock(prepared, kekulize=True)
        if prepared_molblock.strip():
            return prepared_molblock, prepared_molblock.strip() != normalized_molblock
    except Exception:
        return normalized_molblock, False
    return normalized_molblock, False


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
        # When the molblock's atom composition contradicts the SMILES, the
        # pose is not trustworthy: regenerate from the SMILES, keeping the
        # original molblock if regeneration is invalid.
        if normalized_smiles and mol is not None and not smiles_molblock_consistent(normalized_smiles, normalized_molblock):
            try:
                from utils.pose_refinement import transfer_source_pose
                fresh_mol = transfer_source_pose(normalized_smiles, mol) or mol_from_smiles_coordgen(normalized_smiles)
                if (fresh_mol is not None and fresh_mol.GetNumConformers() > 0
                        and smiles_molblock_consistent(normalized_smiles, Chem.MolToMolBlock(fresh_mol))):
                    mol = fresh_mol
            except Exception:  # pragma: no cover - keep the molblock version on failure
                pass
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

    mol = _prepare_mol_for_render(mol)
    optimize_2d_layout(mol)

    drawer = Draw.MolDraw2DCairo(normalized_width, normalized_height)
    apply_atom_render_labels(drawer, mol)
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


@app.post("/api/chem/jsme-molblock", response_model=JsmeMolblockResponse)
async def jsme_molblock_endpoint(payload: JsmeMolblockRequest) -> JsmeMolblockResponse:
    molblock, changed = prepare_molblock_for_jsme(payload.molblock, payload.smiles)
    return JsmeMolblockResponse(molblock=molblock, changed=changed)


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


MAX_PAGE_RANGE_SPAN = 10000
MAX_ZOOM = 8.0


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
                if start < 1 or end - start + 1 > MAX_PAGE_RANGE_SPAN:
                    raise ValueError(f"Invalid page range '{part}'")
                pages.update(range(start, end + 1))
            except ValueError as exc:
                raise ValueError(f"Invalid page range '{part}'") from exc
        else:
            try:
                page = int(part)
                if page < 1:
                    raise ValueError
                pages.add(page)
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
    # Strip the Docker-internal /app/ prefix so paths resolve on the host.
    if value.startswith("/app/"):
        candidate = Path(value[5:])
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    alt = (base_dir / candidate).resolve()
    if alt.exists():
        return str(alt)
    return value


def _normalize_records(records_raw: List[dict], base_dir: Path, *, with_record_ids: bool = False) -> List[dict]:
    internal_columns = {
        "CANDIDATE_SOURCE",
        "FRAGMENT_SMILES",
        "MOLNEXTR_EXPECTED_STRUCTURE_TYPE",
        "MOLNEXTR_EXPERT_WEIGHTS",
        "MOLNEXTR_ROUTED_EXPERT",
        "MOLNEXTR_ROUTING_CONFIDENCE",
        "MOLNEXTR_ROUTING_FORCED_COMPLETE",
        "MOLNEXTR_ROUTING_FORCED_DEFAULT",
        "MOLNEXTR_ROUTING_REQUIRED_THRESHOLD",
        "MOLNEXTR_ROUTING_STRATEGY",
        "MOLNEXTR_CONFIDENCE",
        "STRUCTURE_POSE_CLEANUP",
        "REJECTED_ORIG_SMILES",
        "REJECTED_ORIG_MOLBLOCK",
        "MOLNEXTR_RAW_MOLBLOCK",
    }
    records: List[dict] = []
    for record_index, item in enumerate(records_raw):
        normalized = {}
        for key, value in item.items():
            if key in internal_columns:
                continue
            if isinstance(value, str):
                lower_key = key.lower()
                if "file" in lower_key or "path" in lower_key:
                    normalized[key] = _normalize_artifact_path(value, base_dir)
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        if with_record_ids and not str(normalized.get("RECORD_ID") or "").strip():
            # Stable across reloads before the first edit; subsequent edits carry
            # this opaque identity, independently of compound IDs or row order.
            import uuid
            normalized["RECORD_ID"] = uuid.uuid5(uuid.NAMESPACE_URL, f"biocheminsight:{base_dir.resolve()}:{record_index}").hex
        records.append(normalized)
    return records




_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe_frame(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Neutralize spreadsheet formula injection in cells destined for CSV.

    Cells beginning with = + - @ or a tab are written with a leading single
    quote so Excel/WPS treat them as text, not formulas.
    """
    def _clean(value):
        if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
            return "'" + value
        return value
    return frame.map(_clean)


def _load_csv_records(csv_path: Path, base_dir: Path) -> List[dict]:
    if not csv_path.exists():
        return []
    try:
        # dtype=str: compound ids like "0012" / "3-5" / "1E5" must survive
        # a read-write round trip unchanged.
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return []
    return _normalize_records(df.to_dict(orient="records"), base_dir)


def _get_filtered_structures_csv_path(task: Task) -> Path:
    if task.result_path:
        return Path(task.result_path).with_name("filtered_structures.csv")
    return TASK_OUTPUT_ROOT / task.id / "filtered_structures.csv"


def _get_markush_relationships_path(task: Task) -> Path:
    if task.result_path:
        return Path(task.result_path).with_name("markush_relationships.json")
    return TASK_OUTPUT_ROOT / task.id / "markush_relationships.json"


def _load_markush_relationships(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return []
    return payload if isinstance(payload, list) else []


def _task_status_response(task: Task, *, compact_params: bool = False) -> TaskStatusResponse:
    item = task.to_dict()
    if compact_params:
        item["params"] = _compact_task_params_for_detail(item.get("params") or {})
    item["filename"] = _resolve_task_filename(item)
    return TaskStatusResponse(**item)


def _resolve_task_filename(task: Dict[str, Any]) -> Optional[str]:
    pdf_id = task.get("pdf_id")
    if not pdf_id:
        return None
    try:
        return _ensure_pdf_or_404(str(pdf_id)).filename
    except Exception:
        return None


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


def _clean_unit(value: Any) -> str:
    unit = str(value or "").strip()
    return unit.strip("[]() ")


def _unit_from_assay_name(name: str) -> str:
    matches = re.findall(r"\(([^()]{1,24})\)", str(name or ""))
    if not matches:
        return ""
    candidate = matches[-1].strip()
    if re.search(r"[A-Za-z%μµ]", candidate):
        return _clean_unit(candidate)
    return ""


def _split_value_unit(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    match = re.fullmatch(
        r"([<>≤≥~≈]?\s*-?\d+(?:\.\d+)?(?:\s*[x×]\s*10\^?-?\d+)?)\s*([A-Za-zμµ%/][A-Za-z0-9μµ%/.\-]*)",
        text,
    )
    if not match:
        return text, ""
    return match.group(1).replace(" ", ""), _clean_unit(match.group(2))


def _assay_column_name(name: str, unit: str = "") -> str:
    from utils.measurement_evidence import assay_column_with_unit

    return assay_column_with_unit(str(name or "").strip() or "Assay", _clean_unit(unit))


def _flatten_assay_value(assay_name: str, value: Any) -> tuple[str, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "unit": _unit_from_assay_name(assay_name),
        "method": str(assay_name or "").strip(),
        "description": "",
        "confidence": "",
        "reason": "",
    }
    if isinstance(value, dict):
        raw_value = value.get("value") if "value" in value else value.get("VALUE")
        text_value, inline_unit = _split_value_unit(raw_value)
        metadata.update({key: str(value.get(key) or "") for key in ("endpoint", "target", "source_kind")})
        metadata["unit"] = _clean_unit(value.get("unit") or value.get("UNIT") or inline_unit or metadata["unit"])
        metadata["method"] = str(value.get("method") or value.get("METHOD") or metadata["method"]).strip()
        metadata["description"] = str(value.get("description") or value.get("DESCRIPTION") or "").strip()
        metadata["confidence"] = str(value.get("confidence") or value.get("CONFIDENCE") or "").strip()
        metadata["reason"] = str(value.get("reason") or value.get("REASON") or "").strip()
        return _stringify(text_value), metadata
    text_value, inline_unit = _split_value_unit(value)
    if inline_unit:
        metadata["unit"] = inline_unit
    return _stringify(text_value), metadata


def _records_from_assay_results(assay_results: Dict[str, Any]) -> tuple[List[Dict[str, object]], Dict[str, Dict[str, str]]]:
    from utils.measurement_evidence import assay_source_columns, condition_measurement_values

    record_map: Dict[str, Dict[str, object]] = {}
    metadata_by_column: Dict[str, Dict[str, str]] = {}
    for assay_name, assay_data in (assay_results or {}).items():
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
            qualified = condition_measurement_values(value)
            if qualified:
                record.update(qualified)
                continue
            value_text, metadata = _flatten_assay_value(str(assay_name), value)
            if metadata.get("endpoint"):
                from utils.measurement_evidence import measurement_column
                column_name = measurement_column(metadata)
            else:
                column_name = _assay_column_name(str(assay_name), str(metadata.get("unit") or ""))
            record[column_name] = value_text
            record.update(assay_source_columns(column_name, value))
            if metadata.get("source_kind"):
                record[f"{column_name}__source_kind"] = metadata["source_kind"]
            if metadata.get("method"):
                record[f"{column_name}__method"] = metadata["method"]
            if metadata.get("description"):
                record[f"{column_name}__description"] = metadata["description"]
            if metadata.get("confidence"):
                record[f"{column_name}__confidence"] = metadata["confidence"]
            if metadata.get("reason"):
                record[f"{column_name}__reason"] = metadata["reason"]
            metadata_by_column.setdefault(column_name, {
                "unit": str(metadata.get("unit") or ""),
                "method": str(metadata.get("method") or ""),
                "description": str(metadata.get("description") or ""),
            })
    return list(record_map.values()), metadata_by_column


def _write_assay_metadata(metadata: Dict[str, Dict[str, str]], output_path: Path) -> None:
    lines = [
        "Bioactivity metadata",
        "",
        "Columns ending with __method, __description, __confidence, and __reason preserve extraction context.",
        "The __source_page column records PDF page numbers. A false __structure_id_supported value marks an unlinked measurement.",
        "For symbolic values such as ++, +++, ND, or inactive, check the matching __description column and this summary.",
        "",
    ]
    if not metadata:
        lines.append("No assay metadata available.")
    for column, details in sorted(metadata.items()):
        lines.append(f"- {column}")
        lines.append(f"  unit: {details.get('unit') or 'not specified'}")
        lines.append(f"  method: {details.get('method') or 'not specified'}")
        if details.get("description"):
            lines.append(f"  description: {details['description']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_HEAVY_TASK_PARAM_KEYS = {"structure_records", "assay_records"}


def _compact_task_params_for_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return lightweight params for Jobs list responses.

    Completed full-pipeline tasks keep extracted records in ``params`` so that
    opening the task can restore results. Returning those large arrays on every
    Jobs poll makes the list slow and network-heavy. The list only needs
    summaries; ``/api/tasks/{task_id}`` still returns the full params.
    """
    compact: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key in _HEAVY_TASK_PARAM_KEYS:
            if isinstance(value, (list, dict)):
                compact[f"{key}_count"] = len(value)
            continue
        if isinstance(value, list):
            if len(value) > 80:
                compact[f"{key}_count"] = len(value)
            elif all(not isinstance(item, (dict, list, tuple, set)) for item in value):
                compact[key] = value
            else:
                compact[f"{key}_count"] = len(value)
            continue
        if isinstance(value, dict):
            compact[f"{key}_keys"] = list(value.keys())[:20]
            compact[f"{key}_count"] = len(value)
            continue
        compact[key] = value
    return compact


def _compact_task_params_for_detail(params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep task-detail state restorable without returning embedded result rows."""
    compact: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key in _HEAVY_TASK_PARAM_KEYS:
            if isinstance(value, (list, dict)):
                compact[f"{key}_count"] = len(value)
            continue
        compact[key] = value
    return compact


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


VERIFICATION_ROOT = Path(os.getenv("VERIFICATION_ROOT", str(PROJECT_ROOT / "data" / "verification")))


def _load_verification_items() -> List[dict]:
    items_path = VERIFICATION_ROOT / "items.json"
    if not items_path.exists():
        return []
    try:
        items = json.loads(items_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [item for item in items if isinstance(item, dict)]


def _save_verification_items(items: List[dict]) -> None:
    items_path = VERIFICATION_ROOT / "items.json"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = items_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(items_path)


@app.get("/api/verification/items")
async def list_verification_items() -> dict:
    return {"items": _load_verification_items()}


@app.post("/api/verification/items/{item_id}/decision")
async def record_verification_decision(item_id: str, decision: dict = Body(...)) -> dict:
    verdict = str(decision.get("verdict") or "").strip().lower()
    if verdict not in {"match", "mismatch", "uncertain"}:
        raise HTTPException(status_code=400, detail="verdict must be match|mismatch|uncertain")
    items = _load_verification_items()
    for item in items:
        if str(item.get("id")) == item_id:
            item["human_verdict"] = verdict
            item["human_note"] = str(decision.get("note") or "").strip()
            _save_verification_items(items)
            return item
    raise HTTPException(status_code=404, detail=f"Verification item {item_id!r} not found")


@app.get("/api/verification/sheet/{item_id}")
async def get_verification_sheet(item_id: str):
    for item in _load_verification_items():
        if str(item.get("id")) == item_id:
            sheet_rel = str(item.get("sheet") or "")
            if not sheet_rel:
                raise HTTPException(status_code=404, detail="Item has no comparison sheet")
            sheet_path = (VERIFICATION_ROOT / sheet_rel).resolve()
            ensure_within_root(sheet_path, VERIFICATION_ROOT.resolve())
            if not sheet_path.is_file():
                raise HTTPException(status_code=404, detail="Comparison sheet missing")
            return FileResponse(sheet_path, media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Verification item {item_id!r} not found")


def _heartbeat(task_id: str, message: str) -> None:
    """Refresh a task's liveness timestamp during a long synchronous stage."""
    try:
        task_manager.update(task_id, message=message)
    except Exception:
        pass


def _ensure_pdf_or_404(pdf_id: str):
    try:
        return pdf_manager.ensure_pdf(pdf_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"PDF '{pdf_id}' not found")


async def render_pdf_page(pdf_path: Path, page_num: int, zoom: float = 2.0, max_width: Optional[int] = None) -> str:
    if page_num < 1:
        raise HTTPException(status_code=400, detail="Page numbers are 1-based")
    zoom = max(0.1, min(float(zoom), MAX_ZOOM))
    try:
        with fitz.open(pdf_path) as doc:
            if page_num > doc.page_count:
                raise HTTPException(status_code=404, detail="Page not found")
            page = doc[page_num - 1]
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
    except HTTPException:
        raise
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

    # Prefer the verified summary emitted by pipeline.py. Page-level names are
    # raw detector evidence and are only used when no summary was produced.
    for item in diagnostics or []:
        if not isinstance(item, dict):
            continue
        candidates = item.get("detected_assay_names")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            name = str(candidate or "").strip()
            key = re.sub(r"\s+", " ", name.lower()).strip()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
    if names:
        return names

    for item in diagnostics or []:
        if not isinstance(item, dict):
            continue
        page_decision = item.get("llm_page_detection")
        candidates = page_decision.get("assay_names") if isinstance(page_decision, dict) else None
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            name = str(candidate or "").strip()
            key = re.sub(r"\s+", " ", name.lower()).strip()
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
        try:
            pdf_doc = _ensure_pdf_or_404(pdf_id)
        except HTTPException as exc:
            task_manager.update(
                task_id, status="failed", progress=1.0,
                message="Invalid PDF reference", error=str(exc.detail),
            )
            return
        selected_assay_names = [name.strip() for name in (assay_names or []) if name and name.strip()]
        detected_structure_pages: List[int] = []
        detected_assay_pages: List[int] = []
        detected_assay_names: List[str] = []
        output_dir = TASK_OUTPUT_ROOT / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_audit_path = output_dir / "model_calls.jsonl"

        try:
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.04, message="Preparing automatic page detection")

            if detect_structure_pages:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.12, message="Detecting structure pages")
                detected_structure_pages, structure_diagnostics = await _run_interruptible_step(
                    task_id,
                    "Detecting structure pages",
                    lambda: auto_detect_structure_pages_cached(str(pdf_doc.stored_path), audit_path=str(model_audit_path)),
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

                detected_assay_pages, assay_diagnostics = await _run_interruptible_step(
                    task_id,
                    "Detecting bioactivity pages",
                    lambda: auto_detect_assay_pages_cached(
                        str(pdf_doc.stored_path),
                        assay_names=selected_assay_names,
                        progress_callback=assay_detection_progress,
                        audit_path=str(model_audit_path),
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
                    detected_assay_names = await _run_interruptible_step(
                        task_id,
                        "Detecting assay names",
                        lambda: auto_detect_assay_names(
                            str(pdf_doc.stored_path),
                            assay_pages=detected_assay_pages if detected_assay_pages else None,
                        ),
                    )
                if len(detected_assay_names) > 1 and detected_assay_pages:
                    raw_detected_assay_names = list(detected_assay_names)
                    task_manager.update(task_id, progress=0.88, message="Verifying assay names")
                    detected_assay_names = await _run_interruptible_step(
                        task_id,
                        "Verifying assay names",
                        lambda: verify_assay_names_for_pages(
                            str(pdf_doc.stored_path),
                            detected_assay_pages,
                            raw_detected_assay_names,
                            output_dir=str(output_dir),
                            audit_path=str(model_audit_path),
                        ),
                    )
                else:
                    raw_detected_assay_names = list(detected_assay_names)
                _raise_if_task_canceled(task_id)
                task = _get_task_or_404(task_id)
                params = dict(task.params or {})
                params["raw_detected_assay_names"] = raw_detected_assay_names
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
        except TaskStepTimeout as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Automatic detection timed out",
                error=str(exc),
            )
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
    structure_filter_strictness: str,
    auto_detect_pages: bool = False,
) -> None:
    async with task_semaphore, structure_task_semaphore:
        try:
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.05, message="Preparing extraction")
            pdf_doc = _ensure_pdf_or_404(pdf_id)

            output_dir = TASK_OUTPUT_ROOT / task_id
            output_dir.mkdir(parents=True, exist_ok=True)

            selected_pages = list(pages or [])

            if auto_detect_pages:
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=0.08, message="Auto-detecting structure pages")
                selected_pages, detection_diagnostics = await _run_interruptible_step(
                    task_id,
                    "Auto-detecting structure pages",
                    lambda: auto_detect_structure_pages_cached(str(pdf_doc.stored_path), audit_path=str(output_dir / "model_calls.jsonl")),
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
                    progress = 0.05 + (current_page / total_pages) * 0.8 
                    task_manager.update(task_id, progress=min(progress, 0.85), message=message)

            def task_runner() -> Optional[pd.DataFrame]:
                return extract_structures(
                    pdf_file=str(pdf_doc.stored_path),
                    structure_pages=selected_pages,
                    output_dir=str(output_dir),
                    structure_filter_strictness=structure_filter_strictness,
                    progress_callback=progress_callback,
                    audit_path=str(output_dir / "model_calls.jsonl"),
                )

            df = await _run_interruptible_step(task_id, "Extracting structures", task_runner)
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, progress=0.85, message="Post-processing results")
            csv_path = output_dir / "structures.csv"
            filtered_csv_path = output_dir / "filtered_structures.csv"
            failed_pages = _failed_extraction_pages(output_dir, "structure")
            params = {**(_get_task_or_404(task_id).params or {}), "failed_structure_pages": failed_pages}
            coverage_message = f". Structure extraction failed on {len(failed_pages)} pages" if failed_pages else ""

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
                        + coverage_message
                    ),
                    params=params,
                    data=[],
                    result_path=str(csv_path),
                )
                return

            df = df.fillna("")
            raw_rows = df.to_dict(orient="records")
            records = _normalize_records(raw_rows, output_dir)
            # rewrite the CSV from the raw rows: internal columns such as
            # MOLNEXTR_RAW_MOLBLOCK are training data and must survive on
            # disk; _normalize_records only governs API responses / task.data
            _csv_safe_frame(pd.DataFrame(raw_rows)).to_csv(csv_path, index=False, encoding="utf-8-sig")
            filtered_records = _load_csv_records(filtered_csv_path, output_dir)
            if filtered_records:
                _csv_safe_frame(pd.DataFrame(filtered_records)).to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")
            task_manager.update(
                task_id,
                status="completed",
                progress=1.0,
                message=(
                    f"Extracted {len(records)} structures (strictness: {structure_filter_strictness})"
                    + (f"; filtered {len(filtered_records)} candidates" if filtered_records else "")
                    + coverage_message
                ),
                params=params,
                data=records,
                result_path=str(csv_path),
            )
        except TaskCanceled:
            _mark_task_canceled(task_id)
        except TaskStepTimeout as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Structure extraction timed out",
                error=str(exc),
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
    structure_task_id: Optional[str] = None,
    auto_detect_pages: bool = False,
    auto_detect_assay_names_flag: bool = False,
) -> None:
    async with task_semaphore:
        try:
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.05, message="Preparing assay extraction")
            pdf_doc = _ensure_pdf_or_404(pdf_id)

            output_dir = TASK_OUTPUT_ROOT / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            selected_pages = list(pages or [])
            selected_assay_names = list(assay_names or [])

            compound_id_list = None
            structure_records = None
            if structure_task_id:
                try:
                    structure_task = _get_task_or_404(structure_task_id)
                    if structure_task.status == "completed" and structure_task.type in {"structure_extraction", "full_pipeline"}:
                        structure_csv_path = Path(structure_task.result_path)
                        if structure_csv_path.exists():
                            structures_df = pd.read_csv(structure_csv_path)
                            structure_records = _normalize_records(
                                structures_df.fillna("").to_dict(orient="records"),
                                structure_csv_path.parent,
                            )
                            compound_id_list = assay_subject_ids(structure_records)
                            print(f"Using {len(compound_id_list)} compounds from structure source task for matching")
                            task_manager.update(
                                task_id, 
                                progress=0.08, 
                                message=f"Using {len(compound_id_list)} compounds from structure source for matching"
                            )
                        else:
                            task_manager.update(task_id, progress=0.08, message="Structure file not found, extracting all compounds")
                    else:
                        task_manager.update(task_id, progress=0.08, message="Structure source task not completed, extracting all compounds")
                except Exception as e:
                    print(f"Error loading structure data: {e}")
                    task_manager.update(task_id, progress=0.08, message="Error loading structure data, extracting all compounds")

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

                selected_pages, detection_diagnostics = await _run_interruptible_step(
                    task_id,
                    "Auto-detecting assay pages",
                    lambda: auto_detect_assay_pages(
                        str(pdf_doc.stored_path),
                        assay_names=selected_assay_names,
                        progress_callback=assay_task_detection_progress,
                        audit_path=str(output_dir / "model_calls.jsonl"),
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
                if auto_detect_pages:
                    selected_assay_names = _assay_names_from_detection_diagnostics(detection_diagnostics)
                else:
                    task_manager.update(task_id, progress=0.08, message="Auto-detecting assay names")
                    selected_assay_names = await _run_interruptible_step(
                        task_id,
                        "Auto-detecting assay names",
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

            if len(selected_assay_names) > 1 and selected_pages:
                _raise_if_task_canceled(task_id)
                raw_assay_names = list(selected_assay_names)
                task_manager.update(task_id, progress=0.09, message="Verifying assay names")
                try:
                    selected_assay_names = await _run_interruptible_step(
                        task_id,
                        "Verifying assay names",
                        lambda: verify_assay_names_for_pages(
                            str(pdf_doc.stored_path),
                            selected_pages,
                            raw_assay_names,
                            output_dir=str(output_dir),
                            lang=lang,
                            audit_path=str(output_dir / "model_calls.jsonl"),
                        ),
                    )
                    _raise_if_task_canceled(task_id)
                    task = _get_task_or_404(task_id)
                    next_params = dict(task.params or {})
                    next_params["raw_assay_names"] = raw_assay_names
                    next_params["assay_names"] = selected_assay_names
                    next_params["verified_assay_names"] = selected_assay_names
                    task_manager.update(
                        task_id,
                        params=next_params,
                        message=(
                            f"Verified {len(raw_assay_names)} assay name candidate"
                            f"{'s' if len(raw_assay_names) != 1 else ''} to {len(selected_assay_names)}"
                        ),
                    )
                except AssayOCRPageError:
                    task_manager.update(task_id, message="Assay name verification skipped due to incomplete OCR")

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
                    structure_records=structure_records,
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
            raw_results = await _run_interruptible_step(
                task_id,
                "Extracting bioactivity data",
                lambda: task_runner(compound_id_list),
            )
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, progress=0.9, message="Compiling assay results")
            csv_path = output_dir / "assays.csv"

            records, assay_metadata = _records_from_assay_results(raw_results)
            if records:
                _csv_safe_frame(pd.DataFrame(records)).to_csv(csv_path, index=False, encoding="utf-8-sig")
                _write_assay_metadata(assay_metadata, output_dir / "assay_metadata.txt")
            else:
                pd.DataFrame().to_csv(csv_path, index=False, encoding="utf-8-sig")

            missing_assays = [name for name in selected_assay_names if not (raw_results or {}).get(name)]
            message = f"Extracted assay data for {len(records)} compounds"
            if missing_assays:
                message += f". No records for {len(missing_assays)} requested endpoints"
            failed_pages = _failed_extraction_pages(output_dir, "assay", ASSAY_FAILURE_SCOPES)
            if failed_pages:
                message += f". Assay processing incomplete on {len(failed_pages)} pages"
            params = {
                **(_get_task_or_404(task_id).params or {}),
                "missing_assay_names": missing_assays,
                "failed_assay_pages": failed_pages,
            }
            task_manager.update(
                task_id,
                status="completed",
                progress=1.0,
                message=message,
                params=params,
                data=records,
                result_path=str(csv_path),
            )
        except TaskCanceled:
            _mark_task_canceled(task_id)
        except TaskStepTimeout as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Assay extraction timed out",
                error=str(exc),
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
    async with task_semaphore:
        task_manager.update(task_id, status="running", progress=0.1, message="Preparing data merge")
        
        # 获取结构任务数据
        try:
            structure_task = _get_task_or_404(structure_task_id)
            if structure_task.status != "completed":
                raise ValueError("Structure task must be completed")
            if structure_task.type != "structure_extraction":
                raise ValueError("Invalid structure task type")
        except (HTTPException, ValueError) as exc:
            detail = str(getattr(exc, "detail", exc))
            task_manager.update(
                task_id, status="failed", progress=1.0,
                message="Merge not started", error=detail,
            )
            return
        
        output_dir = TASK_OUTPUT_ROOT / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            structure_csv_path = Path(structure_task.result_path)
            structures_df = pd.read_csv(structure_csv_path, dtype={"COMPOUND_ID": str})
            compound_id_list = assay_subject_ids(structures_df.fillna("").to_dict(orient="records"))
            
            task_manager.update(task_id, progress=0.2, message=f"Loaded {len(structures_df)} structures, extracting assays with structure matching")
            
            # mirror pipeline.py: re-extract assays keyed by structure COMPOUND_IDs
            def task_runner() -> Dict[str, Dict[str, object]]:
                assay_data_dicts = {}
                
                for idx, assay_task_id in enumerate(assay_task_ids):
                    assay_task = _get_task_or_404(assay_task_id)
                    if assay_task.status != "completed":
                        raise ValueError(f"Assay task {assay_task_id} must be completed")
                    if assay_task.type != "bioactivity_extraction":
                        raise ValueError(f"Invalid assay task type: {assay_task_id}")
                    
                    params = assay_task.params
                    pdf_id = assay_task.pdf_id
                    pages = params.get("pages", [])
                    assay_names = params.get("assay_names", [])
                    lang = params.get("lang", DEFAULT_OCR_LANG)
                    
                    pdf_doc = _ensure_pdf_or_404(pdf_id)
                    
                    if assay_names:
                        print(f"Re-extracting assays with shared OCR/chunk path: {assay_names}")
                        batch_results = extract_assays(
                            pdf_file=str(pdf_doc.stored_path),
                            assay_pages=pages,
                            assay_names=assay_names,
                            compound_id_list=compound_id_list,
                            output_dir=str(output_dir),
                            lang=lang,
                            structure_records=structures_df.fillna("").to_dict(orient="records"),
                        )
                        for assay_name, assay_data in (batch_results or {}).items():
                            if assay_data:
                                merge_assay_results(assay_data_dicts.setdefault(assay_name, {}), assay_data)
                    
                    progress = 0.2 + (idx + 1) / len(assay_task_ids) * 0.6
                    task_manager.update(
                        task_id, 
                        progress=progress, 
                        message=f"Re-extracted assay data {idx + 1}/{len(assay_task_ids)} with structure matching"
                    )
                
                return assay_data_dicts
            
            assay_data_dicts = await _run_interruptible_step(
                task_id,
                "Re-extracting assay data with structure matching",
                task_runner,
            )
            
            task_manager.update(task_id, progress=0.9, message="Merging structure and assay data")

            from pipeline import merge_data, synthesize_series_members_stage
            _heartbeat(task_id, "Synthesizing series member structures")
            try:
                structures_df, _series_report = synthesize_series_members_stage(
                    str(output_dir),
                    structures_df=structures_df,
                    audit_path=str(output_dir / "model_calls.jsonl"),
                )
            except Exception as exc:
                print(f"Warning: series member synthesis failed during merge: {exc}")
            _heartbeat(task_id, "Merging structure and assay data")

            merged_csv_path = merge_data(structures_df, assay_data_dicts, str(output_dir))
            _heartbeat(task_id, "Finalizing merged results")
            
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
        except TaskCanceled:
            _mark_task_canceled(task_id)
        except TaskStepTimeout as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Data merge timed out",
                error=str(exc),
            )
        except Exception as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Data merge failed",
                error=str(exc),
            )


MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024




_PATENT_PROXY = os.getenv("PATENT_DOWNLOAD_PROXY", "")


def _normalize_patent_number(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(raw or "")).upper()
    if not re.fullmatch(r"[A-Z]{2}[0-9]{6,}[A-Z]?[0-9]?", text):
        raise HTTPException(status_code=400, detail=(
            f"'{raw}' is not a recognized patent number (expected e.g. WO2024163691A1, US10887749B2)"))
    return text


@app.post("/api/patents/fetch", response_model=UploadPDFResponse)
async def fetch_patent(payload: dict) -> UploadPDFResponse:
    """Download a patent PDF by number (Google Patents) through the
    configured HTTP proxy and register it like a manual upload."""
    number = _normalize_patent_number(payload.get("number", ""))
    import httpx
    proxy = _PATENT_PROXY
    async with httpx.AsyncClient(
            proxy=proxy, timeout=90, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}) as client:
        # Google Patents URLs drop the kind code: /patent/US10887749/en, not
        # /patent/US10887749B2/en. Try the number as given, then stripped.
        candidates = [number]
        stripped = re.sub(r"[A-Z]\d?$", "", number) if re.search(r"[0-9][A-Z]\d?$", number) else number
        if stripped != number:
            candidates.append(stripped)
        page = None
        try:
            for candidate in candidates:
                resp = await client.get(f"https://patents.google.com/patent/{candidate}/en")
                if resp.status_code == 200:
                    page = resp
                    break
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=(
                f"Download proxy {_PATENT_PROXY} unreachable: "
                f"{type(exc).__name__}. Retry when the proxy is back."))
        if page is None:
            raise HTTPException(status_code=404, detail=(
                f"Google Patents has no page for {number}"))
        pdf_url = None
        meta = re.search(r'<meta name="citation_pdf_url" content="([^"]+)"', page.text)
        if meta:
            pdf_url = meta.group(1)
        if not pdf_url:
            raise HTTPException(status_code=404, detail=(
                f"No downloadable PDF found for {number}"))
        pdf_resp = await client.get(pdf_url)
        if pdf_resp.status_code != 200 or not pdf_resp.content.startswith(b"%PDF"):
            raise HTTPException(status_code=502, detail=(
                f"PDF download failed for {number} ({pdf_resp.status_code})"))
        content = pdf_resp.content

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        pdf_doc = pdf_manager.register(tmp_path, filename=f"{number}.pdf")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF for {number}: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)
    return UploadPDFResponse(pdf_id=pdf_doc.id, filename=pdf_doc.filename, total_pages=pdf_doc.total_pages)


@app.post("/api/pdfs", response_model=UploadPDFResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadPDFResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        written = 0
        try:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 500 MB upload limit")
                tmp.write(chunk)
        finally:
            tmp.close()
        tmp_path = Path(tmp.name)
    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        pdf_doc = pdf_manager.register(tmp_path, filename=file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF file: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadPDFResponse(pdf_id=pdf_doc.id, filename=pdf_doc.filename, total_pages=pdf_doc.total_pages)


@app.get("/api/pdfs/{pdf_id}", response_model=UploadPDFResponse)
async def get_pdf(pdf_id: str) -> UploadPDFResponse:
    pdf_id = _ensure_usable_identifier(pdf_id, "PDF ID")
    pdf_doc = _ensure_pdf_or_404(pdf_id)
    return UploadPDFResponse(pdf_id=pdf_doc.id, filename=pdf_doc.filename, total_pages=pdf_doc.total_pages)


@app.get("/api/pdfs/{pdf_id}/file")
async def get_pdf_file(pdf_id: str) -> FileResponse:
    pdf_id = _ensure_usable_identifier(pdf_id, "PDF ID")
    pdf_doc = _ensure_pdf_or_404(pdf_id)
    return FileResponse(pdf_doc.stored_path, media_type="application/pdf",
                        filename=pdf_doc.filename)


@app.get("/api/pdfs/{pdf_id}/search/pages")
def get_pdf_search_pages(
    pdf_id: str, numbers: str = Query(..., max_length=128)
) -> dict:
    from .pdf_search import pages_text

    document = _ensure_pdf_or_404(_ensure_usable_identifier(pdf_id, "PDF ID"))
    try:
        pages = list(dict.fromkeys(int(value) for value in numbers.split(",")))
    except ValueError:
        raise HTTPException(
            status_code=422, detail="Page numbers must be integers"
        ) from None
    if not 1 <= len(pages) <= 8 or any(
        not 1 <= page <= document.total_pages for page in pages
    ):
        raise HTTPException(
            status_code=422, detail="Request between 1 and 8 valid pages"
        )
    return pages_text(
        document.stored_path,
        pages,
        str(getattr(project_constants, "PADDLEOCR_SERVER_URL", "") or ""),
        DEFAULT_OCR_LANG,
    )


@app.get("/api/pdfs/{pdf_id}/search/pages/{page_num}")
def get_pdf_search_page(pdf_id: str, page_num: int, ocr: bool = False) -> dict:
    from .pdf_search import SearchBusy, page_text

    document = _ensure_pdf_or_404(_ensure_usable_identifier(pdf_id, "PDF ID"))
    if not 1 <= page_num <= document.total_pages:
        raise HTTPException(status_code=404, detail="Page not found")
    try:
        return page_text(
            document.stored_path,
            page_num,
            ocr,
            str(getattr(project_constants, "PADDLEOCR_SERVER_URL", "") or ""),
            DEFAULT_OCR_LANG,
        )
    except SearchBusy:
        raise HTTPException(
            status_code=429,
            detail="OCR is busy. Retry shortly.",
            headers={"Retry-After": "2"},
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "PDF search failed: %s page %s", pdf_id, page_num
        )
        raise HTTPException(
            status_code=503, detail="Could not index this page. Retry search."
        ) from None


@app.get("/api/pdfs/{pdf_id}/pages/{page_num}")
async def get_pdf_page(pdf_id: str, page_num: int, zoom: float = 2.0, max_width: Optional[int] = None) -> dict:
    pdf_id = _ensure_usable_identifier(pdf_id, "PDF ID")
    pdf_doc = _ensure_pdf_or_404(pdf_id)
    encoded = await render_pdf_page(pdf_doc.stored_path, page_num, zoom, max_width)
    return {"page": page_num, "image": encoded}


from functools import lru_cache


@lru_cache(maxsize=24)
def _pdf_preview_image(path: str, modified: int, page_num: int) -> bytes:
    with fitz.open(path) as document:
        page = document[page_num - 1]
        scale = min(1.5, 1100 / max(1, page.rect.width))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            output = io.BytesIO()
            image.save(output, format="WEBP", quality=82)
            return output.getvalue()


@app.get("/api/pdfs/{pdf_id}/preview/{page_num}")
def get_pdf_preview(pdf_id: str, page_num: int) -> Response:
    """Readable first paint while the PDF.js module and vector page load."""
    document = _ensure_pdf_or_404(_ensure_usable_identifier(pdf_id, "PDF ID"))
    if not 1 <= page_num <= document.total_pages:
        raise HTTPException(status_code=404, detail="Page not found")
    content = _pdf_preview_image(str(document.stored_path), document.stored_path.stat().st_mtime_ns, page_num)
    return Response(content, media_type="image/webp", headers={"Cache-Control": "private, max-age=86400"})


class ManualRegionRequest(BaseModel):
    page: int
    x: float
    y: float
    w: float
    h: float


@app.post("/api/pdfs/{pdf_id}/regions/recognize")
def recognize_pdf_region(pdf_id: str, payload: ManualRegionRequest) -> dict:
    """Recognize a user-selected region using the shared inference service.

    Coordinates are normalized XYWH in the displayed PDF page, independent
    of browser zoom and device pixel ratio. The result retains source evidence.
    """
    import math
    import uuid
    from utils.api_harness import APIHarness

    pdf_doc = _ensure_pdf_or_404(_ensure_usable_identifier(pdf_id, "PDF ID"))
    values = (payload.x, payload.y, payload.w, payload.h)
    if (not all(math.isfinite(value) for value in values)
            or payload.x < 0 or payload.y < 0 or payload.w < .005 or payload.h < .005
            or payload.x + payload.w > 1.000001 or payload.y + payload.h > 1.000001
            or not 1 <= payload.page <= pdf_doc.total_pages):
        raise HTTPException(status_code=422, detail="Select a valid region inside the PDF page")
    with fitz.open(pdf_doc.stored_path) as document:
        page = document[payload.page - 1]
        bounds = page.rect
        clip = fitz.Rect(payload.x * bounds.width, payload.y * bounds.height,
                         (payload.x + payload.w) * bounds.width,
                         (payload.y + payload.h) * bounds.height)
        # Bound raster size for scanned patents with unusually large pages.
        zoom = min(2.5, 2400 / max(clip.width, clip.height))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        image_bytes = pixmap.tobytes("png")
        page_width, page_height = bounds.width, bounds.height
    with tempfile.TemporaryDirectory(prefix="bci-region-") as temporary:
        crop_path = Path(temporary) / "region.png"
        crop_path.write_bytes(image_bytes)
        base_url = os.environ.get("STRUCTURE_API_BASE_URL", "http://harness-api:8011")
        try:
            client = APIHarness(base_url, timeout=180)
            client.session.trust_env = False
            result = client.image('recognize', crop_path, {}, {})
            candidate_smiles = result.get('smiles')
            pose_block = recover_source_coordinates(candidate_smiles, crop_path) if isinstance(candidate_smiles, str) and candidate_smiles.strip() else None
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Structure recognition unavailable; keep the box and retry") from exc
    smiles = result.get("smiles", "")
    if not isinstance(smiles, str) or not smiles.strip():
        raise HTTPException(status_code=422, detail="No structure recognized. Draw a tighter box around one molecule and retry")
    evidence = pdf_doc.stored_path.parent / "manual_regions" / uuid.uuid4().hex
    evidence.mkdir(parents=True, exist_ok=False)
    segment = evidence / "segment.png"
    segment.write_bytes(image_bytes)
    box_file = evidence / "box.json"
    box_file.write_text(json.dumps({"box": list(clip), "page_width": page_width,
                                    "page_height": page_height, "source": "manual"}))
    return {"COMPOUND_ID": "", "SMILES": smiles, "MOLBLOCK": pose_block or "",
            "STRUCTURE_POSE_SOURCE": "source_image_coordinate_graph_mapping" if pose_block else "source_coordinates_unavailable", "page_num": payload.page,
            "source_pages": str(payload.page), "SEGMENT_FILE": str(segment),
            "BOX_COORDS_FILE": str(box_file), "SOURCE": "manual",
            "REVIEW_STATUS": "pending", "STRUCTURE_MODEL": "molsight",
            "STRUCTURE_MODEL_BRANCH": str((result.get("model") or {}).get("branch") or ""),
            "STRUCTURE_MODEL_POLICY": str((result.get("model") or {}).get("policy") or ""),
            "STRUCTURE_MODEL_SHA256": str((result.get("model") or {}).get("model_sha256") or "")}


@app.post("/api/pdfs/{pdf_id}/regions/measurements")
def parse_manual_measurement_region(pdf_id: str, payload: ManualRegionRequest):
    import math
    import uuid
    from utils.measurement_evidence import parse_measurement_table
    from utils.llm_utils import run_vision_json_task
    from utils.skill_prompt_loader import render_skill_reference
    pdf = _ensure_pdf_or_404(_ensure_usable_identifier(pdf_id, "PDF ID"))
    values = (payload.x, payload.y, payload.w, payload.h)
    if not all(math.isfinite(value) for value in values) or min(payload.x, payload.y) < 0 or min(payload.w, payload.h) < .005 or payload.x + payload.w > 1.000001 or payload.y + payload.h > 1.000001 or not 1 <= payload.page <= pdf.total_pages:
        raise HTTPException(status_code=422, detail="Select a valid table region inside the PDF page")
    folder = pdf.stored_path.parent / "manual_regions" / uuid.uuid4().hex
    with fitz.open(pdf.stored_path) as document:
        page = document[payload.page - 1]
        bounds = page.rect
        clip = fitz.Rect(payload.x*bounds.width, payload.y*bounds.height, (payload.x+payload.w)*bounds.width, (payload.y+payload.h)*bounds.height)
        scale = min(3, 2400/max(clip.width, clip.height))
        image = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).tobytes("png")
    with tempfile.TemporaryDirectory(prefix="bci-user-table-") as temporary:
        crop = Path(temporary) / "table.png"; crop.write_bytes(image)
        try:
            result = run_vision_json_task(task_name="read_measurement_table", image_file=str(crop),
                prompt=render_skill_reference('biocheminsight-vision-models', 'references/read_measurement_table_prompt.md', {}),
                parser=parse_measurement_table, metadata={"scope": "manual_measurement_region", "page": payload.page})
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Table extraction unavailable; keep the box and retry") from exc
    if not result['tables'] or not any(table['rows'] for table in result['tables']):
        raise HTTPException(status_code=422, detail="No result rows found. Include the column headers and compound identifiers in the selection")
    folder.mkdir(parents=True, exist_ok=False)
    crop = folder / "table.png"; crop.write_bytes(image)
    box = folder / "box.json"; box.write_text(json.dumps({"box": list(clip), "page_width": bounds.width, "page_height": bounds.height, "source": "manual_table"}))
    records = []
    for table_index, table in enumerate(result['tables']):
        for row in table['rows']:
            observations = [{**column, "subject_id": row['compound_id'], "value": str(value), "evidence": row['row_evidence'],
                "page": payload.page, "table_index": table_index, "source_region": folder.name} for column, value in zip(table['columns'], row['values']) if value is not None]
            record = {"RECORD_ID": uuid.uuid4().hex, "COMPOUND_ID": row['compound_id'], "SMILES": "", "PAGE_NUM": payload.page,
                "STRUCTURE_TYPE": "measurement_record", "RESULT_ONLY": "true", "BIOACTIVITY_OBSERVATIONS": json.dumps(observations, ensure_ascii=False),
                "RESULT_SOURCE_PAGES": json.dumps([payload.page]), "SOURCE": "manual_table", "SEGMENT_FILE": str(crop), "BOX_COORDS_FILE": str(box),
                "MEASUREMENT_REVIEW_SOURCE": "region_model", "MEASUREMENT_LINK_STATUS": "needs_structure_identity"}
            records.append(record)
    return {"records": records, "gaps": result['gaps']}


@app.post("/api/tasks/auto-detect", response_model=TaskStatusResponse)
async def queue_auto_detect_task(payload: AutoDetectTaskRequest, request: Request) -> TaskStatusResponse:
    if not (payload.detect_structure_pages or payload.detect_assay_pages or payload.detect_assay_names):
        raise HTTPException(status_code=400, detail="Enable at least one automatic detection target")

    pdf_id = _ensure_usable_identifier(payload.pdf_id, "PDF ID")
    _ensure_pdf_or_404(pdf_id)

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
    _ensure_pdf_or_404(pdf_id)

    task = task_manager.create(
        "structure_extraction",
        pdf_id=pdf_id,
        params={
            "pages": pages,
            "structure_filter_strictness": payload.structure_filter_strictness,
            "assemble_markush": payload.assemble_markush,
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
    _ensure_pdf_or_404(pdf_id)

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


def _failed_extraction_pages(output_dir: Path, stage: str, scopes=None) -> list[int]:
    pages = set()
    for path in output_dir.rglob(f"{stage}_extraction_warnings.json"):
        try:
            warnings = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(warnings, list):
            continue
        for entry in warnings:
            if not isinstance(entry, dict) or (scopes is not None and entry.get("scope") not in scopes):
                continue
            affected = entry.get("pages")
            if not isinstance(affected, list):
                affected = [entry.get("page")]
            pages.update(page for page in affected if type(page) is int and page > 0)
    return sorted(pages)


async def launch_full_pipeline_task(
    task_id: str,
    pdf_id: str,
    structure_filter_strictness: str = "strict",
    lang: str = DEFAULT_OCR_LANG,
    assay_names: Optional[List[str]] = None,
) -> None:
    async with task_semaphore, structure_task_semaphore:
        pdf_doc = _ensure_pdf_or_404(pdf_id)
        output_dir = TASK_OUTPUT_ROOT / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            _raise_if_task_canceled(task_id)
            task_manager.update(task_id, status="running", progress=0.02, message="Detecting structure and assay pages")

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

            audit_path = str(output_dir / "model_calls.jsonl")
            (detected_structure_pages, _), (detected_assay_pages, assay_diagnostics) = await _run_parallel_steps(
                _run_interruptible_step(
                    task_id,
                    "Auto-detecting structure pages",
                    lambda: auto_detect_structure_pages_cached(str(pdf_doc.stored_path), audit_path=audit_path),
                ),
                _run_interruptible_step(
                    task_id,
                    "Detecting bioactivity pages",
                    lambda: auto_detect_assay_pages_cached(
                        str(pdf_doc.stored_path),
                        progress_callback=assay_detection_progress,
                        audit_path=audit_path,
                    ),
                ),
            )
            _raise_if_task_canceled(task_id)

            selected_assay_names = [name.strip() for name in (assay_names or []) if name and name.strip()]
            raw_detected_assay_names = selected_assay_names or _assay_names_from_detection_diagnostics(assay_diagnostics)
            if not raw_detected_assay_names and detected_assay_pages:
                task_manager.update(task_id, progress=0.12, message="Detecting assay names")
                raw_detected_assay_names = await _run_interruptible_step(
                    task_id,
                    "Detecting assay names",
                    lambda: auto_detect_assay_names(str(pdf_doc.stored_path), assay_pages=detected_assay_pages),
                )

            verify_names = len(raw_detected_assay_names) > 1 and bool(detected_assay_pages)
            page_markdowns = None
            if verify_names:
                task_manager.update(task_id, progress=0.13, message="Reading assay source pages")
                try:
                    page_markdowns = await _run_interruptible_step(
                        task_id,
                        "Reading assay source pages",
                        lambda: load_assay_name_sources(
                            str(pdf_doc.stored_path), detected_assay_pages, str(output_dir), lang
                        ),
                    )
                except AssayOCRPageError:
                    verify_names = False

            async def resolve_assay_names():
                if not verify_names:
                    return raw_detected_assay_names
                return await _run_interruptible_step(
                    task_id,
                    "Verifying assay names",
                    lambda: verify_assay_names_for_pages(
                        str(pdf_doc.stored_path),
                        detected_assay_pages,
                        raw_detected_assay_names,
                        audit_path=audit_path,
                        page_markdowns=page_markdowns,
                    ),
                )

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
                    structure_filter_strictness=structure_filter_strictness,
                    progress_callback=structure_progress_callback,
                    audit_path=str(output_dir / "model_calls.jsonl"),
                )

            detected_assay_names, structures_df = await _run_parallel_steps(
                resolve_assay_names(),
                _run_interruptible_step(task_id, "Extracting structures", structure_runner),
            )
            _raise_if_task_canceled(task_id)
            task_manager.update(
                task_id,
                params={
                    "detected_structure_pages": detected_structure_pages,
                    "detected_assay_pages": detected_assay_pages,
                    "raw_detected_assay_names": raw_detected_assay_names,
                    "detected_assay_names": detected_assay_names,
                },
            )

            structure_records = []
            compound_id_list = []
            csv_path = output_dir / "structures.csv"
            if structures_df is not None and not structures_df.empty:
                structures_df = structures_df.fillna("")
                raw_rows = structures_df.to_dict(orient="records")
                structure_records = _normalize_records(raw_rows, output_dir)
                compound_id_list = assay_subject_ids(structure_records)
                _csv_safe_frame(pd.DataFrame(raw_rows)).to_csv(csv_path, index=False, encoding="utf-8-sig")
                filtered_csv_path = output_dir / "filtered_structures.csv"
                filtered_records = _load_csv_records(filtered_csv_path, output_dir)
                if filtered_records:
                    _csv_safe_frame(pd.DataFrame(filtered_records)).to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")
            else:
                pd.DataFrame(columns=["COMPOUND_ID", "PAGE_NUM", "SMILES"]).to_csv(
                    csv_path, index=False, encoding="utf-8-sig"
                )

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
                    structure_records=structure_records,
                )

            assay_results = await _run_interruptible_step(task_id, "Extracting bioactivity data", assay_runner)
            _raise_if_task_canceled(task_id)

            assay_records = []
            if assay_results:
                assay_records, assay_metadata = _records_from_assay_results(assay_results)
                _csv_safe_frame(pd.DataFrame(assay_records)).to_csv(output_dir / "assays.csv", index=False, encoding="utf-8-sig")
                _write_assay_metadata(assay_metadata, output_dir / "assay_metadata.txt")

            if assay_results and detected_assay_names:
                from pipeline import synthesize_series_members_stage

                _heartbeat(task_id, "Synthesizing series member structures")
                try:
                    structures_frame = pd.read_csv(output_dir / "structures.csv") if (output_dir / "structures.csv").exists() else None
                    structures_frame, _series_report = synthesize_series_members_stage(
                        str(output_dir),
                        structures_df=structures_frame,
                        audit_path=str(output_dir / "model_calls.jsonl"),
                    )
                    if structures_frame is not None:
                        structure_records = structures_frame.to_dict(orient="records")
                except Exception as exc:
                    print(f"Warning: series member synthesis failed for task {task_id}: {exc}")

            _raise_if_task_canceled(task_id)
            task_manager.update(
                task_id,
                progress=0.92,
                message=f"Extracted {len(assay_records)} assay records",
            )

            _raise_if_task_canceled(task_id)
            missing_assays = [name for name in detected_assay_names if not (assay_results or {}).get(name)]
            failed_structure_pages = _failed_extraction_pages(output_dir, "structure")
            failed_assay_pages = _failed_extraction_pages(output_dir, "assay", ASSAY_FAILURE_SCOPES)
            outcome = "Pipeline finished with incomplete coverage" if failed_structure_pages or failed_assay_pages else "Pipeline complete"
            message = f"{outcome}: {len(structure_records)} structures, {len(assay_records)} assay records"
            if failed_structure_pages:
                message += f". Structure extraction failed on {len(failed_structure_pages)} pages"
            if failed_assay_pages:
                message += f". Assay processing incomplete on {len(failed_assay_pages)} pages"
            if missing_assays:
                message += f". No records for {len(missing_assays)} detected endpoints"
            task_manager.update(
                task_id,
                status="completed",
                progress=1.0,
                message=message,
                result_path=str(output_dir / "structures.csv"),
                params={
                    "detected_structure_pages": detected_structure_pages,
                    "detected_assay_pages": detected_assay_pages,
                    "raw_detected_assay_names": raw_detected_assay_names,
                    "detected_assay_names": detected_assay_names,
                    "structure_task_id": task_id,
                    "structure_records_count": len(structure_records),
                    "assay_records_count": len(assay_records),
                    "missing_assay_names": missing_assays,
                    "failed_structure_pages": failed_structure_pages,
                    "failed_assay_pages": failed_assay_pages,
                },
            )
        except TaskCanceled:
            _mark_task_canceled(task_id)
        except TaskStepTimeout as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Full pipeline timed out",
                error=str(exc),
            )
        except Exception as exc:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Full pipeline failed",
                error=str(exc),
            )


@app.get("/api/capabilities")
def platform_capabilities():
    from utils.structure_mode import markush_assembly_enabled
    return {"markush_assembly": {"available": markush_assembly_enabled(), "requires_review": True}}


@app.post("/api/tasks/full-pipeline", response_model=TaskStatusResponse)
async def queue_full_pipeline_task(payload: FullPipelineRequest, request: Request) -> TaskStatusResponse:
    pdf_id = _ensure_usable_identifier(payload.pdf_id, "PDF ID")
    _ensure_pdf_or_404(pdf_id)
    task = task_manager.create(
        "full_pipeline",
        pdf_id=pdf_id,
        params={
            "structure_filter_strictness": payload.structure_filter_strictness,
            "assemble_markush": payload.assemble_markush,
            "lang": payload.lang,
            "assay_names": payload.assay_names or [],
        },
    )
    _enqueue_work_item(
        task,
        "full_pipeline",
        _request_partition_id(request),
        args=[pdf_id, payload.structure_filter_strictness, payload.lang, payload.assay_names],
    )
    queued = _get_task_or_404(task.id)
    return TaskStatusResponse(**queued.to_dict())


@app.get("/api/tasks", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=5, le=100),
    limit: Optional[int] = Query(default=None, ge=1, le=500),
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
        page_size = max(1, min(int(limit or 20), 500))

    tasks = [task for task in task_manager.list_summaries() if task.get("status") != "deleted"]
    queue_positions = get_queue_positions()
    running_count = sum(1 for task in tasks if task.get("status") == "running")
    pending_count = sum(1 for task in tasks if task.get("status") == "pending")

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
                    str(task.get("task_id") or ""),
                    str(task.get("type") or ""),
                    str(task.get("status") or ""),
                    str(task.get("message") or ""),
                    str(task.get("pdf_id") or ""),
                    str(_resolve_task_filename(task) or ""),
                    str(task.get("error") or ""),
                ]
            ).lower()
        ]
    if normalized_status != "all":
        tasks = [task for task in tasks if task.get("status") == normalized_status]
    if normalized_type != "all":
        tasks = [task for task in tasks if task.get("type") == normalized_type]

    date_start = None
    date_end = None
    if date_from:
        try:
            date_start = datetime.combine(datetime.fromisoformat(date_from).date(), datetime.min.time())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from")
    if date_to:
        try:
            date_end = datetime.combine(datetime.fromisoformat(date_to).date(), datetime.max.time())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to")

    def parse_task_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if value:
            try:
                return datetime.fromisoformat(str(value))
            except ValueError:
                pass
        return datetime.min

    if date_start is not None:
        tasks = [task for task in tasks if parse_task_datetime(task.get("updated_at")) >= date_start]
    if date_end is not None:
        tasks = [task for task in tasks if parse_task_datetime(task.get("updated_at")) <= date_end]

    allowed_sort_fields = {"task_id", "created_at", "updated_at", "type", "status", "progress", "queue_position"}
    normalized_sort_by = sort_by if sort_by in allowed_sort_fields else "updated_at"
    normalized_sort_dir = "asc" if str(sort_dir).lower() == "asc" else "desc"

    def sort_value(task: Dict[str, Any]) -> Any:
        if normalized_sort_by == "task_id":
            return task.get("task_id") or ""
        if normalized_sort_by == "queue_position":
            return queue_positions.get(str(task.get("task_id") or "")) or 10**9
        value = task.get(normalized_sort_by)
        if normalized_sort_by in {"created_at", "updated_at"} and value:
            return parse_task_datetime(value).timestamp()
        if value is None:
            return ""
        return value

    tasks = sorted(tasks, key=sort_value, reverse=normalized_sort_dir == "desc")

    total_count = len(tasks)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)
    start = (page - 1) * page_size
    page_tasks = tasks[start : start + page_size]

    payload = []
    for task in page_tasks:
        item = dict(task)
        item["params"] = {}
        item["queue_position"] = queue_positions.get(str(item.get("task_id") or ""))
        item["filename"] = _resolve_task_filename(item)
        payload.append(TaskStatusResponse(**item))
    revision_source = "|".join(
        [
            str(total_count),
            str(running_count),
            str(pending_count),
            str(page),
            str(page_size),
            *[
                ":".join(
                    [
                        task.task_id,
                        task.status,
                        str(task.progress),
                        task.message,
                        task.updated_at,
                        str(task.queue_position or ""),
                    ]
                )
                for task in payload
            ],
        ]
    )
    revision = hashlib.sha1(revision_source.encode("utf-8")).hexdigest()
    return TaskListResponse(
        tasks=payload,
        revision=revision,
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
    return _task_status_response(task, compact_params=task.type == "full_pipeline")


@app.post("/api/tasks/{task_id}/cancel", response_model=TaskStatusResponse)
async def cancel_task(task_id: str) -> TaskStatusResponse:
    task = _get_task_or_404(task_id)
    if task.status in {"completed", "failed", "canceled"}:
        return TaskStatusResponse(**task.to_dict())

    if task.status == "pending":
        cancel_queued_task(task_id)
        updated = task_manager.update(task_id, status="canceled", progress=1.0, message="Canceled", error=None)
        return TaskStatusResponse(**(updated or task).to_dict())

    updated = task_manager.update(task_id, status="canceled", progress=1.0, message="Cancel requested" if task.status == "running" else "Canceled", error=None)
    # Running tasks are isolated child processes. Keep the inflight slot until
    # the Celery parent observes cancellation and terminates the child, avoiding
    # a brief concurrency oversubscription while cleanup is still in progress.
    return TaskStatusResponse(**(updated or task).to_dict())


@app.get("/api/tasks/{task_id}/structures", response_model=StructuresResultResponse)
async def get_task_structures(task_id: str) -> StructuresResultResponse:
    task = _get_task_or_404(task_id)
    if task.type not in {"structure_extraction", "full_pipeline"}:
        raise HTTPException(status_code=400, detail="Task does not contain structure data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    output_dir = Path(task.result_path).parent if task.result_path else (TASK_OUTPUT_ROOT / task.id)
    if task.type == "full_pipeline":
        csv_path = Path(task.result_path) if task.result_path else (output_dir / "structures.csv")
        records = _load_csv_records(csv_path, output_dir)
        if not csv_path.exists():
            raw_records = task.params.get("structure_records") if isinstance(task.params, dict) else []
            records = _normalize_records(list(raw_records or []), output_dir)
        filtered_records = _load_csv_records(output_dir / "filtered_structures.csv", output_dir)
    else:
        records = _normalize_records(list(task.data or []), output_dir)
        filtered_records = _load_csv_records(_get_filtered_structures_csv_path(task), output_dir)
    records = _normalize_records(records, output_dir, with_record_ids=True)
    markush_relationships = _load_markush_relationships(_get_markush_relationships_path(task))
    return StructuresResultResponse(
        task=_task_status_response(task, compact_params=task.type == "full_pipeline"),
        records=records,
        filtered_records=filtered_records,
        markush_relationships=markush_relationships,
    )


@app.get("/api/tasks/{task_id}/assays", response_model=AssayResultResponse)
async def get_task_assays(task_id: str) -> AssayResultResponse:
    task = _get_task_or_404(task_id)
    if task.type not in {"bioactivity_extraction", "full_pipeline"}:
        raise HTTPException(status_code=400, detail="Task does not contain assay data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    if task.type == "full_pipeline":
        output_dir = Path(task.result_path).parent if task.result_path else (TASK_OUTPUT_ROOT / task.id)
        records = _load_csv_records(output_dir / "assays.csv", output_dir)
        if not records:
            records = task.params.get("assay_records") if isinstance(task.params, dict) else []
    else:
        records = task.data or []
    return AssayResultResponse(task=_task_status_response(task, compact_params=task.type == "full_pipeline"), records=records or [])


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
    # Serialize the revision check and write across web processes.
    import fcntl
    folder = _task_output_dir(_get_task_or_404(task_id))
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.review-write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _commit_structure_edit(task_id, payload)


def _commit_structure_edit(task_id: str, payload: UpdateStructuresRequest) -> StructuresResultResponse:
    task = _get_task_or_404(task_id)
    if task.type not in {"structure_extraction", "full_pipeline"}:
        raise HTTPException(status_code=400, detail="Task does not contain structure data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")

    current_revision = str((task.params or {}).get("records_revision") or "0")
    if payload.base_revision is not None and payload.base_revision != current_revision:
        raise HTTPException(status_code=409, detail="These records changed in another session. Reload before applying your edit.")
    records = payload.records
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="Records must be a list")

    from utils.measurement_evidence import observation_values, user_measurement_edits
    records = [{**row, **observation_values(row), **user_measurement_edits(row)} for row in records]
    csv_path = Path(task.result_path or (TASK_OUTPUT_ROOT / task_id / "structures.csv"))
    filtered_csv_path = _get_filtered_structures_csv_path(task)
    import uuid
    revision = uuid.uuid4().hex
    revision_dir = csv_path.parent / "record_revisions"
    revision_dir.mkdir(parents=True, exist_ok=True)
    previous_records = _load_csv_records(csv_path, csv_path.parent) if csv_path.exists() else list(task.data or [])
    revision_path = revision_dir / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}_{revision}.json"
    from utils.review_audit import changes, digest
    previous_records = _normalize_records(previous_records, csv_path.parent, with_record_ids=True)
    records = _normalize_records(records, csv_path.parent, with_record_ids=True)
    ids = [row['RECORD_ID'] for row in records]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="Each record must have a unique identity")
    record_changes = changes(previous_records, records)
    if not record_changes:
        return StructuresResultResponse(
            task=TaskStatusResponse(**task.to_dict()),
            records=previous_records,
            filtered_records=_load_csv_records(filtered_csv_path, csv_path.parent),
            markush_relationships=_load_markush_relationships(_get_markush_relationships_path(task)),
        )
    session = str(payload.actor_session or '')[:128]
    actor = ({"kind": "browser_session", "label": "Browser " + hashlib.sha256(session.encode()).hexdigest()[:8]}
             if session else {"kind": "unknown", "label": "API · author not recorded"})
    event = {"revision": revision, "base_revision": current_revision,
        "saved_at": datetime.utcnow().isoformat() + "Z", "records": previous_records,
        "after_records": records, "changes": record_changes, "actor": actor,
        "reason": payload.reason[:200], "restore_of": payload.restore_of,
        "before_sha256": digest(previous_records), "after_sha256": digest(records)}
    pending_event = revision_path.with_suffix('.tmp')
    pending_event.write_text(json.dumps(event, ensure_ascii=False))
    original_csv = csv_path.read_bytes() if csv_path.exists() else None
    # Preserve headers when the last record is deleted.
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=["COMPOUND_ID", "SMILES"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=csv_path.parent, suffix=".csv", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        _csv_safe_frame(df).to_csv(temporary_path, index=False, encoding="utf-8-sig")
        temporary_path.replace(csv_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    params = dict(task.params or {})
    params["records_revision"] = revision
    if task.type == "full_pipeline":
        params["structure_records_count"] = len(records)
        if "structure_records" in params:
            params["structure_records"] = records
        # This is a derived file; the next export must merge the edited rows.
        (csv_path.parent / "merged.csv").unlink(missing_ok=True)
    try:
        task_manager.update(task_id, data=records, params=params,
                            result_path=str(csv_path), message="Structures updated")
    except Exception:
        # Keep the export and revision pointer consistent when persistence fails.
        rollback_path = csv_path.with_suffix('.rollback.tmp')
        if original_csv is None:
            csv_path.unlink(missing_ok=True)
        else:
            rollback_path.write_bytes(original_csv)
            rollback_path.replace(csv_path)
        raise
    # Publish the event after task metadata commits.
    pending_event.replace(revision_path)
    updated = _get_task_or_404(task_id)
    try:
        _write_sar_review_report(updated, records)
    except Exception as exc:
        print(f"SAR report refresh failed after save: {type(exc).__name__}")
    filtered_records = _load_csv_records(filtered_csv_path, csv_path.parent)
    return StructuresResultResponse(
        task=TaskStatusResponse(**updated.to_dict()),
        records=records,
        filtered_records=filtered_records,
        markush_relationships=_load_markush_relationships(_get_markush_relationships_path(updated)),
    )


def _write_sar_review_report(task, records):
    from utils.sar_harness import save_sar_report
    folder = _task_output_dir(task)
    try:
        coverage = json.loads((folder / "coverage_report.json").read_text())
    except (OSError, ValueError):
        coverage = {}
    params = task.params or {}
    pages = params.get("detected_assay_pages") if "detected_assay_pages" in params else params.get("assay_pages")
    if pages is not None and not isinstance(pages, list):
        pages = parse_pages_input(str(pages), None)
    return save_sar_report(folder, records, page_entities=coverage.get("page_entities"), result_pages=pages)


@app.get("/api/tasks/{task_id}/sar")
def get_sar_review_report(task_id: str):
    task = _get_task_or_404(task_id)
    records = _structure_records_for_task(task)
    return _write_sar_review_report(task, records)


def _structure_records_for_task(task):
    path = Path(task.result_path) if task.result_path else _task_output_dir(task) / "structures.csv"
    return _load_csv_records(path, path.parent) if path.exists() else list(task.data or [])


def _review_events(task):
    from utils.review_audit import read_events
    path = Path(task.result_path) if task.result_path else _task_output_dir(task) / "structures.csv"
    revision_dir = path.parent / "record_revisions"
    committed = str((task.params or {}).get('records_revision') or '0')
    for pending in revision_dir.glob('*.tmp'):
        try:
            if json.loads(pending.read_text()).get('revision') == committed:
                pending.replace(pending.with_suffix('.json'))
        except (OSError, ValueError): pass
    events = read_events(path.parent / "record_revisions", _structure_records_for_task(task))
    from utils.review_audit import changes
    for event in events:
        event['records'] = _normalize_records(event['records'], path.parent, with_record_ids=True)
        event['after_records'] = _normalize_records(event['after_records'], path.parent, with_record_ids=True)
        event['changes'] = changes(event['records'], event['after_records'])
    return [event for event in events if event['changes']]


@app.get("/api/tasks/{task_id}/structures/history")
def structure_revision_history(task_id: str, record_id: Optional[str] = None):
    task = _get_task_or_404(task_id)
    entries = []
    for item in reversed(_review_events(task)):
        if record_id and not any(c['record_id'] == record_id for c in item['changes']): continue
        entries.append({key: item.get(key) for key in ("revision", "base_revision", "saved_at", "reason", "actor", "restore_of", "changes")})
    return {"revisions": entries, "current_revision": str((task.params or {}).get("records_revision") or "0")}


@app.post("/api/tasks/{task_id}/structures/restore", response_model=StructuresResultResponse)
async def restore_structure_revision(task_id: str, payload: dict = Body(...)):
    from utils.review_audit import restore_records
    if not isinstance(payload.get('base_revision'), str):
        raise HTTPException(status_code=400, detail="A current revision is required")
    task = _get_task_or_404(task_id)
    event = next((e for e in _review_events(task) if e['revision'] == payload.get('revision')), None)
    if event is None: raise HTTPException(status_code=404, detail="Revision not found")
    try:
        records = restore_records(_structure_records_for_task(task), event,
            scope=payload.get('scope', 'version'), record_id=payload.get('record_id'),
            field=payload.get('field'), side=payload.get('side', 'before'))
    except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc))
    return await update_task_structures(task_id, UpdateStructuresRequest(records=records,
        base_revision=payload['base_revision'], actor_session=payload.get('actor_session'),
        reason='restore_' + payload.get('scope', 'version'), restore_of=event['revision']))


@app.post("/api/tasks/{task_id}/structures/undo", response_model=StructuresResultResponse)
async def undo_last_structure_edit(task_id: str, payload: dict = Body(default={})):
    events = _review_events(_get_task_or_404(task_id))
    if not events: raise HTTPException(status_code=404, detail="No saved edit to restore")
    return await restore_structure_revision(task_id, {**payload, 'revision': events[-1]['revision'], 'side': 'before'})


class SARRecoveryRequest(BaseModel):
    max_actions: int = 8


@app.post("/api/tasks/{task_id}/sar/recover", response_model=TaskStatusResponse)
async def queue_sar_recovery(task_id: str, payload: SARRecoveryRequest, request: Request):
    source = _get_task_or_404(task_id)
    if source.status != "completed" or source.type not in {"structure_extraction", "full_pipeline"}:
        raise HTTPException(status_code=409, detail="Open a completed structure task before recovery")
    if not 1 <= payload.max_actions <= 16:
        raise HTTPException(status_code=422, detail="Recovery budget must be between 1 and 16 actions")
    task = task_manager.create("structure_extraction", pdf_id=source.pdf_id,
        params={"source_task_id": task_id, "goal": "SAR evidence recovery", "max_actions": payload.max_actions, "assemble_markush": bool((source.params or {}).get("assemble_markush", True))})
    output = TASK_OUTPUT_ROOT / task.id
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_records.json").write_text(json.dumps(_structure_records_for_task(source), ensure_ascii=False))
    _enqueue_work_item(task, "sar_recovery", _request_partition_id(request), args=[task_id, payload.max_actions])
    return TaskStatusResponse(**_get_task_or_404(task.id).to_dict())


async def launch_sar_recovery_task(task_id: str, source_task_id: str, max_actions: int):
    async with task_semaphore:
        source = _get_task_or_404(source_task_id)
        pdf = _ensure_pdf_or_404(source.pdf_id)
        folder = TASK_OUTPUT_ROOT / task_id
        try:
            records = json.loads((folder / "source_records.json").read_text())
            from utils.sar_harness import new_recovery_session
            source_report = new_recovery_session(_write_sar_review_report(source, records))
            try:
                coverage = json.loads((_task_output_dir(source) / "coverage_report.json").read_text())
            except (OSError, ValueError): coverage = {}
            (folder / "sar_harness.json").write_text(json.dumps(source_report, ensure_ascii=False))
            # Old inventories may have interpreted result cells as sequences or
            # inferred names. Refresh their evidence kinds before planning.
            inventory = coverage.get("page_entities") or []
            inventory_cache = folder / "recovery_page_inventory.json"
            if inventory and any(int(item.get("inventory_version") or 0) < 2 for item in inventory):
                if inventory_cache.exists():
                    inventory = json.loads(inventory_cache.read_text())
                else:
                    from utils.page_extractor import PageEntityExtractor
                    task_manager.update(task_id, status="running", progress=.03, message="Refreshing document evidence inventory")
                    inventory = await _run_interruptible_step(task_id, "Refreshing evidence inventory",
                        lambda: PageEntityExtractor().extract_pages(str(pdf.stored_path), [int(item['page_num']) for item in inventory]))
                    inventory_cache.write_text(json.dumps(inventory, ensure_ascii=False))
                coverage["page_entities"] = inventory
            result_pages = (source.params or {}).get("detected_assay_pages")
            if result_pages is not None: result_pages = list(result_pages)
            task_manager.update(task_id, status="running", progress=.05, message="Inspecting document-wide SAR evidence gaps")
            from utils.sar_harness import recover_sar_rows
            def progress(current, total, action):
                _raise_if_task_canceled(task_id)
                task_manager.update(task_id, progress=min(.95, .05 + .85*current/max(1,total)), message=f"Recovering evidence: {action['operation']}")
            updated, report = await _run_interruptible_step(task_id, "Recovering SAR evidence",
                lambda: recover_sar_rows(records, str(pdf.stored_path), str(folder), page_entities=coverage.get("page_entities"),
                    result_pages=result_pages, max_actions=max_actions, audit_path=str(folder / "model_calls.jsonl"), progress=progress))
            _raise_if_task_canceled(task_id)
            path = folder / "structures.csv"
            _csv_safe_frame(pd.DataFrame(updated)).to_csv(path, index=False, encoding="utf-8-sig")
            if coverage: (folder / "coverage_report.json").write_text(json.dumps(coverage, ensure_ascii=False))
            params = dict(_get_task_or_404(task_id).params or {})
            params['sar_open_gaps'] = report['coverage']['open_gaps']
            if report.get('result_pages') is not None: params['detected_assay_pages'] = report['result_pages']
            blocked = report.get('execution_status') == 'blocked_provider'
            task_manager.update(task_id, status="failed" if blocked else "completed", progress=1, data=updated, result_path=str(path), params=params,
                error="Model planning service unavailable; partial records and checkpoint were preserved" if blocked else None,
                message="Recovery stopped: model service unavailable; saved evidence preserved" if blocked else f"Recovery finished: {len(updated)} records, {report['coverage']['open_gaps']} evidence gaps require review")
        except TaskCanceled:
            _mark_task_canceled(task_id)
        except Exception as exc:
            task_manager.update(task_id, status="failed", progress=1, message="SAR recovery failed", error=str(exc))


# --- Manual annotation: debug-mode training-data collection ---

def _task_output_dir(task) -> Path:
    return Path(task.result_path).parent if task.result_path else (TASK_OUTPUT_ROOT / task.id)


def _annotation_artifact_path(value: str, task_dir: Path) -> str:
    """Resolve a structure artifact path recorded in task outputs.

    Values may be container-absolute, host-absolute (backfilled tasks) or
    task-relative; anything under structures_group_N is remapped into the
    task directory, where a copy always exists.
    """
    if not value:
        return value
    marker = "structures_group_"
    idx = value.find(marker)
    if idx != -1:
        candidate = task_dir / value[idx:]
        if candidate.exists():
            return str(candidate)
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    alt = (task_dir / candidate).resolve()
    if alt.exists():
        return str(alt)
    return value


def _load_annotation_items(task) -> List[dict]:
    import csv as _csv
    output_dir = _task_output_dir(task)
    # raw rows, NOT _normalize_records: the debug view needs MOLNEXTR_RAW_MOLBLOCK,
    # which the API strip list hides, and paths are resolved per item below
    if task.type == "full_pipeline":
        csv_path = Path(task.result_path) if task.result_path else (output_dir / "structures.csv")
        records = []
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8-sig") as fh:
                records = [dict(r) for r in _csv.DictReader(fh)]
    else:
        records = list(task.data or [])
    items: List[dict] = []
    seen_keys = set()
    for idx, rec in enumerate(records):
        if str(rec.get("STRUCTURE_TYPE") or "") != "complete_compound":
            continue
        smiles = str(rec.get("SMILES") or "").strip()
        if not smiles or str(rec.get("FILTERED_OUT") or "").lower() == "true":
            continue
        segment = str(rec.get("SEGMENT_FILE") or rec.get("IMAGE_FILE") or "")
        row_key = os.path.basename(segment) if segment else "row_%d" % idx
        if row_key in seen_keys:
            row_key = "%s_%d" % (row_key, idx)
        seen_keys.add(row_key)
        items.append({
            "row_key": row_key,
            "kind": "complete",
            "compound_id": str(rec.get("COMPOUND_ID") or ""),
            "smiles": smiles,
            # debug mode shows the recognizer's own depiction, never the
            # optimized layout; fall back for tasks parsed before the raw
            # column existed
            "molblock": str(rec.get("MOLNEXTR_RAW_MOLBLOCK") or rec.get("MOLBLOCK") or ""),
            "segment_file": _annotation_artifact_path(segment, output_dir),
            "page": rec.get("PAGE_NUM"),
        })
    for entry in _load_markush_relationships(_get_markush_relationships_path(task)):
        for cand in entry.get("structure_candidates") or []:
            molblock = str(cand.get("molblock_full") or cand.get("molblock") or "")
            smiles = str(cand.get("smiles") or "").strip()
            ref = str(cand.get("ref") or "")
            if not ref or (not molblock and not smiles):
                continue
            items.append({
                "row_key": "raw_%s" % ref,
                "kind": "raw_%s" % (str(cand.get("structure_type") or "candidate")),
                "compound_id": str(cand.get("markush_cell") or ""),
                "smiles": smiles,
                # candidates' molblock_full may be a VLM-correction CoordGen
                # relayout; the raw snapshot keeps the recognizer pose
                "molblock": str(cand.get("molblock_molnextr_raw") or molblock),
                "segment_file": _annotation_artifact_path(
                    str(cand.get("segment_file") or ""), output_dir),
                "page": cand.get("page"),
            })
    return items


def _annotations_path(task) -> Path:
    return _task_output_dir(task) / "annotations.json"


def _load_annotations(task) -> Dict[str, dict]:
    path = _annotations_path(task)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("annotations") or {}
    except (OSError, ValueError) as exc:
        # keep the damaged file for inspection instead of silently treating
        # it as empty (a subsequent save would destroy the history)
        corrupt = path.with_name(path.name + ".corrupt-%d" % int(time.time()))
        try:
            os.replace(path, corrupt)
        except OSError:
            pass
        raise HTTPException(status_code=500,
                            detail="annotations.json unreadable; moved to %s (recover "
                                   "from data/annotations/all.jsonl)" % corrupt.name) from exc


def _save_annotations(task, annotations: Dict[str, dict]) -> None:
    path = _annotations_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"annotations": annotations}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def _smiles_from_molblock(molblock: str) -> str:
    """Canonical SMILES derived from the molblock (single source of truth)."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolBlock(molblock)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


_ANNOTATION_BUCKET = {
    "complete": "ordinary_structure",
    "raw_markush": "markush_layout",
    "raw_fragment": "attachment_fragment",
    "raw_text_substituent": "markush_layout",
}





@app.get("/api/tasks/{task_id}/annotation-items", response_model=AnnotationItemsResponse)
async def get_annotation_items(task_id: str) -> AnnotationItemsResponse:
    task = _get_task_or_404(task_id)
    if task.type not in {"structure_extraction", "full_pipeline"}:
        raise HTTPException(status_code=400, detail="Task does not contain structure data")
    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    items = _load_annotation_items(task)
    counts = {"complete": 0, "raw": 0}
    for item in items:
        counts["raw" if item["kind"].startswith("raw_") else "complete"] += 1
    return AnnotationItemsResponse(items=items, counts=counts)


@app.get("/api/tasks/{task_id}/annotations", response_model=AnnotationsResponse)
async def get_task_annotations(task_id: str) -> AnnotationsResponse:
    task = _get_task_or_404(task_id)
    return AnnotationsResponse(annotations=_load_annotations(task))


@app.post("/api/tasks/{task_id}/annotations", response_model=AnnotationsResponse)
async def upsert_task_annotation(task_id: str, payload: AnnotationUpsertRequest) -> AnnotationsResponse:
    task = _get_task_or_404(task_id)
    if payload.status not in {"confirmed", "wrong", "corrected"}:
        raise HTTPException(status_code=400, detail="status must be confirmed|wrong|corrected")
    annotations = _load_annotations(task)
    record = dict(annotations.get(payload.row_key) or {})
    if payload.status != "corrected":
        record.pop("corrected_molblock", None)
        record.pop("corrected_smiles", None)
    record.update({
        "row_key": payload.row_key,
        "status": payload.status,
        "original_smiles": payload.original_smiles or record.get("original_smiles"),
        "segment_file": payload.segment_file or record.get("segment_file"),
        "compound_id": payload.compound_id or record.get("compound_id"),
        "kind": payload.kind or record.get("kind"),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
    })
    if payload.status == "corrected":
        # the editor molblock is the single source of truth: JSME opened on
        # the item's molblock, so untouched atoms keep their coordinates and
        # the saved block carries the pose.  The SMILES label is derived
        # from the molblock, never supplied separately.
        if not payload.corrected_molblock:
            raise HTTPException(status_code=400,
                                detail="corrected requires the editor molblock")
        smiles = _smiles_from_molblock(payload.corrected_molblock)
        if not smiles:
            raise HTTPException(status_code=400, detail="editor molblock is not readable")
        record["corrected_molblock"] = payload.corrected_molblock
        record["corrected_smiles"] = smiles
    annotations[payload.row_key] = record
    _save_annotations(task, annotations)
    store = DATA_ROOT / "annotations"
    store.mkdir(parents=True, exist_ok=True)
    with open(store / "all.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"task_id": task.id, **record}, ensure_ascii=False) + "\n")
    return AnnotationsResponse(annotations=annotations)


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str) -> TaskStatusResponse:
    """Delete a task's output directory and registry entries.

    The task record itself is kept with status "deleted" so history stays
    auditable; only the (potentially large) artifacts are removed.
    """
    task = _get_task_or_404(task_id)
    if task.status in ("running", "pending"):
        raise HTTPException(status_code=409, detail="Cancel the task before deleting it")
    output_dir = _task_output_dir(task)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    updated = task_manager.update(task_id, status="deleted", data=None,
                                  message="Task artifacts deleted")
    return TaskStatusResponse(**_get_task_or_404(task_id).to_dict())


@app.get("/api/tasks/{task_id}/annotations/export")
async def export_task_annotations(task_id: str):
    task = _get_task_or_404(task_id)
    items = {item["row_key"]: item for item in _load_annotation_items(task)}
    annotations = _load_annotations(task)
    lines = []
    for row_key, record in annotations.items():
        item = items.get(row_key, {})
        line = {
            "row_key": row_key,
            "label": record.get("status"),
            "source": item.get("kind") or record.get("kind"),
            "structure_type_bucket": _ANNOTATION_BUCKET.get(item.get("kind") or ""),
            "raw_molblock": item.get("molblock"),
            "compound_id": record.get("compound_id"),
            "segment_file": item.get("segment_file") or record.get("segment_file"),
            "page": item.get("page"),
            "original_smiles": item.get("smiles") or record.get("original_smiles"),
            "corrected_smiles": record.get("corrected_smiles"),
            "corrected_molblock": record.get("corrected_molblock"),
            "updated_at": record.get("updated_at"),
        }
        lines.append(json.dumps(line, ensure_ascii=False))
    content = "\n".join(lines) + ("\n" if lines else "")
    return Response(
        content=content,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="%s_annotations.jsonl"' % task_id},
    )


def merge_structure_activity_records(
    structures_df: pd.DataFrame,
    activity_data: List[Dict[str, Any]],
    output_dir: Path,
    filename: str = "merged.csv",
) -> Optional[Path]:
    if "COMPOUND_ID" not in structures_df.columns:
        print("Structure data does not contain COMPOUND_ID")
        return None

    assay_data_dict: Dict[str, Dict[str, Any]] = {}
    for record in activity_data:
        if 'COMPOUND_ID' in record and record['COMPOUND_ID']:
            compound_id = str(record['COMPOUND_ID'])
            assay_values: Dict[str, Any] = {}
            for key, value in record.items():
                if key == 'COMPOUND_ID':
                    continue
                if "__" in str(key):
                    assay_values[str(key)] = _stringify(value)
                    continue
                value_text, metadata = _flatten_assay_value(str(key), value)
                assay_values[_assay_column_name(str(key), str(metadata.get("unit") or ""))] = value_text
            if assay_values:
                assay_data_dict.setdefault(compound_id, {}).update(assay_values)
    
    print(f"Activity data for {len(assay_data_dict)} compounds")
    
    # COMPOUND_ID 转为字符串，防止匹配错误
    merged_df = structures_df.copy()
    merged_df['COMPOUND_ID'] = merged_df['COMPOUND_ID'].astype(str)
    
    all_assay_fields = set()
    for assay_values in assay_data_dict.values():
        all_assay_fields.update(assay_values.keys())
    
    for field in all_assay_fields:
        if field not in merged_df.columns:
            merged_df[field] = None
    
    # Only an unambiguous structure identity can claim an assay row.
    from utils.measurement_evidence import measurement_link_status, split_unverified_measurements

    assay_subjects = pd.Series(
        [is_assay_subject_structure(row) for row in merged_df.to_dict("records")],
        index=merged_df.index,
        dtype=bool,
    )
    assay_only_rows = []
    for compound_id, assay_values in assay_data_dict.items():
        assay_values, unverified = split_unverified_measurements(assay_values)
        mask = assay_subjects & (merged_df['COMPOUND_ID'] == compound_id)
        matches = merged_df.loc[mask]
        status = measurement_link_status(matches.get('SMILES', pd.Series(dtype=str)).fillna(''))
        if status:
            unverified = {**assay_values, **unverified}
            assay_values = {}
        if unverified:
            assay_only_rows.append({'COMPOUND_ID': compound_id, 'SMILES': '', **unverified,
                'MEASUREMENT_LINK_STATUS': status or 'identifier_unverified'})
        for field, value in assay_values.items():
            merged_df.loc[mask, field] = value
    
    if assay_only_rows:
        merged_df = pd.concat([merged_df, pd.DataFrame(assay_only_rows)], ignore_index=True)

    # User corrections take precedence over extracted assay values, including
    # deliberately cleared cells. Keep the original extraction as provenance.
    from utils.measurement_evidence import observation_values, user_measurement_edits
    for index, row in merged_df.iterrows():
        for column, value in {**observation_values(row), **user_measurement_edits(row)}.items():
            merged_df.at[index, column] = value

    from utils.measurement_evidence import removed_measurements
    merged_df = merged_df.drop(columns=list(removed_measurements(merged_df.to_dict('records'))), errors='ignore')
    merged_csv_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    _csv_safe_frame(merged_df).to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    print(f"Merged data saved to {merged_csv_path}")
    
    return merged_csv_path


def merge_structure_activity_data(structure_task_id: str, activity_data: List[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    """
    合并结构数据和活性数据，类似 pipeline.py 的 merge_data 函数
    """
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

    return merge_structure_activity_records(structures_df, activity_data, output_dir)


def _linked_structure_task_id(task: Task) -> str:
    metadata = task.metadata or {}
    params = task.params or {}
    candidates = [
        metadata.get("structure_task_id"),
        params.get("structure_task_id"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _assay_dicts_from_json_files(task_output_dir: Path, assay_names=()) -> Dict[str, Dict[str, Any]]:
    names_by_stem: Dict[str, set[str]] = {}
    for name in assay_names:
        stem = name.replace(" ", "_").replace("/", "_")
        names_by_stem.setdefault(stem, set()).add(name)
    assay_dicts: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(task_output_dir.glob("*_assay_data.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload:
            stem = json_path.stem[: -len("_assay_data")]
            names = names_by_stem.get(stem, {stem})
            if len(names) == 1:
                assay_dicts[next(iter(names))] = payload
    return assay_dicts


def _build_results_zip(task: Task, task_output_dir: Path, merged_csv_path: Optional[Path] = None) -> Path:
    zip_path = task_output_dir / f"{task.id}_results.zip"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    if not (task_output_dir / "assay_metadata.txt").exists():
        _write_assay_metadata({}, task_output_dir / "assay_metadata.txt")

    with tempfile.TemporaryDirectory(dir=task_output_dir) as staging:
        temporary_zip = Path(staging) / zip_path.name
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            written_arcnames: set[str] = set()

            def write_file(path: Path | None, arcname: str) -> None:
                if not path or not path.exists() or not path.is_file():
                    return
                if arcname in written_arcnames:
                    return
                archive.write(path, arcname)
                written_arcnames.add(arcname)

            candidates = [
                (merged_csv_path, "merged_results.csv"),
                (task_output_dir / "structures.csv", "structures.csv"),
                (task_output_dir / "filtered_structures.csv", "filtered_structures.csv"),
                (task_output_dir / "markush_relationships.json", "markush_relationships.json"),
                (task_output_dir / "coverage_report.json", "audit/coverage_report.json"),
                (task_output_dir / "document_ledger.json", "audit/document_ledger.json"),
                (task_output_dir / "assays.csv", "bioactivity.csv"),
                (task_output_dir / "assay_metadata.txt", "assay_metadata.txt"),
                (task_output_dir / "model_calls.jsonl", "audit/model_calls.jsonl"),
                (task_output_dir / "assay_extraction_warnings.json", "audit/assay_extraction_warnings.json"),
                (task_output_dir / "structure_extraction_warnings.json", "audit/structure_extraction_warnings.json"),
            ]
            if task.result_path:
                result_path = Path(task.result_path)
                candidates.append((result_path, result_path.name))
            for path, arcname in candidates:
                write_file(path, arcname)

            if task.data:
                archive.writestr(
                    "raw_task_records.json",
                    json.dumps(task.data, ensure_ascii=False, indent=2),
                )

            for json_path in sorted(task_output_dir.rglob("*assay_data.json")):
                if json_path.is_file():
                    write_file(json_path, f"raw_assay_json/{json_path.relative_to(task_output_dir)}")
            for audit_path in sorted(task_output_dir.rglob("model_calls.jsonl")):
                if audit_path.is_file() and audit_path != task_output_dir / "model_calls.jsonl":
                    write_file(audit_path, f"audit/{audit_path.relative_to(task_output_dir)}")
            for warning_path in sorted(task_output_dir.rglob("*warnings.json")):
                if warning_path.is_file() and warning_path not in {task_output_dir / "assay_extraction_warnings.json", task_output_dir / "structure_extraction_warnings.json"}:
                    write_file(warning_path, f"audit/{warning_path.relative_to(task_output_dir)}")
        temporary_zip.replace(zip_path)
    return zip_path


@app.get("/api/tasks/{task_id}/download")
async def download_task_artifact(task_id: str) -> FileResponse:
    task = _get_task_or_404(task_id)

    if task.status != "completed":
        raise HTTPException(status_code=409, detail="Task has not completed")
    if not task.result_path:
        raise HTTPException(status_code=404, detail="No artifact available")

    if task.type == "full_pipeline":
        task_output_dir = Path(task.result_path).parent
        structures_path = task_output_dir / "structures.csv"
        with tempfile.TemporaryDirectory() as staging:
            merged_csv_path = None
            if structures_path.is_file():
                try:
                    from pipeline import merge_data

                    structures_df = pd.read_csv(structures_path, dtype={"COMPOUND_ID": str})
                    assay_data_dicts = _assay_dicts_from_json_files(
                        task_output_dir, (task.params or {}).get("detected_assay_names") or ()
                    )
                    if not structures_df.empty and assay_data_dicts:
                        merged_csv_path = Path(merge_data(structures_df, assay_data_dicts, staging))
                except Exception as exc:
                    print(f"Warning: full-pipeline merge for download failed: {exc}")
            zip_path = _build_results_zip(task, task_output_dir, merged_csv_path)
        return FileResponse(zip_path, filename="biocheminsight_results.zip", media_type="application/zip")

    if task.type == "bioactivity_extraction":
        structure_task_id = _linked_structure_task_id(task)
        if structure_task_id:
            task_output_dir = Path(task.result_path).parent
            merged_csv_path = merge_structure_activity_data(structure_task_id, task.data or [], task_output_dir)

            if merged_csv_path and merged_csv_path.exists():
                zip_path = _build_results_zip(task, task_output_dir, merged_csv_path)
                return FileResponse(zip_path, filename="biocheminsight_results.zip", media_type="application/zip")

    if task.type == "data_merge":
        csv_path = Path(task.result_path)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Artifact file missing")
        zip_path = _build_results_zip(task, csv_path.parent, csv_path)
        return FileResponse(zip_path, filename="biocheminsight_results.zip", media_type="application/zip")

    csv_path = Path(task.result_path)
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    zip_path = _build_results_zip(task, csv_path.parent, csv_path)
    return FileResponse(zip_path, filename="biocheminsight_results.zip", media_type="application/zip")


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
