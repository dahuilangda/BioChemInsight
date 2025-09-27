from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator

try:  # noqa: SIM105
    from constants import DEFAULT_OCR_ENGINE as CONFIG_DEFAULT_OCR_ENGINE
except ImportError:  # pragma: no cover - optional user configuration
    CONFIG_DEFAULT_OCR_ENGINE = 'paddleocr'

SUPPORTED_OCR_ENGINES = {'paddleocr', 'dots_ocr'}
DEFAULT_OCR_ENGINE = (CONFIG_DEFAULT_OCR_ENGINE or 'paddleocr').strip().lower()
if DEFAULT_OCR_ENGINE not in SUPPORTED_OCR_ENGINES:
    DEFAULT_OCR_ENGINE = 'paddleocr'

class UploadPDFResponse(BaseModel):
    pdf_id: str = Field(..., description="Identifier assigned to the uploaded PDF")
    filename: str
    total_pages: int


class TaskStatusResponse(BaseModel):
    task_id: str
    type: str
    status: str
    progress: float
    message: str
    pdf_id: Optional[str]
    result_path: Optional[str]
    error: Optional[str]
    params: dict
    created_at: str
    updated_at: str


class StructureTaskRequest(BaseModel):
    pdf_id: str
    pages: Optional[str] = Field(None, description="Page selection string, e.g. '1,3,5-7'")
    page_numbers: Optional[List[int]] = Field(None, description="Explicit page numbers to process")
    engine: str = Field("molnextr", description="Structure extraction engine")

    @validator("page_numbers", always=True)
    def ensure_pages(cls, v, values):
        if not v and not values.get("pages"):
            raise ValueError("Either pages or page_numbers must be provided")
        return v


class UpdateStructuresRequest(BaseModel):
    records: List[dict]


class StructuresResultResponse(BaseModel):
    task: TaskStatusResponse
    records: List[dict]


class AssayTaskRequest(BaseModel):
    pdf_id: str
    assay_names: List[str]
    pages: Optional[str] = Field(None, description="Shared page selection string for all assays")
    page_numbers: Optional[List[int]] = Field(None, description="Explicit page numbers to process")
    lang: str = Field("en", description="Language hint for OCR/LLM pipeline")
    ocr_engine: str = Field(
        DEFAULT_OCR_ENGINE,
        description="OCR engine identifier (paddleocr | dots_ocr)",
    )
    structure_task_id: Optional[str] = Field(None, description="Optional structure task ID to get compound list for matching")

    @validator("assay_names")
    def ensure_names(cls, value):
        cleaned = [name.strip() for name in value if name and name.strip()]
        if not cleaned:
            raise ValueError("At least one assay name must be provided")
        return cleaned

    @validator("page_numbers", always=True)
    def ensure_pages(cls, v, values):
        if not v and not values.get("pages"):
            raise ValueError("Either pages or page_numbers must be provided")
        return v


class AssayResultResponse(BaseModel):
    task: TaskStatusResponse
    records: List[dict]


class MergeTaskRequest(BaseModel):
    structure_task_id: str = Field(..., description="Task ID containing structure data")
    assay_task_ids: List[str] = Field(..., description="List of task IDs for assay re-extraction with structure matching")


class MergeResultResponse(BaseModel):
    task: TaskStatusResponse
    records: List[dict]
