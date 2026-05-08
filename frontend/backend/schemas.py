from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator
SUPPORTED_STRUCTURE_FILTER_STRICTNESS = {'strict', 'balanced', 'permissive'}

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
    auto_detect_pages: bool = Field(False, description="Automatically detect likely structure pages")
    engine: str = Field("molnextr", description="Structure extraction engine")
    structure_filter_strictness: str = Field(
        "strict",
        description="Structure filter strictness (strict | balanced | permissive)",
    )

    @validator("page_numbers", always=True)
    def ensure_pages(cls, v, values):
        return v

    @validator("auto_detect_pages", always=True)
    def default_structure_auto_detect(cls, auto_detect_pages, values):
        if auto_detect_pages:
            return True
        if values.get("page_numbers") or values.get("pages"):
            return False
        return True

    @validator("structure_filter_strictness")
    def ensure_filter_strictness(cls, value):
        normalized = (value or "strict").strip().lower()
        if normalized not in SUPPORTED_STRUCTURE_FILTER_STRICTNESS:
            supported = ", ".join(sorted(SUPPORTED_STRUCTURE_FILTER_STRICTNESS))
            raise ValueError(f"Unsupported structure_filter_strictness '{value}'. Supported values: {supported}")
        return normalized


class UpdateStructuresRequest(BaseModel):
    records: List[dict]


class StructuresResultResponse(BaseModel):
    task: TaskStatusResponse
    records: List[dict]
    filtered_records: List[dict] = Field(default_factory=list)


class AssayTaskRequest(BaseModel):
    pdf_id: str
    assay_names: List[str]
    pages: Optional[str] = Field(None, description="Shared page selection string for all assays")
    page_numbers: Optional[List[int]] = Field(None, description="Explicit page numbers to process")
    auto_detect_pages: bool = Field(False, description="Automatically detect likely assay pages")
    auto_detect_assay_names: bool = Field(False, description="Automatically detect assay names when none are provided")
    lang: str = Field("en", description="Language hint for OCR/LLM pipeline")
    structure_task_id: Optional[str] = Field(None, description="Optional structure task ID to get compound list for matching")

    @validator("assay_names")
    def ensure_names(cls, value):
        cleaned = [name.strip() for name in value if name and name.strip()]
        if not cleaned:
            return []
        return cleaned

    @validator("page_numbers", always=True)
    def ensure_pages(cls, v, values):
        return v

    @validator("auto_detect_pages", always=True)
    def default_assay_page_auto_detect(cls, auto_detect_pages, values):
        if auto_detect_pages:
            return True
        if values.get("page_numbers") or values.get("pages"):
            return False
        return True


class AutoDetectTaskRequest(BaseModel):
    pdf_id: str
    assay_names: List[str] = Field(default_factory=list)
    detect_structure_pages: bool = True
    detect_assay_pages: bool = True
    detect_assay_names: bool = True

    @validator("assay_names", pre=True, always=True)
    def clean_assay_names(cls, value):
        return [str(name).strip() for name in (value or []) if str(name).strip()]


class AssayResultResponse(BaseModel):
    task: TaskStatusResponse
    records: List[dict]


class MergeTaskRequest(BaseModel):
    structure_task_id: str = Field(..., description="Task ID containing structure data")
    assay_task_ids: List[str] = Field(..., description="List of task IDs for assay re-extraction with structure matching")


class MergeResultResponse(BaseModel):
    task: TaskStatusResponse
    records: List[dict]


class FullPipelineRequest(BaseModel):
    pdf_id: str
    structure_filter_strictness: str = Field("strict", description="Structure filter strictness (strict | balanced | permissive)")
    lang: str = Field("en", description="Language hint for OCR/LLM pipeline")

    @validator("structure_filter_strictness")
    def ensure_filter_strictness(cls, value):
        normalized = (value or "strict").strip().lower()
        if normalized not in SUPPORTED_STRUCTURE_FILTER_STRICTNESS:
            supported = ", ".join(sorted(SUPPORTED_STRUCTURE_FILTER_STRICTNESS))
            raise ValueError(f"Unsupported structure_filter_strictness '{value}'. Supported values: {supported}")
        return normalized
