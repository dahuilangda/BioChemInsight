from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF


@dataclass
class PDFDocument:
    id: str
    filename: str
    stored_path: Path
    total_pages: int
    uploaded_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, str]:
        payload = asdict(self)
        payload["stored_path"] = str(self.stored_path)
        payload["uploaded_at"] = self.uploaded_at.isoformat()
        return payload


class PDFManager:
    """Stores uploaded PDFs on disk and keeps lightweight metadata."""

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._pdfs: Dict[str, PDFDocument] = {}
        self._lock = threading.Lock()

    def register(self, src_path: Path, filename: Optional[str] = None) -> PDFDocument:
        pdf_id = uuid.uuid4().hex
        filename = filename or src_path.name
        pdf_dir = self.storage_root / pdf_id
        pdf_dir.mkdir(parents=True, exist_ok=True)
        target_path = pdf_dir / filename
        shutil.copy2(src_path, target_path)

        with fitz.open(target_path) as doc:
            total_pages = doc.page_count

        pdf_doc = PDFDocument(id=pdf_id, filename=filename, stored_path=target_path, total_pages=total_pages)
        with self._lock:
            self._pdfs[pdf_id] = pdf_doc
        return pdf_doc

    def get(self, pdf_id: str) -> Optional[PDFDocument]:
        with self._lock:
            return self._pdfs.get(pdf_id)

    def list(self) -> List[PDFDocument]:
        with self._lock:
            return list(self._pdfs.values())

    def ensure_pdf(self, pdf_id: str) -> PDFDocument:
        pdf = self.get(pdf_id)
        if pdf is None:
            raise FileNotFoundError(f"PDF '{pdf_id}' not found")
        return pdf
