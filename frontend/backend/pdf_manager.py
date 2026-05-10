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
        self._load_existing()

    def _load_existing(self) -> None:
        """Restore PDF metadata for files already present on disk."""
        restored: Dict[str, PDFDocument] = {}
        for pdf_dir in self.storage_root.iterdir():
            if not pdf_dir.is_dir():
                continue
            pdf_doc = self._load_pdf_dir(pdf_dir)
            if pdf_doc is not None:
                restored[pdf_doc.id] = pdf_doc
        with self._lock:
            self._pdfs.update(restored)

    def _load_pdf_dir(self, pdf_dir: Path) -> Optional[PDFDocument]:
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            return None
        stored_path = pdf_files[0]
        try:
            with fitz.open(stored_path) as doc:
                total_pages = doc.page_count
        except Exception:
            return None
        try:
            uploaded_at = datetime.fromtimestamp(stored_path.stat().st_mtime)
        except OSError:
            uploaded_at = datetime.utcnow()
        return PDFDocument(
            id=pdf_dir.name,
            filename=stored_path.name,
            stored_path=stored_path,
            total_pages=total_pages,
            uploaded_at=uploaded_at,
        )

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
            pdf = self._pdfs.get(pdf_id)
        if pdf is not None:
            return pdf

        # Web uploads register PDFs in the web process, while Celery workers are
        # long-lived and may have loaded their in-memory PDF index before the
        # upload happened. Lazily restore the requested PDF from the shared
        # volume so queued jobs can start without restarting the worker.
        pdf_dir = self.storage_root / pdf_id
        if not pdf_dir.is_dir():
            return None
        pdf = self._load_pdf_dir(pdf_dir)
        if pdf is None:
            return None
        with self._lock:
            self._pdfs[pdf_id] = pdf
        return pdf

    def list(self) -> List[PDFDocument]:
        with self._lock:
            return list(self._pdfs.values())

    def ensure_pdf(self, pdf_id: str) -> PDFDocument:
        pdf = self.get(pdf_id)
        if pdf is None:
            raise FileNotFoundError(f"PDF '{pdf_id}' not found")
        return pdf
