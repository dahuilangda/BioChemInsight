"""User-facing feature choices and retained-document history survive restarts."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import fitz

class PlatformOptionsTest(unittest.TestCase):
    def test_pdf_display_name_survives_process_restart(self):
        from frontend.backend.pdf_manager import PDFManager
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'input.pdf'
            with fitz.open() as document:
                document.new_page();document.save(source)
            manager=PDFManager(Path(temporary)/'uploads')
            registered=manager.register(source,'sample.pdf')
            restored=PDFManager(Path(temporary)/'uploads').get(registered.id)
            self.assertEqual(restored.filename,'sample.pdf')
            self.assertEqual(restored.stored_path.name,'source.pdf')

    def test_jobs_hide_archived_records(self):
        from frontend.backend import main
        from fastapi.testclient import TestClient
        tasks=[main.Task(id='kept',type='structure_extraction',status='completed').to_dict(),main.Task(id='old',type='structure_extraction',status='deleted').to_dict()]
        with patch.object(main,'task_manager',SimpleNamespace(list_summaries=lambda:tasks)),patch.object(main,'get_queue_positions',return_value={}):
            response=TestClient(main.app).get('/api/tasks')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual([task['task_id'] for task in response.json()['tasks']],['kept'])

    def test_markush_is_an_explicit_request_option(self):
        from frontend.backend.schemas import FullPipelineRequest,StructureTaskRequest
        self.assertFalse(FullPipelineRequest(pdf_id='document').assemble_markush)
        self.assertTrue(FullPipelineRequest(pdf_id='document',assemble_markush=True).assemble_markush)
        self.assertTrue(StructureTaskRequest(pdf_id='document',assemble_markush=True).assemble_markush)

    def test_task_child_applies_its_own_assembly_choice(self):
        import os
        from frontend.backend import main,task_child_runner
        from utils.structure_mode import markush_assembly_enabled
        for enabled in (False,True):
            record=SimpleNamespace(status='pending',params={'assemble_markush':enabled})
            with patch.dict(os.environ,{}),patch.object(main,'task_manager',SimpleNamespace(get=lambda key:record)),patch.object(main,'launch_structure_task',new=AsyncMock()):
                asyncio.run(task_child_runner._run_job({'task_id':'chosen','task_name':'structure_extraction'}))
                self.assertEqual(markush_assembly_enabled(),enabled)

if __name__=='__main__':unittest.main()
