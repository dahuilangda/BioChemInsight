"""Exercise the client's HTTP contract without model dependencies."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from utils.api_harness import APIHarness, main


@pytest.fixture
def server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            calls.append((self.path, self.headers.get('Content-Type'), body))
            result = {'ok': True, 'sites': [{'start': 600}]}
            if self.path.endswith('bad/invoke'):
                result = {'ok': False, 'detail': 'cannot assemble'}
            if self.path.endswith('missing/invoke'):
                self.send_response(404)
                result = {'error': 'unknown skill'}
            else:
                self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    client = APIHarness(f'http://127.0.0.1:{httpd.server_port}', 5)
    client.session.trust_env = False
    try:
        yield client, calls
    finally:
        client.session.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def test_json_and_failure_semantics(server):
    client, calls = server
    assert client.request('POST', '/v1/skills/variant_sites/invoke', json={'hgvs': ['p.Val600Glu']})['ok']
    assert json.loads(calls[0][2]) == {'hgvs': ['p.Val600Glu']}
    for name in ('bad', 'missing'):
        with pytest.raises(RuntimeError):
            client.request('POST', f'/v1/skills/{name}/invoke', json={})


def test_image_upload_preserves_form_and_query(server, tmp_path):
    client, calls = server
    image = tmp_path / 'crop.png'
    image.write_bytes(b'actual image bytes')
    client.image('resolve-figure', image, {'document_id': 'patent-1'}, {'conf': '0.3'})
    path, content_type, body = calls[0]
    assert path == '/v1/harness/resolve-figure?conf=0.3'
    assert content_type.startswith('multipart/form-data; boundary=')
    assert b'name="document_id"' in body and b'patent-1' in body
    assert b'filename="crop.png"' in body and b'actual image bytes' in body


def test_invalid_payload_is_nonzero(tmp_path, capsys):
    payload = tmp_path / 'payload.json'
    payload.write_text('[]')
    assert main(['invoke', 'variant_sites', '--json', str(payload)]) == 1
    assert 'JSON object' in capsys.readouterr().err


@pytest.fixture
def workflow_server():
    """Minimal async API for testing task handles across observation timeouts."""
    from io import BytesIO
    import zipfile
    calls = []
    state = {'status': 'pending'}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, payload):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            calls.append(self.path)
            self.reply({'pdf_id': 'pdf-1'} if self.path == '/pdfs' else
                       {'task_id': 'job-1', 'status': state['status'], 'error': 'model unavailable' if state['status'] == 'failed' else None})

        def do_GET(self):
            calls.append(self.path)
            if self.path.endswith('/download'):
                data = BytesIO()
                with zipfile.ZipFile(data, 'w') as archive:
                    archive.writestr('structures.csv', 'COMPOUND_ID,SMILES\n1,CCO\n')
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data.getvalue())
            else:
                self.reply({'records': []} if self.path.endswith(('/structures', '/assays')) else
                           {'task_id': 'job-1', 'status': state['status'], 'error': 'model unavailable' if state['status'] == 'failed' else None})

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    client = APIHarness(f'http://127.0.0.1:{httpd.server_port}', 2)
    client.session.trust_env = False
    try:
        yield client, state, calls
    finally:
        client.session.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def test_workflow_timeout_retains_handle_and_resume_does_not_resubmit(workflow_server, tmp_path):
    from utils.api_harness import run_workflow
    client, state, calls = workflow_server
    pdf = tmp_path / 'page.pdf'
    pdf.write_bytes(b'%PDF-test')
    report = tmp_path / 'report.json'
    with pytest.raises(RuntimeError, match='Resume with --resume job-1'):
        run_workflow(client, pdf=pdf, task_id=None, report_path=report,
                     wait_seconds=0.01, poll_seconds=0.01)
    assert json.loads(report.read_text())['task']['task_id'] == 'job-1'
    assert calls.count('/tasks/full-pipeline') == 1
    state['status'] = 'completed'
    result = run_workflow(client, pdf=None, task_id='job-1', report_path=report,
                          wait_seconds=1, poll_seconds=.01, download_path=tmp_path / 'result.zip')
    assert result['download_files'] == ['structures.csv']
    assert calls.count('/tasks/full-pipeline') == 1
    assert calls.count('/pdfs') == 1
    assert (tmp_path / 'result.zip').exists()


def test_failed_task_is_reported_and_not_downloaded(workflow_server, tmp_path):
    from utils.api_harness import run_workflow
    client, state, calls = workflow_server
    state['status'] = 'failed'
    report = tmp_path / 'report.json'
    with pytest.raises(RuntimeError, match='failed'):
        run_workflow(client, pdf=None, task_id='job-1', report_path=report,
                     wait_seconds=1, poll_seconds=.01, download_path=tmp_path / 'result.zip')
    assert json.loads(report.read_text())['task']['status'] == 'failed'
    assert not any(path.endswith('/download') for path in calls)
