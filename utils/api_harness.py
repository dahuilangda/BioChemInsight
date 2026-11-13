"""Live client for the deployed structure/harness API (no local model imports).

python -m utils.api_harness --base-url http://localhost:8011 health
python -m utils.api_harness skills
python -m utils.api_harness parse-page page.png --query max_structures=4
python -m utils.api_harness resolve-figure crop.png --field document_id=patent-1
python -m utils.api_harness invoke variant_sites --json payload.json
python -m utils.api_harness workflow patent.pdf --report run.json --download results.zip
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import math
import time
import zipfile
import io
import tempfile
from pathlib import Path
import sys
from urllib.parse import quote

import requests


IMAGE_OPERATIONS = {
    'detect': '/v1/structures/detect',
    'recognize': '/v1/structures/recognize',
    'parse': '/v1/structures/parse',
    'ocr': '/v1/ocr',
    **{name: '/v1/harness/' + name for name in (
        'parse-page', 'verify-structure', 'resolve-figure', 'link-activity')},
}


class APIHarness:
    def __init__(self, base_url: str, timeout: float = 600):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def request(self, method: str, path: str, **kwargs):
        response = self.session.request(method, self.base_url + path,
                                        timeout=kwargs.pop('timeout', self.timeout), **kwargs)
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError(f'HTTP {response.status_code}: {response.text[:500]}') from None
        if not response.ok:
            raise RuntimeError(f'HTTP {response.status_code}: {json.dumps(result, ensure_ascii=False)[:1000]}')
        # Several capabilities report domain failures with HTTP 200.
        is_task_status = isinstance(result, dict) and result.get('task_id') and result.get('status') in {'pending', 'running', 'completed', 'failed', 'canceled'}
        if isinstance(result, dict) and not is_task_status and (result.get('error') or result.get('ok') is False):
            raise RuntimeError(json.dumps(result, ensure_ascii=False)[:1000])
        return result

    def image(self, operation: str, image: Path, fields: dict, query: dict):
        with image.open('rb') as stream:
            return self.request('POST', IMAGE_OPERATIONS[operation],
                                files={'file': (image.name, stream,
                                       mimetypes.guess_type(image.name)[0] or 'application/octet-stream')},
                                data=fields, params=query)


def run_workflow(client, *, pdf, task_id, report_path, wait_seconds, poll_seconds,
                 download_path=None, assay_names=None):
    """Submit once or resume an existing job; persist its handle before waiting.

    A wait timeout never cancels or resubmits the remote task. The report keeps
    the last authoritative task status and is usable with --resume.
    """
    report = {'api_base': client.base_url}

    def save():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=report_path.parent,
                                             prefix=report_path.name + '.', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(report_path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()

    # Probe writability before submission without destroying a prior report.
    with tempfile.TemporaryFile(dir=report_path.parent):
        pass
    if task_id:
        task = client.request('GET', '/tasks/' + quote(task_id, safe=''))
    else:
        with pdf.open('rb') as stream:
            info = client.request('POST', '/pdfs',
                                  files={'file': (pdf.name, stream, 'application/pdf')})
        report['pdf'] = info
        save()
        task = client.request('POST', '/tasks/full-pipeline',
                              json={'pdf_id': info['pdf_id'], 'assay_names': assay_names})
    task_id = task.get('task_id')
    if not task_id:
        raise RuntimeError('API did not return a task_id')
    report['task'] = task
    save()
    deadline = time.monotonic() + wait_seconds
    last_message = None
    while True:
        report['task'] = task
        save()
        message = f"{task_id}: {task.get('status')} · {task.get('message', '')}"
        if message != last_message:
            print(message, file=sys.stderr)
            last_message = message
        state = task.get('status')
        if state == 'completed':
            break
        if state in ('failed', 'canceled'):
            raise RuntimeError(f"Task {task_id} {state}: {task.get('error') or task.get('message', '')}")
        if state not in ('pending', 'running'):
            raise RuntimeError(f"Unknown task state: {state!r}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f'Wait expired; task {task_id} remains {state}. Resume with --resume {task_id}')
        time.sleep(min(poll_seconds, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        task = client.request('GET', '/tasks/' + quote(task_id, safe=''),
                              timeout=min(client.timeout, remaining))
    for kind in ('structures', 'assays'):
        report[kind] = client.request('GET', f'/tasks/{quote(task_id, safe="")}/{kind}')
        save()
    if download_path:
        response = client.session.get(client.base_url + f'/tasks/{quote(task_id, safe="")}/download',
                                      timeout=client.timeout)
        response.raise_for_status()
        # The workbench exports a ZIP, not CSV. Validate before writing it.
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('Download failed ZIP integrity validation')
            report['download_files'] = archive.namelist()
        download_path.write_bytes(response.content)
        report['download'] = str(download_path)
        save()
    return report


def pairs(values):
    result = {}
    for value in values:
        key, separator, item = value.partition('=')
        if not separator or not key:
            raise ValueError(f'Expected name=value, got {value!r}')
        result[key] = item
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=os.environ.get('HARNESS_API_URL', 'http://127.0.0.1:8011'))
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--output', type=Path, help='Write the actual JSON response to this file')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('health')
    commands.add_parser('skills')
    invoke = commands.add_parser('invoke')
    invoke.add_argument('name')
    invoke.add_argument('--json', type=Path, required=True, dest='payload')
    for name in IMAGE_OPERATIONS:
        command = commands.add_parser(name)
        command.add_argument('image', type=Path)
        command.add_argument('--field', action='append', default=[])
        command.add_argument('--query', action='append', default=[])
    workflow = commands.add_parser('workflow', help='Run or resume the real PDF workbench pipeline')
    workflow.add_argument('pdf', type=Path, nargs='?')
    workflow.add_argument('--resume', dest='task_id')
    workflow.add_argument('--workbench-url', default=os.environ.get('WORKBENCH_API_URL', 'http://127.0.0.1:8000/api'))
    workflow.add_argument('--report', type=Path, required=True)
    workflow.add_argument('--download', type=Path)
    workflow.add_argument('--wait-seconds', type=float, default=900)
    workflow.add_argument('--poll-seconds', type=float, default=2)
    workflow.add_argument('--assay', action='append', dest='assay_names')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.command == 'workflow':
        if bool(args.pdf) == bool(args.task_id):
            parser.error('Provide a PDF or --resume TASK_ID, exactly one')
        paths = [p.resolve() for p in (args.pdf, args.report, args.download) if p]
        if len(paths) != len(set(paths)):
            parser.error('PDF, report and download must use different file paths')
        if not all(math.isfinite(v) and v > 0 for v in (args.wait_seconds, args.poll_seconds)):
            parser.error('Wait and poll intervals must be positive finite numbers')
    client = APIHarness(args.workbench_url if args.command == 'workflow' else args.base_url, args.timeout)
    try:
        if args.command == 'workflow':
            result = run_workflow(client, pdf=args.pdf, task_id=args.task_id, report_path=args.report,
                                  wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds,
                                  download_path=args.download, assay_names=args.assay_names)
        elif args.command in ('health', 'skills'):
            result = client.request('GET', '/healthz' if args.command == 'health' else '/v1/skills')
        elif args.command == 'invoke':
            payload = json.loads(args.payload.read_text())
            if not isinstance(payload, dict):
                raise ValueError('Skill payload must be a JSON object')
            result = client.request('POST', '/v1/skills/' + quote(args.name, safe='') + '/invoke', json=payload)
        else:
            result = client.image(args.command, args.image, pairs(args.field), pairs(args.query))
        displayed = result
        if args.command == 'workflow' and not args.output:
            displayed = {
                'task_id': result['task']['task_id'], 'status': result['task']['status'],
                'report': str(args.report),
                'structure_count': len(result['structures'].get('records', [])),
                'assay_count': len(result['assays'].get('records', [])),
                'download': result.get('download'),
            }
        rendered = json.dumps(displayed, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.write_text(rendered)
        else:
            print(rendered, end='')
        return 0
    except (requests.RequestException, OSError, ValueError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
        print(f'harness: {exc}', file=sys.stderr)
        return 1
    finally:
        client.session.close()


if __name__ == '__main__':
    raise SystemExit(main())
