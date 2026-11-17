"""Run browser fixtures against local preview and module servers.

Usage: NODE_PATH=/path/to/node_modules python frontend/tests/run_workbench_qa.py
Build first: npm --prefix frontend/ui run build
"""

from pathlib import Path
import argparse
import concurrent.futures
import datetime
import json
import os
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output or root / "output" / (
        "workbench_qa_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if not env.get("NODE_PATH") and Path("/tmp/biochem-ui-qa/node_modules").exists():
        env["NODE_PATH"] = "/tmp/biochem-ui-qa/node_modules"
    env["WORKBENCH_URL"] = "http://127.0.0.1:5173"
    env["WORKBENCH_MODULE_URL"] = "http://127.0.0.1:5174"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    servers = []
    try:
        for port, command, name in [
            (5173, ["preview"], "preview"),
            (5174, [], "module"),
        ]:
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    raise RuntimeError(
                        f"Port {port} is in use; stop that local test server before running QA"
                    )
            with (out / f"{name}-server.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        "./node_modules/.bin/vite",
                        *command,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--strictPort",
                    ],
                    cwd=root / "frontend/ui",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            servers.append(process)
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} server exited; see {out}")
                try:
                    opener.open(f"http://127.0.0.1:{port}/", timeout=1).close()
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"{name} server unavailable")
        suites = sorted((root / "frontend/tests").glob("workbench*.browser.cjs"))
        suites = list(
            dict.fromkeys([*suites, root / "frontend/tests/workbench.browser.cjs"])
        )

        def run(path):
            start = time.monotonic()
            try:
                result = subprocess.run(
                    ["node", str(path)],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                output = result.stdout + result.stderr
                code = result.returncode
            except subprocess.TimeoutExpired as exc:
                output = str(exc)
                code = 124
            (out / (path.name + ".log")).write_text(output)
            result = {
                "suite": path.name,
                "exit_code": code,
                "seconds": round(time.monotonic() - start, 2),
            }
            print(json.dumps(result), flush=True)
            return result

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(4, args.workers))
        ) as pool:
            results = list(pool.map(run, suites))
        (out / "browser-matrix.json").write_text(json.dumps(results, indent=2))
        return int(any(result["exit_code"] for result in results))
    finally:
        for process in servers:
            if process.poll() is None:
                process.terminate()
        for process in servers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
