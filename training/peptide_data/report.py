from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


def summarize(design, manifests):
    identities = {record["id"] for record in design["records"]}
    if len(identities) != len(design["records"]):
        raise ValueError("Duplicate chemical record identifiers")
    coverage, resolutions = {}, {}
    rendered = set()
    for name, manifest in manifests.items():
        if not manifest.get("sources_unchanged"):
            raise ValueError(f"Source changes during rendering: {name}")
        seen = set()
        strata, screens = defaultdict(Counter), defaultdict(Counter)
        for record in manifest["records"]:
            if record["id"] not in identities or record["id"] in seen:
                raise ValueError(f"Unknown or duplicate record: {record['id']}")
            seen.add(record["id"])
            strata[record["stratum"]][record["status"]] += 1
            if record["status"] != "rendered_for_review":
                continue
            rendered.add(record["id"])
            for style in record["images"]:
                for sample in style["inputs"]:
                    path = Path(sample["file"])
                    if (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        != sample["sha256"]
                    ):
                        raise ValueError(f"Image checksum mismatch: {path}")
                    screens[str(sample["size"])][
                        "pass" if sample["resolution_screen_pass"] else "fail"
                    ] += 1
        coverage[name] = dict(strata)
        resolutions[name] = dict(screens)
    return {
        "designed_structures": len(identities),
        "structures_with_review_images": len(rendered),
        "coverage": coverage,
        "resolution_screen": resolutions,
        "training_release_approved": False,
    }


def render_html(design, manifests, summary, output):
    runs = {
        name: {record["id"]: record for record in manifest["records"]}
        for name, manifest in manifests.items()
    }
    rows = []
    for candidate in design["records"]:
        cells = []
        for records in runs.values():
            result = records.get(candidate["id"])
            if result is None:
                cells.append("<td>Not run</td>")
                continue
            body = f"<p>{html.escape(result['status'])}</p>"
            for style in result.get("images", []):
                samples = style["inputs"]
                sample = next(
                    (item for item in samples if item["size"] == 384), samples[0]
                )

                def link(path):
                    return quote(
                        os.path.relpath(Path(path).resolve(), output.parent.resolve())
                    )

                body += f'<p>{html.escape(style["style"])}</p><a href="{link(sample["file"])}"><img loading="lazy" width="384" height="384" src="{link(sample["file"])}"></a><p>'
                for item in samples:
                    status = (
                        "screen pass"
                        if item["resolution_screen_pass"]
                        else "screen fail"
                    )
                    body += f'<a href="{link(item["file"])}">{item["size"]}: {status}</a> · '
                body += "</p>"
            reasons = result.get("geometry", {}).get("reasons", [])
            if reasons:
                body += f"<p>{html.escape(', '.join(reasons))}</p>"
            if result.get("error"):
                body += f"<p>{html.escape(result['error'])}</p>"
            cells.append(f"<td>{body}</td>")
        rows.append(
            f'<tr data-stratum="{candidate["stratum"]}"><td><strong>{html.escape(candidate["stratum"])}</strong><p>{candidate["id"][:12]}</p><p>{candidate["atom_count"]} atoms</p><code>{html.escape(candidate["biln"])}</code></td>{"".join(cells)}</tr>'
        )
    strata = sorted({record["stratum"] for record in design["records"]})
    options = "".join(f"<option>{stratum}</option>" for stratum in strata)
    headings = "".join(f"<th>{html.escape(name)}</th>" for name in runs)
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Peptide image review</title>
<style>body{{font:15px system-ui;margin:24px;color:#18212b}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd3da;padding:12px;vertical-align:top}}td:first-child{{max-width:260px;overflow-wrap:anywhere}}img{{object-fit:contain;border:1px solid #eee}}code{{font-size:12px}}a{{color:#174e91}}select{{padding:6px}}</style>
<h1>Peptide image review</h1>
<p>{summary["designed_structures"]} designed structures; {summary["structures_with_review_images"]} have review images. No training release is approved.</p>
<p>Resolution screens check nominal font, bond and stroke sizes. Glyph collisions, stereo visibility and domain realism still need review. Every attempt remains in the table.</p>
<label>Stratum <select id="stratum"><option value="">All</option>{options}</select></label>
<table><thead><tr><th>Chemical record</th>{headings}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<script>document.querySelector('#stratum').addEventListener('change',event=>{{for(const row of document.querySelectorAll('tbody tr'))row.hidden=Boolean(event.target.value)&&row.dataset.stratum!==event.target.value;}});</script>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    design_hash = hashlib.sha256(args.design.read_bytes()).hexdigest()
    manifests = {}
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        if manifest["design_sha256"] != design_hash:
            raise ValueError("Manifest uses a different chemical design")
        name = path.parent.name
        if name in manifests:
            raise ValueError("Manifest names must be unique")
        manifests[name] = manifest
    summary = summarize(design, manifests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(design, manifests, summary, args.out))
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
