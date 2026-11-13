"""Geometry acquisition only: a detection never claims identity or a decoded graph."""
from pathlib import Path
import json
import os
import uuid


def locate_structure_regions(pdf_file, page_number, output_dir, records, confidence=.15):
    import fitz
    from utils.api_harness import APIHarness
    folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_file) as document:
        page = document[page_number - 1]
        bounds = page.rect
        pixmap = page.get_pixmap(dpi=180, alpha=False)
    page_image = folder / 'source_page.png'; pixmap.save(page_image)
    client = APIHarness(os.environ.get('STRUCTURE_API_BASE_URL', 'http://harness-api:8011'), timeout=120)
    client.session.trust_env = False
    detected = client.image('detect', page_image, {}, {'conf': confidence})
    known = []
    for row in records:
        if str(row.get('PAGE_NUM') or row.get('page_num') or row.get('page')) != str(page_number): continue
        try:
            evidence = json.loads(Path(row['BOX_COORDS_FILE']).read_text())
            box = evidence['box']; w, h = evidence['page_width'], evidence['page_height']
            known.append([box[0]/w, box[1]/h, box[2]/w, box[3]/h])
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError): continue
    def overlap(a, b):
        intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
        return intersection/union if union > 0 else 0
    additions = []
    for item in detected.get('structures', []):
        box = item.get('bbox')
        if not isinstance(box, list) or len(box) != 4: continue
        x0,y0,x1,y1 = box
        if not 0 <= x0 < x1 <= pixmap.width or not 0 <= y0 < y1 <= pixmap.height: continue
        normalized = [x0/pixmap.width,y0/pixmap.height,x1/pixmap.width,y1/pixmap.height]
        if any(overlap(normalized, previous) >= .6 for previous in known): continue
        known.append(normalized)
        region_id = uuid.uuid4().hex
        crop = folder / f'{region_id}.png'
        from PIL import Image
        with Image.open(page_image) as image: image.crop((x0,y0,x1,y1)).save(crop)
        box_path = folder / f'{region_id}.json'
        evidence_box = [normalized[0]*bounds.width,normalized[1]*bounds.height,normalized[2]*bounds.width,normalized[3]*bounds.height]
        box_path.write_text(json.dumps({'box': evidence_box,'page_width':bounds.width,'page_height':bounds.height,'source':'detector_recovery'}))
        additions.append({'RECORD_ID':region_id,'COMPOUND_ID':'','SMILES':'','PAGE_NUM':page_number,
            'SEGMENT_FILE':str(crop),'PAGE_IMAGE_FILE':str(page_image),'BOX_COORDS_FILE':str(box_path),
            'SOURCE':'detector_recovery','STRUCTURE_RECOVERY_STATUS':'geometry_only',
            'DETECTION_CONFIDENCE':str(item.get('score') or '')})
    return additions
