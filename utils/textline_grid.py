"""Grid reconstruction from text-line OCR boxes.

PP-OCR's table-structure module collapses on dense scanned chemical
tables, but its line boxes are reliable (the accepted-row mass deltas
prove character accuracy). This skill rebuilds the table grid
geometrically:

  1. cluster line boxes into visual rows (y-center, tolerance = half the
     median box height)
  2. pick the header row (top row whose cells span the table width) and
     use its cell boxes as column anchors; header wraps are absorbed
     until the first row carrying a printed compound id
  3. assign every data line to one column by x-center; fragments in the
     same column concatenate in x order
  4. heal segmentation damage: tight same-row gaps rejoin with the
     notation's dash separator; rows without a compound id merge upward
     into the row above (wrapped cells, multi-line SMILES)

No layout templates, no per-document rules: the anchors come from the
page's own header row, and unassignable lines are named, never dropped.
"""
from __future__ import annotations

import re
import statistics

_ID_RE = re.compile(r"^\d{1,4}$")


def build_grids(lines: list[dict]) -> tuple[list[list[list[str]]], list[str]]:
    """Text lines -> (grids, notes). One grid per page (sheets carry one
    table region). Each grid is a list of rows of cell strings aligned
    to the header's columns."""
    if not lines:
        return [], ["no lines"]
    heights = [max(l["box"][3] - l["box"][1], 1) for l in lines]
    tol = max(statistics.median(heights) * 0.6, 1)
    gap = max(statistics.median(heights) * 1.2, 1)

    rows: list[list[dict]] = []
    for line in sorted(lines, key=lambda l: _yc(l)):
        if rows and abs(_yc(line) - _rows_yc(rows[-1])) <= tol:
            rows[-1].append(line)
        else:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda l: l["box"][0])

    header = _pick_header(rows)
    if header is None:
        return [], ["no header row found"]
    anchors = _prune_anchors([(_xc(l), _x1(l), _x2(l)) for l in header])
    # long cells (notation/SMILES lines) overrun their column: assign by
    # start position. Threshold: 1.5x the median header column width.
    col_w = statistics.median([a[2] - a[1] for a in anchors])
    long_w = col_w * 1.5

    header_cells: list[list[str]] = [[] for _ in anchors]
    for line in header:
        idx = _match_anchor(line, anchors)
        if idx is None:
            continue
        header_cells[idx].append(line["text"])
    header_row = [" ".join(c) for c in header_cells]

    # absorb wrapped header rows until the first row with a printed id
    # in column 0 — the miner's own row contract
    data_start = len(rows)
    for j, row in enumerate(rows):
        if row is header:
            continue
        col0 = [l["text"] for l in row if _match_anchor(l, anchors) == 0]
        if col0 and all(_ID_RE.match(t.strip()) for t in col0):
            data_start = j
            break
        for line in row:
            idx = _match_anchor(line, anchors)
            if idx is not None:
                header_cells[idx].append(line["text"])
    header_row = [" ".join(c) for c in header_cells]

    grid: list[list[str]] = [header_row]
    notes: list[str] = []
    for row in rows[data_start:]:
        cells = _row_cells(row, anchors, long_w, gap, notes)
        if not _ID_RE.match(cells[0].strip()):
            # continuation row: merge upward cell-wise
            _merge_up(grid, cells)
            continue
        grid.append(cells)
    return [grid], notes


def _row_cells(row, anchors, long_w, gap, notes) -> list[str]:
    cells: list[list[str]] = [[] for _ in anchors]
    prev_idx, prev_line = None, None
    for line in row:
        idx = _match_anchor(line, anchors, long_w)
        if idx is None:
            notes.append(f"unanchored line: {line['text'][:40]}")
            prev_idx, prev_line = None, None
            continue
        healed = False
        if idx == prev_idx and line["box"][0] - prev_line["box"][2] <= gap:
            a, b = prev_line["text"], line["text"]
            sep = "-"
            if not (a and b and _alnum(a[-1]) and _alnum(b[0])):
                sep = " "
            cells[idx][-1] = a + sep + b
            healed = True
        if not healed:
            cells[idx].append(line["text"])
        prev_idx, prev_line = (idx, {"text": cells[idx][-1],
                                     "box": line["box"]})
    return [" ".join(c) for c in cells]


def _merge_up(grid: list[list[str]], cells: list[str]) -> None:
    """A row without a printed id continues the row above: its cell text
    appends to the nearest non-empty cell of the row above (wrapped
    cells, wrapped SMILES). Column 0 is the printed-id column —
    continuation content never lands there."""
    if not grid:
        return
    above = grid[-1]
    for i, text in enumerate(cells):
        if not text:
            continue
        if i > 0 and above[i]:
            above[i] = f"{above[i]} {text}"
        else:
            # column drifted or empty above: nearest non-empty neighbour,
            # printed-id column excluded
            j = _nearest_filled(above[1:], i - 1) + 1 if len(above) > 1 else 0
            above[j] = f"{above[j]} {text}" if above[j] else text


def _nearest_filled(row: list[str], i: int) -> int:
    best, best_d = i, len(row) + 1
    for j, text in enumerate(row):
        if text:
            d = abs(j - i)
            if d < best_d:
                best, best_d = j, d
    return best



def _alnum(ch: str) -> bool:
    return ch.isalnum()


def _prune_anchors(anchors: list[tuple[float, float, float]],
                   min_gap: float = 200) -> list[tuple[float, float, float]]:
    """Keep the largest cluster of anchors separated by < min_gap;
    running heads and page numbers sit 200px+ beyond the table's edge,
    while intra-table column gaps stay under ~150px."""
    if not anchors:
        return anchors
    anchors = sorted(anchors)
    clusters: list[list[int]] = [[0]]
    for k in range(1, len(anchors)):
        if anchors[k][1] - anchors[k - 1][2] < min_gap:
            clusters[-1].append(k)
        else:
            clusters.append([k])
    best = max(clusters, key=len)
    return [anchors[k] for k in best]


def _yc(line: dict) -> float:
    return (line["box"][1] + line["box"][3]) / 2


def _x1(line: dict) -> float:
    return line["box"][0]


def _x2(line: dict) -> float:
    return line["box"][2]


def _xc(line: dict) -> float:
    return (_x1(line) + _x2(line)) / 2


def _rows_yc(row: list[dict]) -> float:
    return statistics.mean(_yc(l) for l in row)


def _row_width(row: list[dict]) -> float:
    return max(_x2(l) for l in row) - min(_x1(l) for l in row)


def _pick_header(rows: list[list[dict]]):
    """The first row (top third of the page) spanning most of the table
    width with several cells."""
    if not rows:
        return None
    width = max(_row_width(r) for r in rows)
    top = rows[: max(len(rows) // 3, 1)]
    for row in top:
        if len(row) >= 3 and _row_width(row) >= width * 0.7:
            return row
    return None


def _match_anchor(line: dict, anchors, long_w: float = 0) -> int | None:
    """Long boxes (notation/SMILES lines) assign by start position —
    they overrun their column; short boxes by x-center."""
    width = line["box"][2] - line["box"][0]
    xc = _x1(line) if width > long_w else _xc(line)
    if xc < anchors[0][1] - 40 or xc > anchors[-1][2] + 40:
        return None                 # outside the table (running heads)
    for i, (_axc, ax1, ax2) in enumerate(anchors):
        if ax1 - 20 <= xc <= ax2 + 20:
            return i
    # x-center between anchors: nearest anchor
    best, best_d = None, None
    for i, (_axc, ax1, ax2) in enumerate(anchors):
        d = ax1 - xc if xc < ax1 else (xc - ax2 if xc > ax2 else 0)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best
