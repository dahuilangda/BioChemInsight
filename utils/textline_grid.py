"""Table grid reconstruction from PP-OCR text-line boxes."""
from __future__ import annotations

import re
import statistics

_ID_RE = re.compile(r"^\d{1,4}$")


def build_grids(lines: list[dict]) -> tuple[list[list[list[str]]], list[str]]:
    if not lines:
        return [], ["no lines"]
    heights = [max(l["box"][3] - l["box"][1], 1) for l in lines]
    tol = max(statistics.median(heights) * 0.6, 1)
    gap = max(statistics.median(heights) * 1.2, 1)

    rows: list[list[dict]] = []
    for line in sorted(lines, key=lambda l: _y_center(l)):
        if rows and abs(_y_center(line) - _row_y_center(rows[-1])) <= tol:
            rows[-1].append(line)
        else:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda l: l["box"][0])

    header = _pick_header(rows)
    if header is None:
        return [], ["no header row found"]
    anchors = _prune_anchors([(_x_center(l), _x_start(l), _x_end(l)) for l in header])
    col_w = statistics.median([a[2] - a[1] for a in anchors])
    long_w = col_w * 1.5

    header_cells: list[list[str]] = [[] for _ in anchors]
    for line in header:
        idx = _match_anchor(line, anchors)
        if idx is None:
            continue
        header_cells[idx].append(line["text"])
    header_row = [" ".join(c) for c in header_cells]

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
        rejoined = False
        if idx == prev_idx and line["box"][0] - prev_line["box"][2] <= gap:
            a, b = prev_line["text"], line["text"]
            sep = "-"
            if not (a and b and _alnum(a[-1]) and _alnum(b[0])):
                sep = " "
            cells[idx][-1] = a + sep + b
            rejoined = True
        if not rejoined:
            cells[idx].append(line["text"])
        prev_idx, prev_line = (idx, {"text": cells[idx][-1],
                                     "box": line["box"]})
    return [" ".join(c) for c in cells]


def _merge_up(grid: list[list[str]], cells: list[str]) -> None:
    if not grid:
        return
    above = grid[-1]
    for i, text in enumerate(cells):
        if not text:
            continue
        if i > 0 and above[i]:
            above[i] = f"{above[i]} {text}"
        else:
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


def _y_center(line: dict) -> float:
    return (line["box"][1] + line["box"][3]) / 2


def _x_start(line: dict) -> float:
    return line["box"][0]


def _x_end(line: dict) -> float:
    return line["box"][2]


def _x_center(line: dict) -> float:
    return (_x_start(line) + _x_end(line)) / 2


def _row_y_center(row: list[dict]) -> float:
    return statistics.mean(_y_center(l) for l in row)


def _row_width(row: list[dict]) -> float:
    return max(_x_end(l) for l in row) - min(_x_start(l) for l in row)


def _pick_header(rows: list[list[dict]]):
    if not rows:
        return None
    width = max(_row_width(r) for r in rows)
    top = rows[: max(len(rows) // 3, 1)]
    for row in top:
        if len(row) >= 3 and _row_width(row) >= width * 0.7:
            return row
    return None


def _match_anchor(line: dict, anchors, long_w: float = 0) -> int | None:
    width = line["box"][2] - line["box"][0]
    xc = _x_start(line) if width > long_w else _x_center(line)
    if xc < anchors[0][1] - 40 or xc > anchors[-1][2] + 40:
        return None
    for i, (_axc, ax1, ax2) in enumerate(anchors):
        if ax1 - 20 <= xc <= ax2 + 20:
            return i
    best, best_d = None, None
    for i, (_axc, ax1, ax2) in enumerate(anchors):
        d = ax1 - xc if xc < ax1 else (xc - ax2 if xc > ax2 else 0)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best
