"""
title_block_v3.py — Robust title-block detector for DXF and PDF drawings.

Improvements over v2
--------------------
1. PDF COORDINATE BUG FIX: PyMuPDF's get_text() returns coordinates in the
   PDF's internal un-rotated space.  For pages with rotation != 0 this means
   the word positions do NOT match the rendered pixel image.
   Fix: apply page.derotation_matrix to every word bbox before density
   analysis.  Now word positions and pixel positions are in the same space.

2. ORIENTATION DETECTION:
   • PDF: reads page.rotation (0/90/180/270) directly from the PDF header.
     Falls back to voting on text 'dir' vectors from get_text('dict').
   • DXF: votes on the 'rotation' attribute of every TEXT entity.
   • Raster images: uses EasyOCR bbox aspect-ratio clustering to guess
     whether the majority of text is horizontal or vertical.

3. AUTOMATIC IMAGE ROTATION:
   After detecting the orientation the rendered image is rotated to upright
   BEFORE title-block detection runs, so the crop / debug overlay are always
   in the visually-correct orientation.

4. MONOCHROME PRE-PROCESSING FOR OCR FALLBACK:
   When EasyOCR is needed (outlined/scanned PDFs, raster images) the image
   is first converted to grayscale and contrast-enhanced with CLAHE before
   being handed to the OCR engine.  The colour image is still used for
   the final debug overlay and crop.
"""

import cv2
import numpy as np
import fitz            # PyMuPDF
import ezdxf
from ezdxf.addons.drawing import matplotlib as dxf_matplotlib
from ezdxf.addons.drawing import RenderContext, Frontend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import os
import json
from PIL import Image
from dataclasses import dataclass
from collections import Counter
from typing import Optional, Tuple, List

# ── Convenience types ─────────────────────────────────────────────────────────
BBox   = Tuple[int, int, int, int]           # (x1,y1,x2,y2) pixel space
DXFBox = Tuple[float, float, float, float]   # (x1,y1,x2,y2) DXF units


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – FILE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf_as_image(filepath: str, dpi: int = 300) -> np.ndarray:
    """Rasterise page-0 of a PDF (PyMuPDF applies page.rotation automatically)."""
    print(f"  Rasterising PDF at {dpi} DPI…")
    doc  = fitz.open(filepath)
    page = doc.load_page(0)
    pix  = page.get_pixmap(dpi=dpi)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)


def load_dxf_as_image(filepath: str, dpi: int = 200) -> np.ndarray:
    """Render DXF MODEL SPACE to an image (paper-space only holds VIEWPORT)."""
    print("  Rendering DXF model space…")
    doc    = ezdxf.readfile(filepath)
    layout = doc.modelspace()

    fig = plt.figure()
    ax  = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = dxf_matplotlib.MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(layout, finalize=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def load_file(filepath: str) -> np.ndarray:
    ext = filepath.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return load_pdf_as_image(filepath)
    elif ext == "dxf":
        return load_dxf_as_image(filepath)
    elif ext in ("png", "jpg", "jpeg", "tif", "tiff"):
        img = cv2.imread(filepath)
        if img is None:
            raise IOError(f"Could not read image: {filepath}")
        return img
    raise ValueError(f"Unsupported file type: .{ext}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – ORIENTATION DETECTION & IMAGE CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _rotation_to_cv2(rotation_deg: int) -> Optional[int]:
    """Map a clockwise rotation angle (degrees) to a cv2.ROTATE_* constant."""
    # We want to UNDO the stored rotation, so we rotate the image by the
    # negative / complement amount.
    correction = rotation_deg % 360
    return {
        0:   None,
        90:  cv2.ROTATE_90_COUNTERCLOCKWISE,   # stored 90° CW  → correct CCW
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,           # stored 270° CW → correct CW
    }.get(correction)


def correct_image_orientation(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Rotate `image` to undo a stored `rotation_deg`-degree CW rotation."""
    op = _rotation_to_cv2(rotation_deg)
    if op is None:
        return image
    print(f"  Correcting image orientation by rotating {(360 - rotation_deg) % 360}° "
          f"(stored rotation was {rotation_deg}°)")
    return cv2.rotate(image, op)


# ── 2a. PDF orientation ───────────────────────────────────────────────────────

def detect_pdf_orientation(filepath: str) -> int:
    """
    Return the stored clockwise rotation of page-0 (0 / 90 / 180 / 270).

    Strategy:
      1. page.rotation  — most reliable; set by the PDF creator.
      2. Voting on text 'dir' vectors — fallback for rare PDFs where
         page.rotation=0 but content is rotated via a CTM.
    """
    doc  = fitz.open(filepath)
    page = doc.load_page(0)

    # Primary: explicit page rotation stored in PDF
    page_rot = page.rotation % 360
    if page_rot != 0:
        print(f"  PDF page.rotation = {page_rot}°")
        return page_rot

    # Fallback: vote on text direction vectors
    dir_map = {
        (1,  0):  0,
        (0, -1): 90,    # text going "up" in PDF coords = 90° CW content
        (-1, 0): 180,
        (0,  1): 270,
    }
    votes: Counter = Counter()
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            dv = tuple(round(c) for c in line.get("dir", (1, 0)))
            angle = dir_map.get(dv, 0)
            span_len = sum(len(s.get("text", "")) for s in line.get("spans", []))
            votes[angle] += span_len   # weight by character count

    dominant = votes.most_common(1)[0][0] if votes else 0
    if dominant != 0:
        print(f"  PDF text-dir vote → {dominant}° (dir-vector fallback)")
    return dominant


def derotate_pdf_words(words: list, page) -> list:
    """
    Transform word bounding boxes from PDF internal space into the visual
    pixel coordinate system that PyMuPDF's get_pixmap() already produces.

    PyMuPDF's get_pixmap() applies page.rotation automatically, but
    get_text('words') returns coordinates in the UN-rotated mediabox space.
    Multiplying each bbox by page.derotation_matrix aligns them.
    """
    drm = page.derotation_matrix
    corrected = []
    for w in words:
        r = fitz.Rect(w[:4]) * drm
        corrected.append((r.x0, r.y0, r.x1, r.y1) + w[4:])
    return corrected


# ── 2b. DXF orientation ───────────────────────────────────────────────────────

def detect_dxf_orientation(filepath: str) -> int:
    """
    Vote on TEXT entity .dxf.rotation values to determine the dominant
    text angle in the DXF.  Returns 0 / 90 / 180 / 270.
    """
    doc = ezdxf.readfile(filepath)
    ms  = doc.modelspace()
    votes: Counter = Counter()
    for e in ms:
        if e.dxftype() in ("TEXT", "MTEXT"):
            raw = getattr(e.dxf, "rotation", 0.0)
            # Snap to nearest cardinal direction
            snapped = round(raw / 90) * 90 % 360
            votes[snapped] += 1
    dominant = votes.most_common(1)[0][0] if votes else 0
    print(f"  DXF text-rotation vote: {dict(votes)} → dominant={dominant}°")
    return dominant


# ── 2c. Raster image orientation (OCR-based heuristic) ───────────────────────

def detect_raster_orientation_from_ocr(ocr_results: list) -> int:
    """
    Heuristic: if most OCR bboxes are taller than they are wide, the image
    is likely rotated 90° or 270°.  We return 90 or 270 based on whether
    the text cluster is on the left or right half.
    (This is approximate; only used when no metadata is available.)
    """
    tall = 0
    wide = 0
    for bbox, text, _ in ocr_results:
        if not (text or "").strip():
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        if h > w * 1.5:
            tall += 1
        elif w > h * 1.5:
            wide += 1
    ratio = tall / (tall + wide) if (tall + wide) > 0 else 0
    print(f"  Raster orientation heuristic: tall={tall} wide={wide} ratio={ratio:.2f}")
    if ratio > 0.6:
        return 90    # caller may refine to 270 if needed
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – OCR PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Convert to greyscale and enhance contrast before passing to EasyOCR.

    Steps:
      1. Greyscale conversion — OCR engines work on luminance; colour adds
         noise.
      2. CLAHE (Contrast Limited Adaptive Histogram Equalisation) — lifts
         local contrast without washing out bright regions.
      3. Adaptive thresholding (optional) — binarise the image.  We keep
         this as a greyscale (not boolean) so EasyOCR can still use it.

    The colour image is kept separately for the debug overlay / crop output.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE on L channel
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq     = clahe.apply(gray)

    # Adaptive binarisation to handle uneven lighting (CAD drawings with
    # coloured backgrounds or faint lines)
    binary = cv2.adaptiveThreshold(
        eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=10
    )
    # Convert back to 3-channel so EasyOCR's internal pipeline is happy
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – TITLE-BLOCK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. Edge-density helper ───────────────────────────────────────────────────

def find_title_block_by_density(
    points: List[Tuple[float, float]],
    coord_range: Tuple[float, float, float, float],
    edge_margin: float = 0.30,
    padding_frac: float = 0.02,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Find the page edge strip (right / left / bottom / top) that contains
    the most text points.  Returns a bounding box in the same coordinate
    system as coord_range, snapped to the page edge.
    """
    if not points:
        return None

    x0, y0, x1, y1 = coord_range
    W = x1 - x0;  H = y1 - y0

    strips = {
        "right":  lambda p: p[0] >= x1 - W * edge_margin,
        "left":   lambda p: p[0] <= x0 + W * edge_margin,
        "bottom": lambda p: p[1] >= y1 - H * edge_margin,
        "top":    lambda p: p[1] <= y0 + H * edge_margin,
    }

    best, best_n = None, 0
    strip_pts: dict = {}
    for name, pred in strips.items():
        pts = [p for p in points if pred(p)]
        strip_pts[name] = pts
        if len(pts) > best_n:
            best_n = len(pts); best = name

    if best is None or best_n == 0:
        print("  ⚠  No edge strip with text found.")
        return None
    print(f"  [OK] Winning edge: '{best}' ({best_n} text items)")

    pts   = strip_pts[best]
    xs    = [p[0] for p in pts]; ys = [p[1] for p in pts]
    pad_x = W * padding_frac;   pad_y = H * padding_frac

    tb_x0 = max(x0, min(xs) - pad_x); tb_y0 = max(y0, min(ys) - pad_y)
    tb_x1 = min(x1, max(xs) + pad_x); tb_y1 = min(y1, max(ys) + pad_y)

    snap = W * 0.03
    if abs(tb_x1 - x1) < snap: tb_x1 = x1
    if abs(tb_x0 - x0) < snap: tb_x0 = x0
    if abs(tb_y1 - y1) < snap: tb_y1 = y1
    if abs(tb_y0 - y0) < snap: tb_y0 = y0

    return (tb_x0, tb_y0, tb_x1, tb_y1)


# ── 4b. PDF native ────────────────────────────────────────────────────────────

def _pdf_extract_lines(page) -> list:
    """
    Extract text as LINE-level entries using rawdict, preserving exact font
    sizes.  Returns list of (x0, y0, x1, y1, line_text, max_font_size_pt)
    in the UN-rotated PDF coordinate space (caller applies derotation_matrix).

    Why rawdict and not 'words'?
      • 'words' splits into individual tokens with nearly-identical bbox heights,
        making font-size comparison useless.
      • rawdict preserves span.size (actual font point size) and groups chars
        into logical lines, so "1B FLOOR BEAM DETAILS" becomes one entry with
        size=8.28pt instead of four separate 12.3pt fragments.
    """
    d = page.get_text("rawdict")
    lines_out = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            # Assemble text char-by-char (handles RTL / CJK / outlined fonts)
            text = "".join(
                c.get("c", "")
                for span in spans
                for c in span.get("chars", [])
            ).strip()
            if not text:
                continue
            font_size = max(s.get("size", 0.0) for s in spans)
            bbox = line["bbox"]   # (x0, y0, x1, y1) in PDF space
            lines_out.append((bbox[0], bbox[1], bbox[2], bbox[3], text, font_size))
    return lines_out


def detect_title_block_from_pdf_native(
    filepath: str, image: np.ndarray
) -> Tuple[Optional[BBox], list, bool]:
    """
    Extract TEXT LINES with PyMuPDF rawdict (preserving exact font sizes),
    apply derotation_matrix, then run density detection.

    Entry format returned: (tl_px, br_px, text, font_size_pt)
    The 4th element (font_size_pt) is used by extract_drawing_title().
    """
    doc  = fitz.open(filepath)
    page = doc.load_page(0)
    W_pt = page.rect.width;  H_pt = page.rect.height
    H_px, W_px = image.shape[:2]
    drm  = page.derotation_matrix

    raw_lines = _pdf_extract_lines(page)
    print(f"  Native PDF text: {len(raw_lines)} lines")
    if len(raw_lines) < 5:
        print("  ⚠  Too few native lines – will fall back to OCR")
        return None, [], False

    def pt_to_px(x_pt: float, y_pt: float):
        return (int(x_pt / W_pt * W_px), int(y_pt / H_pt * H_px))

    # Apply derotation_matrix to each line bbox
    lines_derot = []
    for x0, y0, x1, y1, text, fsz in raw_lines:
        r = fitz.Rect(x0, y0, x1, y1) * drm
        # After derotation the coordinate space matches the rendered image:
        # (0,0)=top-left, x→right, y→down
        lines_derot.append((r.x0, r.y0, r.x1, r.y1, text, fsz))

    # Use line centres for edge-density detection
    centres = [((l[0]+l[2])/2, (l[1]+l[3])/2) for l in lines_derot]
    tb_pt   = find_title_block_by_density(centres, (0, 0, W_pt, H_pt))
    if tb_pt is None:
        return None, [], True

    tl_tb = pt_to_px(tb_pt[0], tb_pt[1]); br_tb = pt_to_px(tb_pt[2], tb_pt[3])
    tb_px: BBox = (tl_tb[0], tl_tb[1], br_tb[0], br_tb[1])

    bx0, by0, bx1, by1 = tb_pt
    entries_px = []
    for x0, y0, x1, y1, text, fsz in lines_derot:
        cx = (x0 + x1) / 2;  cy = (y0 + y1) / 2
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            # Entry: (tl_px, br_px, text, font_size_pt)
            entries_px.append((pt_to_px(x0, y0), pt_to_px(x1, y1), text, fsz))

    return tb_px, entries_px, True


# ── 4c. DXF vector ────────────────────────────────────────────────────────────

@dataclass
class DXFExtent:
    x0: float; y0: float; x1: float; y1: float
    img_w: int; img_h: int

    def to_pixel(self, dxf_x: float, dxf_y: float) -> Tuple[int, int]:
        px = int((dxf_x - self.x0) / (self.x1 - self.x0) * self.img_w)
        py = int((1.0 - (dxf_y - self.y0) / (self.y1 - self.y0)) * self.img_h)
        return (max(0, min(px, self.img_w-1)), max(0, min(py, self.img_h-1)))

    def box_to_pixel(self, box: DXFBox) -> BBox:
        p_lo = self.to_pixel(box[0], box[1])
        p_hi = self.to_pixel(box[2], box[3])
        return (min(p_lo[0], p_hi[0]), min(p_lo[1], p_hi[1]),
                max(p_lo[0], p_hi[0]), max(p_lo[1], p_hi[1]))


def _full_span_boundary(ms, total_w: float, total_h: float,
                        span_ratio: float = 0.40):
    """Detect full-width/height separator lines that bound the content area."""
    min_h = total_w * span_ratio;  min_v = total_h * span_ratio
    h_rights, h_lefts, v_tops, v_bots = [], [], [], []

    for e in ms:
        t = e.dxftype()
        segs = []
        if t == "LINE":
            segs = [(e.dxf.start.x, e.dxf.start.y,
                     e.dxf.end.x,   e.dxf.end.y)]
        elif t == "LWPOLYLINE":
            pts = list(e.get_points())
            segs = [(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                    for i in range(len(pts)-1)]
        for x1, y1, x2, y2 in segs:
            dx = abs(x2-x1); dy = abs(y2-y1)
            if dx >= min_h:
                h_rights.append(max(x1, x2)); h_lefts.append(min(x1, x2))
            if dy >= min_v:
                v_tops.append(max(y1, y2));   v_bots.append(min(y1, y2))

    right = float(np.median(h_rights)) if h_rights else None
    left  = float(np.median(h_lefts))  if h_lefts  else None
    top   = float(np.median(v_tops))   if v_tops   else None
    bot   = float(np.median(v_bots))   if v_bots   else None
    return right, left, top, bot


def detect_title_block_from_dxf(
    filepath: str, image: np.ndarray
) -> Tuple[Optional[BBox], list]:
    doc = ezdxf.readfile(filepath)
    ms  = doc.modelspace()

    all_pts:   list = []
    # text_data: (x, y, text, dxf_height)  ← height now included
    text_data: list = []

    for e in ms:
        t = e.dxftype()
        if t == "TEXT":
            x, y = e.dxf.insert.x, e.dxf.insert.y
            h    = getattr(e.dxf, "height", 0.0)
            all_pts.append((x, y))
            text_data.append((x, y, e.dxf.text or "", h))
        elif t == "MTEXT":
            x, y = e.dxf.insert.x, e.dxf.insert.y
            h    = getattr(e.dxf, "char_height", 0.0)
            all_pts.append((x, y))
            text_data.append((x, y, (e.text or "")[:80], h))
        elif t == "LINE":
            for pt in (e.dxf.start, e.dxf.end): all_pts.append((pt.x, pt.y))
        elif t == "LWPOLYLINE":
            for pt in e.get_points():            all_pts.append((pt[0], pt[1]))

    if not all_pts:
        return None, []

    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    ext = DXFExtent(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys),
                    img_w=image.shape[1], img_h=image.shape[0])
    total_w = ext.x1 - ext.x0; total_h = ext.y1 - ext.y0

    print(f"  DXF extent: ({ext.x0:.0f},{ext.y0:.0f}) → ({ext.x1:.0f},{ext.y1:.0f})")
    print(f"  TEXT entities: {len(text_data)}")

    # Stage 1 – separator lines
    re, le, te, be = _full_span_boundary(ms, total_w, total_h)
    min_gap_x = total_w * 0.02; min_gap_y = total_h * 0.02
    candidates: dict = {}
    if re and (ext.x1 - re) > min_gap_x:
        candidates["right"]  = [(x,y,t,h) for x,y,t,h in text_data if x > re]
    if le and (le - ext.x0) > min_gap_x:
        candidates["left"]   = [(x,y,t,h) for x,y,t,h in text_data if x < le]
    if te and (ext.y1 - te) > min_gap_y:
        candidates["top"]    = [(x,y,t,h) for x,y,t,h in text_data if y > te]
    if be and (be - ext.y0) > min_gap_y:
        candidates["bottom"] = [(x,y,t,h) for x,y,t,h in text_data if y < be]

    tb_texts: list = []
    if candidates:
        best = max(candidates, key=lambda k: len(candidates[k]))
        tb_texts = candidates[best]
        print(f"  Stage-1: '{best}' edge — {len(tb_texts)} text(s) outside content boundary")

    # Stage 2 – density fallback
    if not tb_texts:
        print("  Stage-1 found nothing → density fallback")
        tb_dxf = find_title_block_by_density(
            [(x,y) for x,y,_,_ in text_data],
            coord_range=(ext.x0, ext.y0, ext.x1, ext.y1)
        )
        if tb_dxf is None:
            return None, []
        bx0, by0, bx1, by1 = tb_dxf
        tb_texts = [(x,y,t,h) for x,y,t,h in text_data
                    if bx0 <= x <= bx1 and by0 <= y <= by1]

    if not tb_texts:
        return None, []

    pad = total_w * 0.01
    tx = [x for x,_,_,_ in tb_texts]; ty = [y for _,y,_,_ in tb_texts]
    tb_dxf_box: DXFBox = (
        max(ext.x0, min(tx)-pad), max(ext.y0, min(ty)-pad),
        min(ext.x1, max(tx)+pad), min(ext.y1, max(ty)+pad),
    )
    tb_px = ext.box_to_pixel(tb_dxf_box)

    # Scale DXF height units → approximate pixels for the entry bbox.
    # DXF height is in drawing units; we scale by image_h / total_dxf_h.
    dxf_to_px = image.shape[0] / total_h
    char_w    = total_w * 0.005   # still used for width estimate only

    entries_px = []
    for x, y, t, h in tb_texts:
        h_px  = max(4, int(h * dxf_to_px))   # pixel height from REAL DXF height
        w_px_approx = int(len(t) * char_w * dxf_to_px)
        tl_px = ext.to_pixel(x,                y + h)
        br_px = ext.to_pixel(x + len(t)*char_w, y)
        # Entry: (tl_px, br_px, text, dxf_height_as_pt_equiv)
        # We store dxf_height directly — scoring treats it as the size signal
        entries_px.append((tl_px, br_px, t, h))

    return tb_px, entries_px


# ── 4d. OCR fallback ──────────────────────────────────────────────────────────

def run_ocr(image: np.ndarray) -> list:
    import easyocr
    print("  Running EasyOCR (CPU)…")
    reader = easyocr.Reader(["en"], gpu=False)
    proc   = preprocess_for_ocr(image)
    return reader.readtext(proc)


def detect_title_block_from_ocr(
    ocr_results: list, image: np.ndarray
) -> Tuple[Optional[BBox], list]:
    H, W = image.shape[:2]
    centres = []
    for bbox, text, _ in ocr_results:
        if not (text or "").strip(): continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        centres.append((sum(xs)/4, sum(ys)/4))

    tb_f = find_title_block_by_density(centres, (0, 0, W, H))
    if tb_f is None:
        return None, []

    tb_px: BBox = tuple(int(v) for v in tb_f)  # type: ignore
    bx0, by0, bx1, by1 = tb_f
    inside = []
    for bbox, text, prob in ocr_results:
        if not (text or "").strip(): continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        cx, cy = sum(xs)/4, sum(ys)/4
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            tl = (int(min(xs)), int(min(ys))); br = (int(max(xs)), int(max(ys)))
            # 4th element: bbox height as proxy for font size (None = not native)
            inside.append((tl, br, text, None))

    return tb_px, inside


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – DEBUG OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def generate_debug_pdf(
    image: np.ndarray,
    title_block_bbox: Optional[BBox],
    text_entries: list,
    output_path: str,
    source_label: str = "",
    title_candidates: list = None
):
    if title_candidates is None:
        title_candidates = []

    debug = image.copy()

    # 1. Title block outline – thick green
    if title_block_bbox:
        x1, y1, x2, y2 = title_block_bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 200, 0), 6)
        label = f"TITLE BLOCK ({source_label})" if source_label else "TITLE BLOCK"
        cv2.putText(debug, label, (x1, max(y1-15, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

    # 2. All text entries – thin orange (support both 3- and 4-tuples)
    for entry in text_entries:
        tl, br, txt = entry[0], entry[1], entry[2]
        cv2.rectangle(debug, tl, br, (200, 80, 0), 1)
        display = (txt[:20]+"…") if len(txt) > 20 else txt
        cv2.putText(debug, display, (tl[0], max(tl[1]-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 80, 0), 1)

    # 3. Title candidates – thick red, annotated with rank + score
    for i, c in enumerate(title_candidates):
        tl, br = c['box']
        color  = (0, 0, 255) if i == 0 else (180, 0, 180)   # red=best, magenta=others
        thick  = 4 if i == 0 else 2
        cv2.rectangle(debug, tl, br, color, thick)
        tag = f"#{i+1} {c['score']:.1f} | {c['text'][:30]}"
        cv2.putText(debug, tag, (tl[0], max(tl[1]-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path, "PDF", resolution=100.0)
    print(f"  Debug PDF → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def extract_drawing_title(text_entries: list, title_block_bbox: Optional[BBox] = None) -> list:
    """
    Identify the drawing title/name from title-block text entries using three
    independent signals combined into a single score.  Language-agnostic by
    design — no language-specific string matching is required for the primary
    path; multilingual label keywords are used as a confirmatory bonus only.

    Entry format accepted: (tl_px, br_px, text, font_size)
      • font_size  – pt (PDF span size), DXF height units, or None (OCR/fallback)
      • When font_size is None, bbox height (br[1]-tl[1]) is used as a proxy.

    Returns a list of candidate dicts sorted by score (highest first):
      {
        'text':         str,    # cleaned text
        'raw_text':     str,    # original (pre-cleanup) text
        'score':        float,
        'score_detail': dict,   # breakdown for debugging
        'box':          (tl_px, br_px),
        'font_size':    float,
      }

    ── Signal 1: Native font size ────────────────────────────────────────────
    Exact pt/DXF-unit size, not a pixel estimate.  Normalised 0–1 across
    all valid entries inside the title block.

    ── Signal 2: Position within title block ─────────────────────────────────
    Drawing titles sit in the lower 40–85% of the block.  Entries above the
    midpoint get no bonus; those in the lower quarter get up to +1.0.

    ── Signal 3: Text quality + English keyword fallback ─────────────────────
    Hard filters (entry excluded):
      • Pure numbers, single/double characters, coordinate labels
      • Drawing/revision numbers  (e.g. 1405/PH2/SVMC…)
      • Rebar notation            (e.g. 3T25, T10-250+HOOK)
      • Cross-section refs        (e.g. 27-27)
      • Dimension ratios          (e.g. 0.3 x L1, 0.1xL)
      • Continuation notes        (cont'd, continued, below, above)
      • Field labels only         (Issued by:, Name, Date:, Scale:)
      • Company/person markers    (IC:, SDN BHD, PTY LTD)
      • Date strings              (5 SEPT 2018, 16.4.2014)

    Soft bonuses / penalties:
      +2.0  Text immediately below a multilingual title-label keyword
      +1.5  Contains drawing-type keywords (English fallback, see list)
      +0.5  ALL-CAPS, multi-word (typical formal title style)
      −1.0  Very long text (> 50 chars) — project descriptions, not titles
      −0.5  Contains floor-count pattern (e.g. "4 TINGKAT", "18 STOREY")

    Multilingual label keywords — if these appear as text in the block, the
    entry immediately below them gets +2.0:
      EN  : DRAWING TITLE, DRAWING NAME, DWG TITLE, TITLE, DESCRIPTION
      MS  : TAJUK LUKISAN, NAMA LUKISAN, TAJUK PROJEK, TAJUK
      ZH  : 图纸名称, 图名, 工程名称
      ID  : NAMA GAMBAR, JUDUL GAMBAR, JUDUL
      AR  : عنوان الرسم
      FR  : TITRE DU DESSIN, INTITULÉ
      DE  : ZEICHNUNGSTITEL, BEZEICHNUNG
    """
    import re

    if not text_entries:
        return []

    # ── DXF special-code stripper ──────────────────────────────────────────
    # DXF embeds formatting codes: %%U=underline, %%O=overline, %%D=degree,
    # %%P=plusminus, %%C=diameter.  Strip them before any scoring.
    _DXF_CODES = re.compile(r'%%[A-Za-z]', re.I)
    def _strip_dxf(text: str) -> str:
        return _DXF_CODES.sub('', text).strip()

    # ── Normalise entry format (3-tuple or 4-tuple) ────────────────────────
    norm = []
    for entry in text_entries:
        tl, br, raw_text = entry[0], entry[1], entry[2]
        fsz = entry[3] if len(entry) > 3 else None
        clean = _strip_dxf(raw_text)
        if clean:
            norm.append((tl, br, clean, raw_text, fsz))

    # ── Hard-filter regexes ────────────────────────────────────────────────
    RE_DRAWING_NO   = re.compile(r'[A-Z0-9]{2,}/[A-Z0-9]{2,}')
    RE_REBAR        = re.compile(r'\b\d+[TR]\d+\b|\b[TR]\d+-\d+\b', re.I)
    RE_CROSS_REF    = re.compile(r'^\d+-\d+$')
    RE_DIM_RATIO    = re.compile(r'\d+\.?\d*\s*[xX]\s*[A-Z]\d*')
    RE_COORD_LABEL  = re.compile(r'^[A-Z]\d*$')
    RE_DATE_STR     = re.compile(
        r'\b\d{1,2}[\s./]\d{1,2}[\s./]\d{2,4}\b'          # 16.4.2014
        r'|\b\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b',
        re.I)
    RE_LABEL_ONLY   = re.compile(
        r'^(issued\s+by|verified\s+by|checked\s+by|drawn\s+by|'
        r'approved\s+by|name|initial|date|scale|rev\.?|no\.?|'
        r'signature|tandatangan)\s*:?\s*$', re.I)
    RE_COMPANY      = re.compile(r'SDN\s+BHD|PTY\s+LTD|PTE\s+LTD|INC\b|LLC\b|GmbH|IC\s*:\s*\d', re.I)
    RE_CONTINUATION = re.compile(r"cont'?d\b|continued|^below$|^above$", re.I)
    RE_PURE_NUMBER  = re.compile(r'^[\d\s.,:/\\-]+$')

    def _is_hard_filtered(text: str) -> bool:
        if len(text) <= 2:                    return True
        if RE_PURE_NUMBER.match(text):        return True
        if RE_COORD_LABEL.match(text):        return True
        if RE_CROSS_REF.match(text):          return True
        if RE_REBAR.search(text):             return True
        if RE_DIM_RATIO.search(text):         return True
        if RE_DRAWING_NO.search(text):        return True
        if RE_DATE_STR.search(text):          return True
        if RE_LABEL_ONLY.match(text):         return True
        if RE_CONTINUATION.search(text):      return True
        if RE_COMPANY.search(text):           return True
        return False

    # ── Multilingual title-label set ──────────────────────────────────────
    TITLE_LABELS = {
        'drawing title', 'drawing name', 'dwg title', 'title', 'description',
        'tajuk lukisan', 'nama lukisan', 'tajuk projek', 'tajuk',
        '图纸名称', '图名', '工程名称',
        'nama gambar', 'judul gambar', 'judul',
        'عنوان الرسم', 'titre du dessin', 'intitulé',
        'zeichnungstitel', 'bezeichnung',
    }

    # ── Engineering drawing-type keywords (English fallback bonus) ─────────
    # Deliberately excludes generic words that also appear in project
    # descriptions (e.g. PODIUM, HOSPITAL, TOWER, BLOCK) to avoid false
    # positives on multilingual project-description text.
    ENG_KEYWORDS = re.compile(
        r'\b(FLOOR\s+BEAM|ROOF\s+BEAM|GROUND\s+BEAM|'
        r'FLOOR\s+PLAN|ROOF\s+PLAN|SITE\s+PLAN|'
        r'FLOOR\s+SLAB|ROOF\s+SLAB|'
        r'STAIR|RAMP|SECTION|ELEVATION|DETAIL|SCHEDULE|'
        r'LAYOUT|FRAMING|REINFORCEMENT|STRUCTURAL|MECHANICAL|'
        r'ELECTRICAL|PLUMBING|HVAC|DRAINAGE|LANDSCAPE|CIVIL|'
        r'GENERAL\s+ARRANGEMENT|GENERAL\s+NOTES|'
        r'ASSEMBLY|FABRICATION|INSTALLATION|SETTING.?OUT|AS.?BUILT|'
        r'COMPOSITE|ENLARGED|REFLECTED|DEMOLITION|'
        r'FOUNDATION|FOOTING|PILE\s+CAP|TIE\s+BEAM|'
        r'COLUMN\s+SCHEDULE|BEAM\s+DETAILS?|SLAB\s+DETAILS?)\b',
        re.I
    )

    # Floor-count patterns that signal project descriptions, not titles
    RE_FLOOR_COUNT = re.compile(
        r'\b\d+\s*(TINGKAT|STOREY|STOREYS|STORY|STORIES|FLOOR|FLOORS|LEVEL)\b', re.I
    )

    # ── Collect valid entries ──────────────────────────────────────────────
    valid  = []
    label_ys = []

    for tl, br, text, raw_text, fsz in norm:
        if _is_hard_filtered(text):
            continue
        eff_size = fsz if fsz is not None else max(1, br[1] - tl[1])
        cy = (tl[1] + br[1]) / 2
        valid.append({'tl': tl, 'br': br, 'text': text, 'raw_text': raw_text,
                      'eff_size': eff_size, 'cy': cy})

    for tl, br, text, raw_text, fsz in norm:
        if text.lower().strip().rstrip(':') in TITLE_LABELS:
            label_ys.append((tl[1] + br[1]) / 2)

    if not valid:
        return []

    # ── Normalise size and Y ───────────────────────────────────────────────
    sizes  = [e['eff_size'] for e in valid]
    min_sz = min(sizes); max_sz = max(sizes)
    sz_rng = max_sz - min_sz if max_sz != min_sz else 1.0

    all_ys = [e['cy'] for e in valid]
    min_y  = min(all_ys); max_y = max(all_ys)
    y_rng  = max_y - min_y if max_y != min_y else 1.0

    # ── Score ──────────────────────────────────────────────────────────────
    scored = []
    for e in valid:
        text   = e['text']
        cy     = e['cy']
        eff_sz = e['eff_size']

        sz_norm    = (eff_sz - min_sz) / sz_rng
        y_norm     = (cy - min_y) / y_rng
        pos_bonus  = max(0.0, (y_norm - 0.5) * 2.0)

        prox_bonus = 0.0
        for label_y in label_ys:
            gap = cy - label_y
            if 0 < gap < (max_y - min_y) * 0.15:
                prox_bonus = 2.0; break

        kw_bonus      = 1.5 if ENG_KEYWORDS.search(text) else 0.0
        words_list    = text.split()
        fmt_bonus     = 0.5 if (text == text.upper() and len(words_list) >= 2) else 0.0
        long_penalty  = -1.0 if len(text) > 50 else 0.0
        floor_penalty = -0.5 if RE_FLOOR_COUNT.search(text) else 0.0

        score = ((sz_norm * 2.5) + pos_bonus + prox_bonus
                 + kw_bonus + fmt_bonus + long_penalty + floor_penalty)

        detail = {
            'sz_norm':      round(sz_norm, 3),
            'pos_bonus':    round(pos_bonus, 3),
            'prox_bonus':   prox_bonus,
            'kw_bonus':     kw_bonus,
            'fmt_bonus':    fmt_bonus,
            'long_penalty': long_penalty,
            'floor_penalty':floor_penalty,
            'eff_size':     round(eff_sz, 2),
        }
        scored.append({
            'text':         text,
            'raw_text':     e['raw_text'],
            'score':        round(score, 3),
            'score_detail': detail,
            'box':          (e['tl'], e['br']),
            'font_size':    eff_sz,
        })

    scored.sort(key=lambda x: -x['score'])
    return scored[:5]

def process_drawing(file_path: str, output_dir: str = "."):
    ext      = file_path.lower().rsplit(".", 1)[-1]
    base     = os.path.splitext(os.path.basename(file_path))[0]
    out_name = os.path.join(output_dir, base)

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    # ── Step 1: Detect orientation BEFORE rasterising ────────────────────────
    rotation_deg = 0
    if ext == "pdf":
        rotation_deg = detect_pdf_orientation(file_path)
    elif ext == "dxf":
        rotation_deg = detect_dxf_orientation(file_path)
    # Raster images: orientation detected AFTER OCR (see below)

    # ── Step 2: Load and rasterise ────────────────────────────────────────────
    image = load_file(file_path)
    print(f"  Raw image: {image.shape[1]} × {image.shape[0]} px")

    # ── Step 3: Correct orientation ───────────────────────────────────────────
    # For PDF: PyMuPDF get_pixmap() already applies page.rotation visually,
    # so the rendered image IS upright.  No extra rotation needed here.
    # For DXF:  if the dominant TEXT rotation != 0 we do need to rotate.
    corrected_image = image
    if ext == "dxf" and rotation_deg != 0:
        corrected_image = correct_image_orientation(image, rotation_deg)
    # Note: PDF coordinate derotation is handled inside
    # detect_title_block_from_pdf_native() via derotate_pdf_words().

    # ── Step 4: Title-block detection ─────────────────────────────────────────
    title_block_px: Optional[BBox] = None
    text_entries:   list           = []
    source_label:   str            = ""

    if ext == "dxf":
        print("[DXF] Vector-based detection…")
        title_block_px, text_entries = detect_title_block_from_dxf(
            file_path, corrected_image
        )
        source_label = "DXF vector"

    elif ext == "pdf":
        print("[PDF] Native text detection…")
        title_block_px, text_entries, success = detect_title_block_from_pdf_native(
            file_path, corrected_image
        )
        if success:
            source_label = f"PDF native (rot={rotation_deg}°)"
        else:
            print("[PDF] → OCR fallback…")
            ocr_results    = run_ocr(corrected_image)
            # Check raster orientation via OCR bboxes
            r_rot          = detect_raster_orientation_from_ocr(ocr_results)
            if r_rot != 0:
                corrected_image = correct_image_orientation(corrected_image, r_rot)
                ocr_results     = run_ocr(corrected_image)   # re-run on upright image
            title_block_px, text_entries = detect_title_block_from_ocr(
                ocr_results, corrected_image
            )
            source_label = "OCR fallback"

    else:
        print("[Image] OCR…")
        ocr_results = run_ocr(corrected_image)
        r_rot       = detect_raster_orientation_from_ocr(ocr_results)
        if r_rot != 0:
            corrected_image = correct_image_orientation(corrected_image, r_rot)
            ocr_results     = run_ocr(corrected_image)
        title_block_px, text_entries = detect_title_block_from_ocr(
            ocr_results, corrected_image
        )
        source_label = "OCR"

    # ── Step 5: Report ────────────────────────────────────────────────────────
    title_candidates = [] # Initialize empty list

    if title_block_px:
        x1, y1, x2, y2 = title_block_px
        print(f"\n[OK] Title block @ px ({x1},{y1}) → ({x2},{y2})  "
              f"[{x2-x1}×{y2-y1} px]  source: {source_label}")
        print(f"  Text entries inside: {len(text_entries)}")
        
        # Extract candidates with new multi-signal scorer
        title_candidates = extract_drawing_title(text_entries, title_block_bbox=title_block_px)
        print("\n  Drawing Title Candidates (multi-signal score):")
        for i, c in enumerate(title_candidates[:3], 1):
            d = c['score_detail']
            print(f"    {i}. score={c['score']:.2f}  "
                  f"[sz={d['sz_norm']:.2f}×2.5  pos={d['pos_bonus']:.2f}  "
                  f"kw={d['kw_bonus']:.1f}  lbl={d['prox_bonus']:.1f}  fmt={d['fmt_bonus']:.1f}]")
            print(f"       font={d['eff_size']:.1f}  → \"{c['text']}\"")
        if title_candidates:
            print(f"\n  [OK] Best title: \"{title_candidates[0]['text']}\"  "
                  f"(score={title_candidates[0]['score']:.2f})")
            
    else:
        print("\n[FAIL] Title block NOT detected.")

    # ── Step 6: Save outputs (always use the orientation-corrected image) ─────
    upright_path = f"{out_name}_upright.png"
    cv2.imwrite(upright_path, corrected_image)
    print(f"  Upright image → {upright_path}")

    debug_path = f"{out_name}_debug.pdf"
    generate_debug_pdf(corrected_image, title_block_px, text_entries,
                       debug_path, source_label, title_candidates=title_candidates)

    if title_block_px:
        x1, y1, x2, y2 = title_block_px
        crop = corrected_image[y1:y2, x1:x2]
        crop_path = f"{out_name}_titleblock_crop.png"
        cv2.imwrite(crop_path, crop)
        print(f"  Crop → {crop_path}")

    json_path = f"{out_name}_titleblock_text.json"
    with open(json_path, "w", encoding="utf-8") as f:
        out_list = []
        for entry in text_entries:
            tl, br, txt = entry[0], entry[1], entry[2]
            fsz = entry[3] if len(entry) > 3 else None
            out_list.append({"tl": list(tl), "br": list(br),
                             "text": txt, "font_size": fsz})
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"  Text JSON → {json_path}")

    return title_block_px, text_entries, corrected_image


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    files = sys.argv[1:] or [
        #r"dwg_compare-5.dxf",
        r"C:\Users\glodon\Desktop\Projects\dwg_compare-6.dxf",
    ]
    for f in files:
        process_drawing(f, output_dir=".")