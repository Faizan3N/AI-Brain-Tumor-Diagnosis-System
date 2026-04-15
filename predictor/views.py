import os
from uuid import uuid4
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from django.conf import settings
from django.http import FileResponse, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense
from keras.utils import get_custom_objects

try:
    from lime import lime_image
except ImportError:
    lime_image = None

try:
    from skimage.segmentation import mark_boundaries
except ImportError:
    mark_boundaries = None


class PatchedDense(Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


get_custom_objects()["Dense"] = PatchedDense


def _log_pipeline(message):
    print(f"[Brain Tumor Pipeline] {message}", flush=True)

MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "model.h5")
model = load_model(MODEL_PATH)
_log_pipeline("Model loaded successfully")
class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}
# Order used in the PDF probability section (matches common clinical listing).
CLASS_PDF_ORDER = ["meningioma", "glioma", "pituitary", "notumor"]

# Footer (two-column template) — edit if your roster changes.
REPORT_AUTHOR_LINE_1 = "Faizan Ali (FA22-BSCS-187)"
REPORT_AUTHOR_LINE_2 = "M. Mudassar (FA22-BSCS-204)"
# Older PDF code paths (or stale .pyc) may still reference this name — keep defined.
REPORT_AUTHORS_LINE = f"{REPORT_AUTHOR_LINE_1} · {REPORT_AUTHOR_LINE_2}"
REPORT_SUPERVISOR_LINE = "Supervised by Ms. Fatima Aslam"
REPORT_DEPARTMENT_LINE = "Department of Computer Science"
REPORT_INSTITUTION_LINE = "Lahore Garrison University, Lahore"
REPORT_DISCLAIMER_HEADING = "Not a substitute for professional medical diagnosis"
REPORT_DISCLAIMER_BODY = (
    "This AI-generated report is a preliminary analysis to assist clinicians and researchers. "
    "Final diagnosis and treatment decisions must be made by qualified healthcare "
    "professionals based on complete clinical evaluation."
)


def _relative_under_media(media_url_path):
    """Strip MEDIA_URL prefix so the path can be joined with MEDIA_ROOT."""
    if not media_url_path:
        return None
    base = settings.MEDIA_URL
    if isinstance(media_url_path, str) and media_url_path.startswith(base):
        return media_url_path[len(base) :].lstrip("/")
    return media_url_path


def _parse_confidence_percent(conf_str):
    try:
        return float(str(conf_str).replace("%", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _prediction_headline(result_str, top_key):
    """Human-readable headline for the diagnosis card."""
    if top_key == "notumor":
        return "No Tumor Indicated"
    name = CLASS_DISPLAY.get(top_key, top_key.title())
    return f"{name} Detected"


def _interpretation_blurb(top_key):
    blurbs = {
        "glioma": (
            "Gliomas arise from glial cells and can appear as heterogeneous masses with "
            "variable intensity on MRI. The model’s signal aligns with patterns commonly "
            "associated with this class in the training distribution."
        ),
        "meningioma": (
            "Meningiomas are typically extra-axial, dural-based lesions. The model "
            "highlights regions consistent with this presentation relative to the "
            "learned decision boundary."
        ),
        "pituitary": (
            "Pituitary-region lesions often localize near the sella. The classifier "
            "emphasizes intensity and shape cues that discriminated this class during training."
        ),
        "notumor": (
            "The model did not assign high probability to any tumor class for this slice. "
            "This does not rule out pathology elsewhere in a full MRI study."
        ),
    }
    return blurbs.get(top_key, blurbs["glioma"])


def download_report_pdf(request):
    """Build a formal multi-section PDF of the last detection (session)."""
    _log_pipeline("Generating PDF report")
    from io import BytesIO

    from reportlab.graphics import renderPDF
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    payload = request.session.get("mri_report")
    if not payload:
        return HttpResponse(
            "No report is available yet. Upload an MRI image and run detection first.",
            status=400,
            content_type="text/plain",
        )

    page_width, page_height = letter
    margin = 36
    gutter = 8
    # Footer y is computed from the quad bottom (fixed y here caused footer text to paint over row 2).
    footer_gap_below_quad = 16
    footer_min_top_y = margin + 58  # minimum y for separator rule (room for footer block above page bottom)
    footer_secondary = colors.HexColor("#4d6b8c")  # grey-blue for affiliation / disclaimer body
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    # Brand palette (navy / clinical blues / success green)
    navy = colors.HexColor("#152238")
    navy_light = colors.HexColor("#1e3a5f")
    panel_blue = colors.HexColor("#e8f1fb")
    panel_green = colors.HexColor("#e8f5e9")
    track = colors.HexColor("#dde3ea")
    fill_bar = colors.HexColor("#1e3a5f")
    text_muted = colors.HexColor("#5c6570")
    success_green = colors.HexColor("#1b5e20")

    now = timezone.localtime(timezone.now())
    report_id = f"BT-{now.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:4].upper()}"

    probs = dict(payload.get("probs") or {})
    top_key = payload.get("top_class")
    if not top_key and payload.get("result"):
        raw = str(payload["result"]).lower()
        if "no tumor" in raw:
            top_key = "notumor"
        else:
            for k in ("glioma", "meningioma", "pituitary", "notumor"):
                if k in raw:
                    top_key = k
                    break
        if not top_key:
            top_key = "glioma"
    elif not top_key:
        top_key = "glioma"
    # Older sessions without per-class probs: show confidence on predicted class only.
    if not probs:
        c = _parse_confidence_percent(payload.get("confidence")) / 100.0
        probs = {k: (c if k == top_key else 0.0) for k in CLASS_PDF_ORDER}
    conf_pct = _parse_confidence_percent(payload.get("confidence"))
    headline = _prediction_headline(payload.get("result", ""), top_key)
    is_no_tumor = top_key == "notumor"

    def path_for_rel(rel):
        if not rel:
            return None
        p = os.path.join(settings.MEDIA_ROOT, rel)
        return p if os.path.isfile(p) else None

    def draw_header():
        h = 88
        pdf.setFillColor(navy)
        pdf.roundRect(0, page_height - h, page_width, h, 8, stroke=0, fill=1)

        # Metadata card (top-right) — dark text on light blue for contrast
        mw, mh = 184, 58
        mx = page_width - margin - mw
        my = page_height - h + 10
        pdf.setFillColor(panel_blue)
        pdf.roundRect(mx, my, mw, mh, 6, stroke=0, fill=1)
        meta_label = navy
        meta_value = colors.HexColor("#0d1b2a")
        lh = 12
        ty = my + mh - 10
        pdf.setFillColor(meta_label)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(mx + 8, ty, "REPORT ID")
        ty -= lh
        pdf.setFillColor(meta_value)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(mx + 8, ty, report_id)
        ty -= lh + 2
        pdf.setFillColor(meta_label)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(mx + 8, ty, "GENERATED ON")
        ty -= lh
        pdf.setFillColor(meta_value)
        pdf.setFont("Helvetica", 10)
        gen_short = now.strftime("%d %b %Y, %H:%M")
        pdf.drawString(mx + 8, ty, gen_short)

        # Left title block
        pdf.setFillColor(colors.HexColor("#b8c5d6"))
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(margin, page_height - 26, "AI BRAIN TUMOR DIAGNOSTIC SYSTEM")
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 17)
        pdf.drawString(margin, page_height - 44, "Brain MRI Diagnostic Report")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(
            margin,
            page_height - 60,
            "VGG16 Transfer Learning · LIME Explainability · Automated Analysis",
        )

    def draw_wrapped_paragraph(text, x, y_top, max_width, line_height, font_name, font_size, max_lines=12):
        """y_top is top baseline area (we work downward). Returns new y below paragraph."""
        from reportlab.pdfbase.pdfmetrics import stringWidth

        pdf.setFont(font_name, font_size)
        words = text.split()
        lines = []
        line = []
        for w in words:
            trial = " ".join(line + [w])
            if stringWidth(trial, font_name, font_size) <= max_width:
                line.append(w)
            else:
                if line:
                    lines.append(" ".join(line))
                line = [w]
                if stringWidth(w, font_name, font_size) > max_width:
                    lines.append(w)
                    line = []
        if line:
            lines.append(" ".join(line))
        lines = lines[:max_lines]
        y = y_top
        for ln in lines:
            pdf.drawString(x, y, ln)
            y -= line_height
        return y

    def draw_donut(cx, cy, size, pct):
        d = Drawing(size, size)
        pie = Pie()
        pie.x = 4
        pie.y = 4
        pie.width = pie.height = size - 8
        pie.data = [max(pct, 0.001), max(100.0 - pct, 0.001)]
        pie.slices[0].fillColor = success_green if not is_no_tumor else navy_light
        pie.slices[1].fillColor = track
        pie.slices.strokeWidth = 0
        pie.innerRadiusFraction = 0.58
        d.add(pie)
        renderPDF.draw(d, pdf, cx - size / 2, cy - size / 2)

    def draw_probability_section(y0):
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(margin, y0, "Class probabilities")
        y = y0 - 17
        bar_left = margin + 112
        bar_width = page_width - margin - bar_left - 50
        row_h = 13
        for key in CLASS_PDF_ORDER:
            p = float(probs.get(key, 0.0))
            label = CLASS_DISPLAY.get(key, key)
            pdf.setFont("Helvetica", 10)
            pdf.drawString(margin, y + 2, label)
            pdf.setFillColor(track)
            pdf.roundRect(bar_left, y, bar_width, row_h, 3, stroke=0, fill=1)
            fill_w = bar_width * min(max(p, 0.0), 1.0)
            if fill_w > 0:
                pdf.setFillColor(fill_bar)
                pdf.roundRect(bar_left, y, max(fill_w, 1.5), row_h, 3, stroke=0, fill=1)
            pdf.setFillColor(colors.black)
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawRightString(page_width - margin, y + 2, f"{p * 100:.2f}%")
            y -= 20
        return y - 4

    def draw_image_row(y_top):
        """Three columns: original, XAI, LIME. y_top is baseline for section title."""
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(margin, y_top, "Visualizations")
        y_label = y_top - 15
        col_w = (page_width - 2 * margin - 2 * gutter) / 3.0
        max_h = 118  # slightly shorter so Findings 2×2 + footer fit without overlap
        titles = [
            ("Original MRI", payload.get("upload_rel")),
            ("XAI heatmap (model attention)", payload.get("xai_rel")),
            ("LIME explanation", payload.get("lime_rel")),
        ]
        x0 = margin
        bottoms = []
        for i, (title, rel) in enumerate(titles):
            xc = x0 + i * (col_w + gutter)
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(xc, y_label, title)
            img_y_top = y_label - 10
            path = path_for_rel(rel)
            bottom = img_y_top
            if path:
                try:
                    img = ImageReader(path)
                    iw, ih = img.getSize()
                    scale = min(col_w / float(iw), max_h / float(ih), 1.0)
                    w, h = float(iw) * scale, float(ih) * scale
                    pdf.setFillColor(colors.HexColor("#0d0d0d"))
                    pdf.roundRect(xc - 2, img_y_top - h - 2, w + 4, h + 4, 4, stroke=0, fill=1)
                    pdf.drawImage(img, xc, img_y_top - h, width=w, height=h)
                    bottom = img_y_top - h - 8
                except Exception:
                    pdf.setFont("Helvetica-Oblique", 8)
                    pdf.drawString(xc, img_y_top - 40, "Image unavailable")
                    bottom = img_y_top - 60
            else:
                pdf.setFont("Helvetica-Oblique", 8)
                pdf.setFillColor(text_muted)
                pdf.drawString(xc, img_y_top - 40, "Not available")
                pdf.setFillColor(colors.black)
                bottom = img_y_top - 60
            bottoms.append(bottom)
        return min(bottoms) if bottoms else y_label - 55

    # --- Single page layout ---
    draw_header()
    y = page_height - 100

    # Summary row: two cards
    card_h = 102
    left_w = (page_width - 2 * margin - gutter) * 0.52
    right_x = margin + left_w + gutter

    pdf.setFillColor(panel_blue)
    pdf.roundRect(margin, y - card_h, left_w, card_h, 10, stroke=0, fill=1)
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin + 10, y - 17, "Analysis summary")
    pdf.setFont("Helvetica", 9)
    summary_lines = [
        ("AI model", "VGG16 (transfer learning, 4-class head)"),
        ("Input", "Single 2D slice, resized to 128×128 RGB"),
        ("Pipeline", "Inference → occlusion XAI → LIME overlay"),
        ("Classes", "Glioma · Meningioma · Pituitary · No Tumor"),
    ]
    ty = y - 32
    for k, v in summary_lines:
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(margin + 10, ty, f"{k}:")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(margin + 72, ty, v)
        ty -= 14

    pdf.setFillColor(colors.white)
    pdf.roundRect(right_x, y - card_h, page_width - margin - right_x, card_h, 8, stroke=0, fill=1)
    pdf.setStrokeColor(colors.HexColor("#cfd8e3"))
    pdf.setLineWidth(0.8)
    pdf.roundRect(
        right_x, y - card_h, page_width - margin - right_x, card_h, 8, stroke=1, fill=0
    )

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(right_x + 10, y - 17, "Diagnosis")
    pdf.setFont("Helvetica-Bold", 8)
    pdf.setFillColor(text_muted)
    pdf.drawString(right_x + 10, y - 32, "PREDICTION")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(success_green if not is_no_tumor else navy_light)
    pred_line = headline
    pdf.drawString(right_x + 10, y - 48, pred_line)
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(right_x + 10, y - 64, f"Raw label: {payload.get('result', '')}")
    pdf.drawString(right_x + 10, y - 77, f"Top-class confidence: {payload.get('confidence', '')}")

    donut_cx = page_width - margin - 48
    donut_cy = y - card_h / 2 + 2
    draw_donut(donut_cx, donut_cy, 60, conf_pct)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(colors.black)
    pdf.drawCentredString(donut_cx, donut_cy - 40, f"{conf_pct:.2f}%")
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(text_muted)
    pdf.drawCentredString(donut_cx, donut_cy - 50, "CONFIDENCE")

    y = y - card_h - 14
    y = draw_probability_section(y)
    y = draw_image_row(y)

    # Findings / interpretation — 2×2 grid: draw ALL panels first, then ALL text (text stays on top).
    # Never place row2 using min(y1,y2) then "clamp" upward for the footer — that paints row2 boxes
    # over row1 text. Row2 top is fixed below row1 bottom with a clear gap.
    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Findings & analysis")
    y -= 18
    y_findings_top = y
    quad_gutter = 16
    pad_q = 14
    text_w_q = (page_width - 2 * margin - quad_gutter) / 2 - 2 * pad_q
    box_w = (page_width - 2 * margin - quad_gutter) / 2
    h_box = 118
    row_gap = 22
    notes_h = 88
    right_xq = margin + box_w + quad_gutter
    # Fit footer entirely below the quad; shrink panels if the separator would sit inside row 2
    y_row2_top = y_findings_top - h_box - row_gap
    quad_bottom_y = y_row2_top - notes_h
    foot_top = quad_bottom_y - footer_gap_below_quad
    while foot_top < footer_min_top_y and (h_box > 96 or notes_h > 72):
        if h_box > 96:
            h_box -= 6
        if notes_h > 72:
            notes_h -= 4
        y_row2_top = y_findings_top - h_box - row_gap
        quad_bottom_y = y_row2_top - notes_h
        foot_top = quad_bottom_y - footer_gap_below_quad
    foot_top = quad_bottom_y - footer_gap_below_quad

    pdf.setFillColor(panel_blue)
    pdf.roundRect(margin, y_findings_top - h_box, box_w, h_box, 8, stroke=0, fill=1)
    pdf.setFillColor(panel_green)
    pdf.roundRect(right_xq, y_findings_top - h_box, box_w, h_box, 8, stroke=0, fill=1)
    # Row 2 backgrounds (always below row 1 — no vertical overlap)
    pdf.setFillColor(colors.HexColor("#f7f9fc"))
    pdf.roundRect(margin, y_row2_top - notes_h, box_w, notes_h, 8, stroke=0, fill=1)
    pdf.setFillColor(panel_blue)
    pdf.roundRect(right_xq, y_row2_top - notes_h, box_w, notes_h, 8, stroke=0, fill=1)

    # Text on top of backgrounds (order avoids panels hiding each other’s copy)
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin + pad_q, y_findings_top - 20, "Model interpretation")
    interp = _interpretation_blurb(top_key)
    draw_wrapped_paragraph(
        interp,
        margin + pad_q,
        y_findings_top - 36,
        text_w_q,
        12,
        "Helvetica",
        10,
        max_lines=8,
    )

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(right_xq + pad_q, y_findings_top - 20, "LIME explainability summary")
    lime_txt = (
        "LIME perturbs superpixels and fits a local linear model to approximate the "
        "classifier near this image. Boundaries mark regions that pushed the score toward "
        f"the predicted class ({CLASS_DISPLAY.get(top_key, top_key)}). Use alongside clinical context."
    )
    draw_wrapped_paragraph(
        lime_txt,
        right_xq + pad_q,
        y_findings_top - 36,
        text_w_q,
        12,
        "Helvetica",
        10,
        max_lines=8,
    )

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(margin + pad_q, y_row2_top - 18, "Report notes")
    pdf.setFont("Helvetica", 9)
    note_items = [
        "· For research, education, or demonstration only.",
        "· Not FDA-cleared; not a substitute for a radiologist.",
        "· Single-slice uploads omit 3D context and sequence.",
        "· Verify against the full MRI study and patient history.",
    ]
    ny = y_row2_top - 34
    for line in note_items:
        pdf.drawString(margin + pad_q, ny, line)
        ny -= 12

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(right_xq + pad_q, y_row2_top - 18, "Technical details")
    pdf.setFont("Helvetica", 9)
    tech = [
        ("Architecture", "VGG16 backbone + dense classifier"),
        ("Input resolution", "128 × 128 (preprocessed)"),
        ("Explainability", "Occlusion sensitivity map; LIME"),
        ("Report ID", report_id),
    ]
    ty = y_row2_top - 34
    for k, v in tech:
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(right_xq + pad_q, ty, f"{k}:")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(right_xq + pad_q + 88, ty, v)
        ty -= 12

    # Footer: rule sits below the 2×2 grid (foot_top computed above — not a fixed constant)
    mid_x = page_width / 2
    v_bottom = margin
    col_pad = 10
    tx = mid_x + col_pad
    right_text_w = page_width - margin - tx

    pdf.setStrokeColor(navy)
    pdf.setLineWidth(1.25)
    pdf.line(margin, foot_top, page_width - margin, foot_top)

    pdf.setStrokeColor(navy_light)
    pdf.setLineWidth(0.85)
    v_top = max(v_bottom + 24, foot_top - 2)
    pdf.line(mid_x, v_top, mid_x, v_bottom)

    # Left column: authors (bold navy) + supervisor / dept / uni (grey-blue)
    ly = foot_top - 14
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(margin, ly, REPORT_AUTHOR_LINE_1)
    ly -= 12
    pdf.drawString(margin, ly, REPORT_AUTHOR_LINE_2)
    ly -= 10
    pdf.setFillColor(footer_secondary)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(margin, ly, REPORT_SUPERVISOR_LINE)
    ly -= 11
    pdf.drawString(margin, ly, REPORT_DEPARTMENT_LINE)
    ly -= 11
    pdf.drawString(margin, ly, REPORT_INSTITUTION_LINE)

    # Right column: bold heading + body
    ry = foot_top - 14
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(tx, ry, REPORT_DISCLAIMER_HEADING)
    pdf.setFillColor(footer_secondary)
    draw_wrapped_paragraph(
        REPORT_DISCLAIMER_BODY,
        tx,
        ry - 13,
        right_text_w,
        11,
        "Helvetica",
        9,
        max_lines=6,
    )

    pdf.save()
    buffer.seek(0)
    fname = f"brain_mri_diagnostic_report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    return FileResponse(
        buffer,
        as_attachment=True,
        filename=fname,
        content_type="application/pdf",
    )


def predict_tumor(image_path):
    image_size = 128
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    pred_vec = predictions[0]
    predicted_class_index = int(np.argmax(pred_vec))
    confidence_score = float(np.max(pred_vec))
    probs = {class_labels[i]: float(pred_vec[i]) for i in range(len(class_labels))}
    top_key = class_labels[predicted_class_index]

    if top_key == "notumor":
        return "No Tumor", confidence_score, probs, top_key
    return f"Tumor: {top_key}", confidence_score, probs, top_key


def generate_xai_overlay(image_path):
    # Model-agnostic XAI: occlusion sensitivity map.
    _log_pipeline("Generating XAI overlay")
    image_size = 128
    original_img = load_img(image_path, target_size=(image_size, image_size))
    original_arr = img_to_array(original_img) / 255.0
    input_tensor = np.expand_dims(original_arr, axis=0)

    base_pred = model.predict(input_tensor, verbose=0)[0]
    target_class = int(np.argmax(base_pred))
    base_conf = float(base_pred[target_class])

    patch = 64
    stride = 32
    heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    counts = np.zeros((image_size, image_size), dtype=np.float32)

    for y in range(0, image_size - patch + 1, stride):
        for x in range(0, image_size - patch + 1, stride):
            occluded = np.array(original_arr, copy=True)
            occluded[y:y + patch, x:x + patch, :] = 0.0
            pred = model.predict(np.expand_dims(occluded, axis=0), verbose=0)[0]
            drop = max(0.0, base_conf - float(pred[target_class]))
            heatmap[y:y + patch, x:x + patch] += drop
            counts[y:y + patch, x:x + patch] += 1.0

    counts[counts == 0] = 1.0
    heatmap = heatmap / counts
    max_value = float(np.max(heatmap))
    if max_value > 0:
        heatmap = heatmap / max_value

    # Create RGB heatmap without external dependencies.
    heat_r = (heatmap * 255).astype(np.uint8)
    heat_g = np.zeros_like(heat_r, dtype=np.uint8)
    heat_b = ((1.0 - heatmap) * 255).astype(np.uint8)
    heat_rgb = np.stack([heat_r, heat_g, heat_b], axis=-1).astype(np.float32)

    original = (original_arr * 255.0).astype(np.float32)
    overlay = (0.6 * original) + (0.4 * heat_rgb)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    base_name = os.path.basename(image_path)
    xai_name = f"xai_{base_name}.png"
    storage = FileSystemStorage(location=settings.MEDIA_ROOT)
    from PIL import Image
    from io import BytesIO

    image = Image.fromarray(overlay)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    saved_name = storage.save(xai_name, ContentFile(buffer.getvalue()))
    return f"{settings.MEDIA_URL}{saved_name}"


def generate_lime_overlay(image_path):
    if lime_image is None or mark_boundaries is None:
        # Keep the app functional even if optional explainability deps are missing.
        return None

    _log_pipeline("Generating LIME overlay")
    image_size = 128
    original_img = load_img(image_path, target_size=(image_size, image_size))
    image_uint8 = img_to_array(original_img).astype(np.uint8)

    def classifier_fn(images):
        batch = np.array(images).astype(np.float32) / 255.0
        return model.predict(batch, verbose=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_uint8,
        classifier_fn,
        top_labels=1,
        hide_color=0,
        num_samples=300,
    )

    top_label = explanation.top_labels[0]
    explained_image, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=8,
        hide_rest=False,
    )

    boundary = mark_boundaries(explained_image / 255.0, mask)
    boundary_uint8 = np.clip(boundary * 255.0, 0, 255).astype(np.uint8)

    base_name = os.path.basename(image_path)
    lime_name = f"lime_{base_name}.png"
    storage = FileSystemStorage(location=settings.MEDIA_ROOT)

    from PIL import Image
    from io import BytesIO

    image = Image.fromarray(boundary_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    saved_name = storage.save(lime_name, ContentFile(buffer.getvalue()))
    return f"{settings.MEDIA_URL}{saved_name}"


def index(request):
    if request.method == "POST" and request.FILES.get("file"):
        _log_pipeline("Upload received - starting analysis (0%)")
        uploaded_file = request.FILES["file"]
        storage = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = storage.save(uploaded_file.name, uploaded_file)
        file_location = storage.path(filename)

        _log_pipeline("File saved - running prediction (25%)")
        result, confidence, probs, top_key = predict_tumor(file_location)
        xai_path = None
        lime_path = None
        try:
            _log_pipeline("Prediction complete - creating XAI overlay (50%)")
            xai_path = generate_xai_overlay(file_location)
        except Exception:
            xai_path = None
        try:
            _log_pipeline("XAI complete - creating LIME overlay (75%)")
            lime_path = generate_lime_overlay(file_location)
        except Exception:
            lime_path = None
        _log_pipeline("Analysis complete - preparing results (100%)")
        request.session["mri_report"] = {
            "result": result,
            "confidence": f"{confidence * 100:.2f}%",
            "probs": probs,
            "top_class": top_key,
            "upload_rel": filename,
            "xai_rel": _relative_under_media(xai_path),
            "lime_rel": _relative_under_media(lime_path),
        }
        return render(
            request,
            "index.html",
            {
                "result": result,
                "confidence": f"{confidence * 100:.2f}%",
                "file_path": f"{settings.MEDIA_URL}{filename}",
                "xai_path": xai_path,
                "lime_path": lime_path,
            },
        )

    return render(request, "index.html", {"result": None})
