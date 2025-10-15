import os
import io
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st

# Lazy imports where possible to keep startup fast

APP_DIR = Path(__file__).parent.resolve()
DEFAULT_WEIGHTS = str(APP_DIR / "best (1).pt")
WORK_DIR = APP_DIR / "workspace"
PDF_DIR = WORK_DIR / "pdfs"
IMG_DIR = WORK_DIR / "images"
OUT_DIR = WORK_DIR / "outputs"


def ensure_dirs() -> None:
    for d in (WORK_DIR, PDF_DIR, IMG_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    ensure_dirs()
    dest = WORK_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.read())
    return dest


def convert_dwf_to_pdf(dwf_path: Path, out_dir: Path) -> List[Path]:
    """Convert DWF -> PDF using ConvertAPI if configured. Returns list of PDFs."""
    secret = os.getenv("CONVERTAPI_SECRET")
    if not secret:
        raise RuntimeError("ConvertAPI secret not set. Set CONVERTAPI_SECRET env var to enable DWF conversion.")

    import convertapi  # type: ignore

    convertapi.api_credentials = secret
    result = convertapi.convert(
        'pdf',
        {'File': str(dwf_path)},
        from_format='dwf'
    )
    saved = result.save_files(str(out_dir))
    return [Path(p) for p in saved]


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    """Render each PDF page to PNG using PyMuPDF."""
    import fitz  # PyMuPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[Path] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = out_dir / f"{pdf_path.stem}_p{page_index+1:03}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
    return image_paths


def run_yolo_inference(image_paths: List[Path], weights_path: Path, conf: float = 0.10, iou: float = 0.2) -> Tuple[Dict[str, int], List[Path]]:
    """Run YOLO on images, return counts and paths to annotated images."""
    from collections import Counter
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    out_proj = str(OUT_DIR / "runs")
    name = "streamlit"
    counts = Counter()
    annotated: List[Path] = []

    for img in image_paths:
        results = model(
            str(img),
            conf=conf,
            iou=iou,
            save=True,
            save_txt=False,
            project=out_proj,
            name=name,
            exist_ok=True,
        )
        # Collect detections from first result object
        if not results:
            continue
        r0 = results[0]
        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                cls_id = int(b.cls[0])
                cls_name = r0.names.get(cls_id, str(cls_id)) if hasattr(r0, 'names') else str(cls_id)
                counts[cls_name] += 1

        # Locate saved annotated image (Ultralytics may change extension)
        run_dir = Path(out_proj) / name
        stem = Path(img).stem
        for ext in (".jpg", ".png", ".jpeg"):
            ann = run_dir / f"{stem}{ext}"
            if ann.exists():
                annotated.append(ann)
                break

    return dict(counts), annotated


def reset_workspace() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)


def main() -> None:
    st.set_page_config(page_title="Electrical Symbols Inference", layout="wide")
    st.title("Electrical Symbols Inference (DWF/PDF → Images → YOLO)")

    with st.sidebar:
        st.header("Settings")
        weights = st.text_input("Model weights path", value=DEFAULT_WEIGHTS)
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.10, 0.01)
        iou = st.slider("IoU threshold", 0.1, 0.95, 0.20, 0.01)
        dpi = st.slider("Render DPI", 72, 400, 200, 4)
        if st.button("Reset workspace"):
            reset_workspace()
            st.success("Workspace cleared")

    st.write("Upload a `.dwf`, `.pdf`, or image (`.jpg/.png/.tif/.bmp/.webp`). DWFs are converted to PDF via ConvertAPI.")

    up = st.file_uploader(
        "Upload file",
        type=["dwf", "pdf", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=False,
    )
    if not up:
        st.info("Awaiting upload…")
        return

    ensure_dirs()
    uploaded_path = save_uploaded_file(up)
    st.write(f"Saved: `{uploaded_path.name}`")

    try:
        all_images: List[Path] = []
        suffix = uploaded_path.suffix.lower()

        if suffix == ".dwf":
            with st.spinner("Converting DWF → PDF via ConvertAPI…"):
                pdf_paths = convert_dwf_to_pdf(uploaded_path, PDF_DIR)
            with st.spinner("Rendering PDF pages to images…"):
                for p in pdf_paths:
                    imgs = render_pdf_to_images(p, IMG_DIR, dpi=dpi)
                    all_images.extend(imgs)
        elif suffix == ".pdf":
            with st.spinner("Rendering PDF pages to images…"):
                imgs = render_pdf_to_images(uploaded_path, IMG_DIR, dpi=dpi)
                all_images.extend(imgs)
        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            # Use image directly
            all_images = [uploaded_path]
        else:
            st.error("Unsupported file type")
            return

        if not all_images:
            st.error("No images were rendered from the PDF.")
            return

        st.success(f"Prepared {len(all_images)} page image(s)")

        weights_path = Path(weights)
        if not weights_path.exists():
            st.warning(f"Weights not found at {weights_path}. Using default if available.")

        with st.spinner("Running YOLO inference…"):
            counts, annotated = run_yolo_inference(all_images, weights_path, conf=conf, iou=iou)

        st.subheader("Detection Counts")
        if counts:
            st.json(counts)
        else:
            st.write("No detections.")

        st.subheader("Annotated Images")
        if not annotated:
            st.warning("No annotated images found. Check thresholds or model weights.")
        else:
            from PIL import Image  # lazy import
            cols = st.columns(2)
            for idx, img_path in enumerate(annotated):
                try:
                    img_obj = Image.open(img_path)
                    cols[idx % 2].image(img_obj, caption=img_path.name, use_container_width=True)
                except Exception as _:
                    with open(img_path, "rb") as f:
                        cols[idx % 2].image(f.read(), caption=img_path.name, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()


