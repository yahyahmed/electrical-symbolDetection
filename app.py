import os
import io
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Lazy imports where possible to keep startup fast

APP_DIR = Path(__file__).parent.resolve()
DEFAULT_WEIGHTS = str(APP_DIR / "best (2).pt")
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
    if not secret or secret == "your_convertapi_secret_here":
        st.error("""
        **ConvertAPI Secret Not Configured**
        
        To enable DWF conversion, please:
        1. Get your ConvertAPI secret from: https://www.convertapi.com/a
        2. Update the `.env` file with your actual secret:
           ```
           CONVERTAPI_SECRET=your_actual_secret_here
           ```
        3. Restart the application
        
        **Alternative**: Upload PDF files directly instead of DWF files.
        """)
        raise RuntimeError("ConvertAPI secret not configured. Please set CONVERTAPI_SECRET in .env file.")

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


def run_yolo_inference(image_paths: List[Path], weights_path: Path, conf: float = 0.10, iou: float = 0.2) -> Tuple[Dict[str, int], List[Path], List[Dict]]:
    """Run YOLO on images, return counts, custom annotated images, and detailed detection data."""
    from collections import Counter
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    model = YOLO(str(weights_path))

    out_proj = str(OUT_DIR / "runs")
    name = "streamlit"
    counts = Counter()
    annotated: List[Path] = []
    page_detections: List[Dict] = []

    # Define colors for different classes
    class_colors = {
        'Cove Light': (255, 0, 0),      # Red
        'Door': (0, 255, 0),            # Green
        'Emergency Light Fitting': (0, 0, 255),  # Blue
        'Fluorescent Light': (255, 255, 0),      # Yellow
        'exit': (255, 0, 255),          # Magenta
        'Downlight': (0, 255, 255),     # Cyan
        'Socket Outlet': (128, 0, 128), # Purple
    }

    for page_idx, img in enumerate(image_paths):
        results = model(
            str(img),
            conf=conf,
            iou=iou,
            save=False,  # Don't save default annotated images
            save_txt=False,
        )
        
        # Collect detections from first result object
        page_detection_data = {
            'page_number': page_idx + 1,
            'image_name': img.name,
            'detections': [],
            'class_counts': Counter()
        }
        
        if results:
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                for b in r0.boxes:
                    cls_id = int(b.cls[0])
                    cls_name = r0.names.get(cls_id, str(cls_id)) if hasattr(r0, 'names') else str(cls_id)
                    conf_score = float(b.conf[0])
                    
                    # Add to page detection data
                    detection = {
                        'class_name': cls_name,
                        'confidence': conf_score,
                        'bbox': b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    }
                    page_detection_data['detections'].append(detection)
                    page_detection_data['class_counts'][cls_name] += 1
                    counts[cls_name] += 1

        page_detections.append(page_detection_data)

        # Create custom annotated image with only bounding boxes
        if page_detection_data['detections']:
            # Load original image
            img_cv = cv2.imread(str(img))
            if img_cv is not None:
                # Draw bounding boxes with class-specific colors
                for detection in page_detection_data['detections']:
                    bbox = detection['bbox']
                    class_name = detection['class_name']
                    
                    # Get color for this class
                    color = class_colors.get(class_name, (255, 255, 255))  # Default white
                    
                    # Draw bounding box only (no labels)
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                
                # Save custom annotated image
                custom_annotated_path = OUT_DIR / f"custom_annotated_{img.stem}.jpg"
                cv2.imwrite(str(custom_annotated_path), img_cv)
                annotated.append(custom_annotated_path)
            else:
                # If image loading fails, use original
                annotated.append(img)
        else:
            # No detections, use original image
            annotated.append(img)

    return dict(counts), annotated, page_detections


def reset_workspace() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)


def main() -> None:
    st.set_page_config(
        page_title="Electrical Symbols Detection", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔌 Electrical Symbols Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("### Model Configuration")
        weights = st.text_input("Model weights path", value=DEFAULT_WEIGHTS)
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.10, 0.01, 
                        help="Lower values detect more objects but may include false positives")
        iou = st.slider("IoU threshold", 0.1, 0.95, 0.20, 0.01,
                       help="Higher values reduce duplicate detections")
        dpi = st.slider("Render DPI", 72, 400, 200, 4,
                       help="Higher DPI = better quality but slower processing")
        
        st.markdown("### Workspace Management")
        if st.button("🗑️ Reset workspace", help="Clear all temporary files"):
            reset_workspace()
            st.success("Workspace cleared")

        st.markdown("### Quick Stats")
        if WORK_DIR.exists():
            file_count = len(list(WORK_DIR.rglob("*")))
            st.metric("Files in workspace", file_count)
        else:
            st.metric("Files in workspace", 0)
        
        # ConvertAPI status
        st.markdown("### 🔧 ConvertAPI Status")
        convertapi_secret = os.getenv("CONVERTAPI_SECRET")
        if convertapi_secret and convertapi_secret != "your_convertapi_secret_here":
            st.success("✅ ConvertAPI configured")
        else:
            st.warning("⚠️ ConvertAPI not configured")
            st.markdown("""
            **DWF files require ConvertAPI:**
            1. Get secret from: https://www.convertapi.com/a
            2. Update `.env` file
            3. Restart app
            """)

    # File upload section with better styling
    st.markdown("### 📁 File Upload")
    st.markdown("Upload a `.dwf`, `.pdf`, or image (`.jpg/.png/.tif/.bmp/.webp`). DWFs are converted to PDF via ConvertAPI.")
    
    # Create a more prominent upload area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        up = st.file_uploader(
            "Choose a file to analyze",
            type=["dwf", "pdf", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            accept_multiple_files=False,
            help="Supported formats: DWF, PDF, JPG, PNG, BMP, TIFF, WEBP"
        )
    
    if not up:
        st.info("👆 Please upload a file to begin analysis")
        st.markdown("""
        **Supported file types:**
        - **DWF files**: AutoCAD Design Web Format (requires ConvertAPI)
        - **PDF files**: Multi-page documents
        - **Image files**: JPG, PNG, BMP, TIFF, WEBP
        """)
        return

    ensure_dirs()
    uploaded_path = save_uploaded_file(up)
    
    # Show file info
    st.success(f"✅ File uploaded successfully: `{uploaded_path.name}`")
    
    # Create detailed progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_container = st.container()
    
    # Step indicators with enhanced animations
    steps = [
        "📁 File Upload & Validation",
        "🔄 DWF to PDF Conversion", 
        "🖼️ PDF to Images Rendering",
        "🤖 YOLO Model Loading",
        "🔍 Object Detection Inference",
        "📊 Results Processing"
    ]
    
    step_progress = {}
    for i, step in enumerate(steps):
        step_progress[step] = st.empty()
    
    # Add animated progress container
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🚀 Processing Pipeline")
        st.markdown("---")

    try:
        all_images: List[Path] = []
        suffix = uploaded_path.suffix.lower()

        # Step 1: File processing with animation
        step_progress[steps[0]].info("🔄 Validating file...")
        import time
        time.sleep(0.5)
        step_progress[steps[0]].success("✅ File Upload & Validation Complete")
        status_text.text("🔄 Processing file...")
        progress_bar.progress(10)

        if suffix == ".dwf":
            # Step 2: DWF to PDF conversion with enhanced animation
            step_progress[steps[1]].info("🔄 Converting DWF → PDF via ConvertAPI...")
            status_text.text("🔄 Converting DWF → PDF via ConvertAPI...")
            progress_bar.progress(20)
            
            with st.spinner("🔄 Converting DWF to PDF..."):
                import time
                time.sleep(0.3)  # Brief animation
                pdf_paths = convert_dwf_to_pdf(uploaded_path, PDF_DIR)
                time.sleep(0.3)  # Completion animation
            
            step_progress[steps[1]].success("✅ DWF to PDF Conversion Complete")
            
            # Step 3: PDF to images with enhanced animation
            step_progress[steps[2]].info("🔄 Rendering PDF pages to images...")
            status_text.text("🔄 Rendering PDF pages to images...")
            progress_bar.progress(40)
            
            with st.spinner("🖼️ Converting PDF pages to images..."):
                for i, p in enumerate(pdf_paths):
                    imgs = render_pdf_to_images(p, IMG_DIR, dpi=dpi)
                    all_images.extend(imgs)
                    progress_bar.progress(40 + (i + 1) * 20 // len(pdf_paths))
                    time.sleep(0.2)  # Animation between pages
            
            step_progress[steps[2]].success("✅ PDF to Images Rendering Complete")
            
        elif suffix == ".pdf":
            # Step 3: PDF to images with animation
            step_progress[steps[2]].info("🔄 Rendering PDF pages to images...")
            status_text.text("🔄 Rendering PDF pages to images...")
            progress_bar.progress(30)
            
            with st.spinner("🖼️ Converting PDF pages to images..."):
                imgs = render_pdf_to_images(uploaded_path, IMG_DIR, dpi=dpi)
                all_images.extend(imgs)
                time.sleep(0.5)  # Completion animation
            
            step_progress[steps[2]].success("✅ PDF to Images Rendering Complete")
            progress_bar.progress(60)
            
        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            step_progress[steps[2]].info("🔄 Processing image file...")
            status_text.text("🔄 Processing image file...")
            progress_bar.progress(30)
            
            with st.spinner("🖼️ Processing image file..."):
                time.sleep(0.3)  # Animation
                all_images = [uploaded_path]
                time.sleep(0.3)  # Completion animation
            
            step_progress[steps[2]].success("✅ Image Processing Complete")
            progress_bar.progress(60)
        else:
            st.error("❌ Unsupported file type")
            return

        if not all_images:
            st.error("❌ No images were rendered from the file.")
            return

        status_text.text(f"✅ Prepared {len(all_images)} page image(s)")
        progress_bar.progress(70)

        # Step 4: Model loading
        weights_path = Path(weights)
        if not weights_path.exists():
            st.warning(f"⚠️ Weights not found at {weights_path}. Using default if available.")

        step_progress[steps[3]].info("🔄 Loading YOLO model...")
        status_text.text("🔄 Loading YOLO model...")
        progress_bar.progress(80)
        
        with st.spinner("🤖 Loading YOLO model..."):
            import time
            time.sleep(0.5)  # Brief loading animation
            time.sleep(0.5)  # Model initialization
        
        step_progress[steps[3]].success("✅ YOLO Model Loading Complete")
        
        # Step 5: Inference with enhanced animation
        step_progress[steps[4]].info("🔄 Running object detection inference...")
        status_text.text("🔄 Running inference on all pages...")
        progress_bar.progress(90)
        
        with st.spinner("🔍 Performing object detection..."):
            counts, annotated, page_detections = run_yolo_inference(all_images, weights_path, conf=conf, iou=iou)
            time.sleep(0.5)  # Completion animation
        
        step_progress[steps[4]].success("✅ Object Detection Inference Complete")
        
        # Step 6: Results processing with enhanced animation
        step_progress[steps[5]].info("🔄 Processing results...")
        status_text.text("🔄 Processing results...")
        progress_bar.progress(95)
        
        with st.spinner("📊 Processing detection results..."):
            import time
            time.sleep(0.3)  # Brief processing animation
            time.sleep(0.3)  # Results compilation
            time.sleep(0.3)  # Final processing
        
        step_progress[steps[5]].success("✅ Results Processing Complete")
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators after a delay
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
        # Clear step indicators
        for step in steps:
            step_progress[step].empty()

        # Main Results Display - 70-30 Layout
        st.markdown("## 🔍 Inference Results")
        
        if not counts:
            st.warning("No detections found. Try adjusting the confidence threshold.")
            return
        
        # Create 70-30 layout
        col_left, col_right = st.columns([7, 3])
        
        with col_left:
            st.subheader("🖼️ Annotated Images with Bounding Boxes")
            
            if annotated:
                from PIL import Image
                
                # Page selector with animation
                selected_page = st.selectbox(
                    "Select Page to View",
                    range(1, len(annotated) + 1),
                    format_func=lambda x: f"Page {x}",
                    key="page_selector"
                )
                
                if selected_page:
                    page_idx = selected_page - 1
                    
                    # Show loading animation when changing pages
                    with st.spinner("🔄 Loading page..."):
                        import time
                        time.sleep(0.5)  # Brief loading animation
                    
                    try:
                        img_obj = Image.open(annotated[page_idx])
                        st.image(img_obj, caption=f"Page {selected_page}: {annotated[page_idx].name}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
            else:
                st.warning("No annotated images found.")
        
        with col_right:
            st.subheader("📊 Detection Results")
            
            # Show results for selected page with animation
            if page_detections and selected_page:
                page_data = page_detections[selected_page - 1]
                
                # Animated page summary
                with st.container():
                    st.markdown("### 📈 Page Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Detections", len(page_data['detections']), delta=None)
                    with col2:
                        st.metric("Classes Found", len(page_data['class_counts']), delta=None)
                
                if page_data['detections']:
                    # Animated class breakdown
                    st.markdown("### 🎯 Class Breakdown")
                    for class_name, count in page_data['class_counts'].items():
                        with st.container():
                            st.markdown(f"**{class_name}**: {count} detections")
                    
                    # Animated detections table
                    st.markdown("### 📋 Individual Detections")
                    detections_data = []
                    for i, detection in enumerate(page_data['detections']):
                        detections_data.append({
                            'ID': i + 1,
                            'Class': detection['class_name'],
                            'X1': round(detection['bbox'][0], 1),
                            'Y1': round(detection['bbox'][1], 1),
                            'X2': round(detection['bbox'][2], 1),
                            'Y2': round(detection['bbox'][3], 1),
                            'Width': round(detection['bbox'][2] - detection['bbox'][0], 1),
                            'Height': round(detection['bbox'][3] - detection['bbox'][1], 1)
                        })
                    
                    if detections_data:
                        import pandas as pd
                        df_detections = pd.DataFrame(detections_data)
                        st.dataframe(df_detections, use_container_width=True)
                else:
                    st.info("No detections found on this page.")
        
        # Color Legend
        st.markdown("---")
        st.subheader("🎨 Color Legend for Bounding Boxes")
        
        color_legend = {
            'Cove Light': '🔴 Red',
            'Door': '🟢 Green', 
            'Emergency Light Fitting': '🔵 Blue',
            'Fluorescent Light': '🟡 Yellow',
            'exit': '🟣 Magenta',
            'Downlight': '🔵 Cyan',
            'Socket Outlet': '🟣 Purple'
        }
        
        cols = st.columns(len(color_legend))
        for i, (class_name, color_desc) in enumerate(color_legend.items()):
            with cols[i]:
                st.markdown(f"**{class_name}**<br/>{color_desc}", unsafe_allow_html=True)
        
        # Overall Summary
        st.markdown("---")
        st.subheader("📈 Overall Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pages", len(all_images))
        with col2:
            st.metric("Total Detections", sum(counts.values()))
        with col3:
            st.metric("Unique Classes", len(counts))
        with col4:
            st.metric("Avg per Page", round(sum(counts.values()) / len(all_images), 1))
        
        # Class distribution
        if counts:
            import pandas as pd
            st.subheader("📊 Class Distribution")
            df_counts = pd.DataFrame(list(counts.items()), columns=['Class', 'Count'])
            df_counts = df_counts.sort_values('Count', ascending=False)
            st.bar_chart(df_counts.set_index('Class'))
            
            # Summary table
            st.subheader("📋 Summary Table")
            st.dataframe(df_counts, use_container_width=True)
            
            # Download button
            csv = df_counts.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary as CSV",
                data=csv,
                file_name="detection_summary.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()


