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
DEFAULT_WEIGHTS = str(APP_DIR / "best.onnx")
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
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced CSS for better styling and animations
    st.markdown("""
    <style>
    /* Main Header with Animation */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1f77b4, #ff6b6b, #4ecdc4);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4f8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin: 10px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    /* Enhanced Circular Progress with Animation */
    .circular-progress {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: conic-gradient(#1f77b4 0deg, #e0e0e0 0deg);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        margin: 20px auto;
        animation: pulse 2s ease-in-out infinite;
        box-shadow: 0 8px 25px rgba(31, 119, 180, 0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .circular-progress::before {
        content: '';
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: white;
        position: absolute;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .circular-progress-text {
        position: relative;
        z-index: 1;
        font-weight: bold;
        color: #1f77b4;
        font-size: 12px;
        text-align: center;
        line-height: 1.2;
    }
    
    /* Enhanced Step Animation */
    .step-container {
        display: flex;
        align-items: center;
        padding: 15px 20px;
        margin: 8px 0;
        border-radius: 12px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .step-container.active {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #1f77b4;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    
    .step-container.completed {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        transform: translateX(0);
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(5px); opacity: 1; }
    }
    
    .step-icon {
        font-size: 28px;
        margin-right: 20px;
        transition: transform 0.3s ease;
    }
    
    .step-container.active .step-icon {
        animation: bounce 1s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-5px); }
        60% { transform: translateY(-3px); }
    }
    
    .step-text {
        flex: 1;
        font-weight: 600;
        font-size: 16px;
        color: #2c3e50;
    }
    
    /* Enhanced Color Legend */
    .color-legend-item {
        text-align: center;
        padding: 20px 15px;
        border: 3px solid;
        border-radius: 15px;
        margin: 10px 5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        cursor: pointer;
    }
    
    .color-legend-item:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    /* Image Comparison Styling */
    .image-comparison {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .image-comparison:hover {
        transform: scale(1.02);
    }
    
    /* Results Panel Styling */
    .results-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #1f77b4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Sidebar */
    .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Progress Pipeline */
    .progress-pipeline {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    /* Remove extra spacing */
    .stApp > div:first-child {
        padding-top: 0;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add logo and company header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.webp", width=200)
        st.markdown('<h1 class="main-header">🔌 Electrical Symbols Detection System</h1>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; color: #2c3e50; font-weight: bold; margin-top: -10px;">DIGITAL PROCESSING SYSTEMS KUWAIT</h2>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Model Configuration")
        weights = st.text_input("Model weights path", value=DEFAULT_WEIGHTS, help="Path to your YOLO model weights")
        
        col1, col2 = st.columns(2)
        with col1:
            conf = st.slider("Confidence", 0.05, 0.95, 0.10, 0.01, 
                            help="Detection confidence threshold")
        with col2:
            iou = st.slider("IoU", 0.1, 0.95, 0.20, 0.01,
                           help="Intersection over Union threshold")
        
        dpi = st.slider("Render DPI", 72, 400, 300, 4,
                       help="Image quality (higher = better but slower)")
        
        st.markdown("### 🗂️ Workspace Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Reset", help="Clear all temporary files", use_container_width=True):
                reset_workspace()
            st.success("Workspace cleared")
        with col2:
            if st.button("📊 Stats", help="View workspace statistics", use_container_width=True):
                st.rerun()
        
        st.markdown("### 📈 Quick Stats")
        if WORK_DIR.exists():
            file_count = len(list(WORK_DIR.rglob("*")))
            st.metric("Files in workspace", file_count, delta=None)
        else:
            st.metric("Files in workspace", 0, delta=None)
        
        # ConvertAPI status with better styling
        st.markdown("### 🔧 ConvertAPI Status")
        convertapi_secret = os.getenv("CONVERTAPI_SECRET")
        if convertapi_secret and convertapi_secret != "your_convertapi_secret_here":
            st.success("✅ ConvertAPI configured")
        else:
            st.warning("⚠️ ConvertAPI not configured")
            with st.expander("Setup Instructions"):
                st.markdown("""
                **DWF files require ConvertAPI:**
                1. Get secret from: https://www.convertapi.com/a
                2. Update `.env` file
                3. Restart app
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Create enhanced progress indicators
    progress_container = st.container()
    status_text = st.empty()
    
    # Step indicators with enhanced animations
    steps = [
        ("📁", "File Upload & Validation"),
        ("🔄", "DWF to PDF Conversion"), 
        ("🖼️", "PDF to Images Rendering"),
        ("🤖", "YOLO Model Loading"),
        ("🔍", "Object Detection Inference"),
        ("📊", "Results Processing")
    ]
    
    step_progress = {}
    for i, (icon, step_name) in enumerate(steps):
        step_progress[step_name] = st.empty()
    
    # Circular progress indicator
    def create_circular_progress(percentage, text):
        return f"""
        <div class="circular-progress" style="background: conic-gradient(#1f77b4 {percentage*3.6}deg, #e0e0e0 0deg);">
            <div class="circular-progress-text">
                {text}<br>
                <span style="font-size: 18px; font-weight: bold;">{percentage}%</span>
            </div>
        </div>
        """
    
    # Add enhanced animated progress container
    with progress_container:
        st.markdown('<div class="progress-pipeline">', unsafe_allow_html=True)
        st.markdown("### 🚀 Processing Pipeline")
        st.markdown("---")
        
        # Display steps in a visual format with better positioning
        for i, (icon, step_name) in enumerate(steps):
            step_progress[step_name].markdown(f"""
            <div class="step-container" id="step-{i}">
                <div class="step-icon">{icon}</div>
                <div class="step-text">{step_name}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    try:
        all_images: List[Path] = []
        suffix = uploaded_path.suffix.lower()

        # Step 1: File processing with enhanced animation
        step_progress["File Upload & Validation"].markdown(f"""
        <div class="step-container active">
            <div class="step-icon">📁</div>
            <div class="step-text">File Upload & Validation - In Progress...</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show circular progress
        progress_display = st.empty()
        progress_display.markdown(create_circular_progress(10, "Validating"), unsafe_allow_html=True)
        
        import time
        time.sleep(0.2)  # Faster animation
        
        step_progress["File Upload & Validation"].markdown(f"""
        <div class="step-container completed">
            <div class="step-icon">✅</div>
            <div class="step-text">File Upload & Validation - Complete</div>
        </div>
        """, unsafe_allow_html=True)
        
        status_text.text("🔄 Processing file...")
        progress_display.markdown(create_circular_progress(20, "Processing"), unsafe_allow_html=True)

        if suffix == ".dwf":
            # Step 2: DWF to PDF conversion with enhanced animation
            step_progress["DWF to PDF Conversion"].markdown(f"""
            <div class="step-container active">
                <div class="step-icon">🔄</div>
                <div class="step-text">DWF to PDF Conversion - In Progress...</div>
            </div>
            """, unsafe_allow_html=True)
            
            status_text.text("🔄 Converting DWF → PDF via ConvertAPI...")
            progress_display.markdown(create_circular_progress(30, "Converting"), unsafe_allow_html=True)
            
            with st.spinner("🔄 Converting DWF to PDF..."):
                import time
                time.sleep(0.1)  # Faster animation
                pdf_paths = convert_dwf_to_pdf(uploaded_path, PDF_DIR)
                time.sleep(0.1)  # Faster completion animation
            
            step_progress["DWF to PDF Conversion"].markdown(f"""
            <div class="step-container completed">
                <div class="step-icon">✅</div>
                <div class="step-text">DWF to PDF Conversion - Complete</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Step 3: PDF to images with enhanced animation
            step_progress["PDF to Images Rendering"].markdown(f"""
            <div class="step-container active">
                <div class="step-icon">🖼️</div>
                <div class="step-text">PDF to Images Rendering - In Progress...</div>
            </div>
            """, unsafe_allow_html=True)
            
            status_text.text("🔄 Rendering PDF pages to images...")
            progress_display.markdown(create_circular_progress(50, "Rendering"), unsafe_allow_html=True)
            
            with st.spinner("🖼️ Converting PDF pages to images..."):
                for i, p in enumerate(pdf_paths):
                    imgs = render_pdf_to_images(p, IMG_DIR, dpi=dpi)
                    all_images.extend(imgs)
                    progress = 50 + (i + 1) * 20 // len(pdf_paths)
                    progress_display.markdown(create_circular_progress(progress, f"Page {i+1}"), unsafe_allow_html=True)
                    time.sleep(0.05)  # Faster animation between pages
            
            step_progress["PDF to Images Rendering"].markdown(f"""
            <div class="step-container completed">
                <div class="step-icon">✅</div>
                <div class="step-text">PDF to Images Rendering - Complete</div>
            </div>
            """, unsafe_allow_html=True)
            
        elif suffix == ".pdf":
            # Step 3: PDF to images with animation
            step_progress["PDF to Images Rendering"].markdown(f"""
            <div class="step-container active">
                <div class="step-icon">🖼️</div>
                <div class="step-text">PDF to Images Rendering - In Progress...</div>
            </div>
            """, unsafe_allow_html=True)
            
            status_text.text("🔄 Rendering PDF pages to images...")
            progress_display.markdown(create_circular_progress(40, "Rendering"), unsafe_allow_html=True)
            
            with st.spinner("🖼️ Converting PDF pages to images..."):
                imgs = render_pdf_to_images(uploaded_path, IMG_DIR, dpi=dpi)
                all_images.extend(imgs)
                time.sleep(0.1)  # Faster completion animation
            
            step_progress["PDF to Images Rendering"].markdown(f"""
            <div class="step-container completed">
                <div class="step-icon">✅</div>
                <div class="step-text">PDF to Images Rendering - Complete</div>
            </div>
            """, unsafe_allow_html=True)
            progress_display.markdown(create_circular_progress(60, "Complete"), unsafe_allow_html=True)
            
        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            step_progress["PDF to Images Rendering"].markdown(f"""
            <div class="step-container active">
                <div class="step-icon">🖼️</div>
                <div class="step-text">Image Processing - In Progress...</div>
            </div>
            """, unsafe_allow_html=True)
            
            status_text.text("🔄 Processing image file...")
            progress_display.markdown(create_circular_progress(40, "Processing"), unsafe_allow_html=True)
            
            with st.spinner("🖼️ Processing image file..."):
                time.sleep(0.1)  # Faster animation
            all_images = [uploaded_path]
            time.sleep(0.1)  # Faster completion animation
            
            step_progress["PDF to Images Rendering"].markdown(f"""
            <div class="step-container completed">
                <div class="step-icon">✅</div>
                <div class="step-text">Image Processing - Complete</div>
            </div>
            """, unsafe_allow_html=True)
            progress_display.markdown(create_circular_progress(60, "Complete"), unsafe_allow_html=True)
        else:
            st.error("❌ Unsupported file type")
            return

        if not all_images:
            st.error("❌ No images were rendered from the file.")
            return

        status_text.text(f"✅ Prepared {len(all_images)} page image(s)")
        progress_display.markdown(create_circular_progress(70, "Prepared"), unsafe_allow_html=True)

        # Step 4: Model loading
        weights_path = Path(weights)
        if not weights_path.exists():
            st.warning(f"⚠️ Weights not found at {weights_path}. Using default if available.")

        step_progress["YOLO Model Loading"].markdown(f"""
        <div class="step-container active">
            <div class="step-icon">🤖</div>
            <div class="step-text">YOLO Model Loading - In Progress...</div>
        </div>
        """, unsafe_allow_html=True)
        
        status_text.text("🔄 Loading YOLO model...")
        progress_display.markdown(create_circular_progress(70, "Loading Model"), unsafe_allow_html=True)
        
        with st.spinner("🤖 Loading YOLO model..."):
            import time
            time.sleep(0.2)  # Faster loading animation
            time.sleep(0.2)  # Faster model initialization
        
        step_progress["YOLO Model Loading"].markdown(f"""
        <div class="step-container completed">
            <div class="step-icon">✅</div>
            <div class="step-text">YOLO Model Loading - Complete</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Step 5: Inference with enhanced animation
        step_progress["Object Detection Inference"].markdown(f"""
        <div class="step-container active">
            <div class="step-icon">🔍</div>
            <div class="step-text">Object Detection Inference - In Progress...</div>
        </div>
        """, unsafe_allow_html=True)
        
        status_text.text("🔄 Running inference on all pages...")
        progress_display.markdown(create_circular_progress(80, "Detecting"), unsafe_allow_html=True)
        
        with st.spinner("🔍 Performing object detection..."):
            counts, annotated, page_detections = run_yolo_inference(all_images, weights_path, conf=conf, iou=iou)
            time.sleep(0.1)  # Faster completion animation
        
        step_progress["Object Detection Inference"].markdown(f"""
        <div class="step-container completed">
            <div class="step-icon">✅</div>
            <div class="step-text">Object Detection Inference - Complete</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Step 6: Results processing with enhanced animation
        step_progress["Results Processing"].markdown(f"""
        <div class="step-container active">
            <div class="step-icon">📊</div>
            <div class="step-text">Results Processing - In Progress...</div>
        </div>
        """, unsafe_allow_html=True)
        
        status_text.text("🔄 Processing results...")
        progress_display.markdown(create_circular_progress(95, "Processing"), unsafe_allow_html=True)
        
        with st.spinner("📊 Processing detection results..."):
            import time
            time.sleep(0.1)  # Faster processing animation
            time.sleep(0.1)  # Faster results compilation
            time.sleep(0.1)  # Faster final processing
        
        step_progress["Results Processing"].markdown(f"""
        <div class="step-container completed">
            <div class="step-icon">✅</div>
            <div class="step-text">Results Processing - Complete</div>
        </div>
        """, unsafe_allow_html=True)
        
        status_text.text("✅ Analysis complete!")
        progress_display.markdown(create_circular_progress(100, "Complete!"), unsafe_allow_html=True)
        
        # Clear progress indicators after a shorter delay
        import time
        time.sleep(1.5)  # Faster cleanup
        progress_display.empty()
        status_text.empty()
        
        # Clear step indicators
        for step_name in ["File Upload & Validation", "DWF to PDF Conversion", "PDF to Images Rendering", "YOLO Model Loading", "Object Detection Inference", "Results Processing"]:
            step_progress[step_name].empty()

        # Main Results Display - Side by Side Layout
        st.markdown("## 🔍 Inference Results")
        
        if not counts:
            st.warning("No detections found. Try adjusting the confidence threshold.")
            return
        
        # Page selector at the top
        if annotated:
            selected_page = st.selectbox(
                "📄 Select Page to View",
                range(1, len(annotated) + 1),
                format_func=lambda x: f"Page {x}",
                key="page_selector"
            )
            
            # Show loading animation when changing pages
            with st.spinner("🔄 Loading page..."):
                import time
                time.sleep(0.1)  # Faster loading animation
            
            # Create enhanced side-by-side layout for original and inference images
            col_original, col_inference = st.columns(2)
            
            with col_original:
                st.markdown("### 📷 Original Image")
                try:
                    # Load original image with enhanced styling
                    original_img_path = all_images[selected_page - 1]
                    from PIL import Image
                    original_img = Image.open(original_img_path)
                    st.markdown('<div class="image-comparison">', unsafe_allow_html=True)
                    st.image(original_img, caption=f"Original - Page {selected_page}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading original image: {e}")
            
            with col_inference:
                st.markdown("### 🎯 Inference Results")
                try:
                    # Load inference image with enhanced styling
                    inference_img = Image.open(annotated[selected_page - 1])
                    st.markdown('<div class="image-comparison">', unsafe_allow_html=True)
                    st.image(inference_img, caption=f"Inference - Page {selected_page}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading inference image: {e}")
            
            # Color Legend - Centered below images
            st.markdown("---")
            st.subheader("🎨 Color Legend for Bounding Boxes")
            
            # Create color legend with better visual representation
            color_legend = {
                'Cove Light': ('🔴', '#FF0000'),
                'Door': ('🟢', '#00FF00'), 
                'Emergency Light Fitting': ('🔵', '#0000FF'),
                'Fluorescent Light': ('🟡', '#FFFF00'),
                'exit': ('🟣', '#FF00FF'),
                'Downlight': ('🔵', '#00FFFF'),
                'Socket Outlet': ('🟣', '#800080')
            }
            
            # Display color legend in a grid
            cols_legend = st.columns(len(color_legend))
            for i, (class_name, (emoji, color)) in enumerate(color_legend.items()):
                with cols_legend[i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 8px; margin: 5px;">
                        <div style="font-size: 24px;">{emoji}</div>
                        <div style="font-weight: bold; color: {color};">{class_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced Detection Results Panel
            st.markdown("---")
            st.markdown('<div class="results-panel">', unsafe_allow_html=True)
            st.subheader("📊 Detection Analysis")
            
            if page_detections and selected_page:
                page_data = page_detections[selected_page - 1]
                
                # Create metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Detections", len(page_data['detections']))
                with col2:
                    st.metric("Classes Found", len(page_data['class_counts']))
                with col3:
                    if page_data['detections']:
                        avg_size = sum((d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) for d in page_data['detections']) / len(page_data['detections'])
                        st.metric("Avg Size", f"{avg_size:.0f}px²")
                with col4:
                    if page_data['detections']:
                        st.metric("Density", f"{len(page_data['detections'])/100:.1f}/100px²")
                
                if page_data['detections']:
                    # Class breakdown with color coding
                    st.markdown("### 🎯 Class Breakdown")
                    for class_name, count in page_data['class_counts'].items():
                        color = color_legend.get(class_name, ('⚪', '#FFFFFF'))[1]
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; padding: 8px; margin: 4px 0; border-left: 4px solid {color}; background-color: rgba(0,0,0,0.05);">
                            <span style="font-weight: bold; color: {color};">{class_name}</span>
                            <span style="margin-left: auto; font-size: 18px; font-weight: bold;">{count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed detections table
                    st.markdown("### 📋 Individual Detections")
                    detections_data = []
                    for i, detection in enumerate(page_data['detections']):
                        class_name = detection['class_name']
                        color = color_legend.get(class_name, ('⚪', '#FFFFFF'))[1]
                        detections_data.append({
                            'ID': i + 1,
                            'Class': f'<span style="color: {color}; font-weight: bold;">{class_name}</span>',
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
            else:
                st.warning("No annotated images found.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
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


