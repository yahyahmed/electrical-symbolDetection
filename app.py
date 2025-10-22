import os
import io
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

    model = YOLO(str(weights_path))

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
            save=False,
            save_txt=False,
        )
        
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
                    
                    detection = {
                        'class_name': cls_name,
                        'confidence': conf_score,
                        'bbox': b.xyxy[0].tolist()
                    }
                    page_detection_data['detections'].append(detection)
                    page_detection_data['class_counts'][cls_name] += 1
                    counts[cls_name] += 1

        page_detections.append(page_detection_data)

        # Create custom annotated image
        if page_detection_data['detections']:
            img_cv = cv2.imread(str(img))
            if img_cv is not None:
                for detection in page_detection_data['detections']:
                    bbox = detection['bbox']
                    class_name = detection['class_name']
                    color = class_colors.get(class_name, (255, 255, 255))
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                
                custom_annotated_path = OUT_DIR / f"custom_annotated_{img.stem}.jpg"
                cv2.imwrite(str(custom_annotated_path), img_cv)
                annotated.append(custom_annotated_path)
            else:
                annotated.append(img)
        else:
            annotated.append(img)

    return dict(counts), annotated, page_detections


def reset_workspace() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)


def main() -> None:
    st.set_page_config(
        page_title="DPS - Electrical Symbol Detector", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 1
    if 'view' not in st.session_state:
        st.session_state.view = 'landing'
    
    # Route to page
    if st.session_state.view == 'landing':
        render_landing()
    else:
        render_analysis_page()


def render_landing() -> None:
    """Render modern landing page."""
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }
    .stButton > button {
        width: 100%;
        background: #e63946 !important;
        color: white !important;
        padding: 16px 32px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 18px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #c72d3a !important;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Top-left corner branding - Logo and text together
    col_logo, col_text = st.columns([0.08, 0.92])

    with col_logo:
        logo_path = APP_DIR / "logo.webp"
        if logo_path.exists():
            st.image(str(logo_path), width=50, use_container_width=False)
        else:
            st.markdown("""
            <div style="width: 50px; height: 50px; background: #e63946; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px;">
                DPS
            </div>
            """, unsafe_allow_html=True)

    with col_text:
        st.markdown("""
        <div style="padding: 8px 0; margin-left: 8px;">
            <div style="font-size: 11px; color: #718096; margin: 0; font-weight: 600; letter-spacing: 0.5px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">DIGITAL PROCESSING SYSTEMS</div>
            <div style="font-size: 16px; font-weight: 700; color: #2d3748; margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">Electrical Symbol Detector</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 60px 20px; text-align: center; border-radius: 12px; margin: 20px 0;">
        <h1 style="font-size: 42px; font-weight: 700; color: #2d3748; margin-bottom: 20px; line-height: 1.2;">
            Streamline Your <span style="color: #e63946;">Drawing Analysis</span>
        </h1>
        <p style="font-size: 18px; color: #4a5568; max-width: 700px; margin: 0 auto 30px; line-height: 1.6;">
            AI-powered analysis of engineering drawings for DPS Kuwait. Automatically analyze, detect, and count electrical symbols to find patterns and inconsistencies faster than ever before.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Your Analysis", key="hero_cta", use_container_width=True):
            st.session_state.view = 'analysis'
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("### ✨ Key Features")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 30px; border-radius: 12px; text-align: center; border: 1px solid #90caf9; height: 100%;">
            <div style="font-size: 40px; margin-bottom: 16px;">🔍</div>
            <h4 style="font-size: 18px; font-weight: 700; color: #1565c0; margin-bottom: 12px;">Smart Symbol Detection</h4>
            <p style="font-size: 14px; color: #1976d2; line-height: 1.5;">Detect cove lights, downlights, sockets, doors and more using a trained YOLO model with configurable confidence/IoU thresholds.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 30px; border-radius: 12px; text-align: center; border: 1px solid #81c784; height: 100%;">
            <div style="font-size: 40px; margin-bottom: 16px;">📄</div>
            <h4 style="font-size: 18px; font-weight: 700; color: #2e7d32; margin-bottom: 12px;">Multi-Page Support</h4>
            <p style="font-size: 14px; color: #388e3c; line-height: 1.5;">Upload PDFs with dozens of pages. We render each page to high-DPI images with seamless navigation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 30px; border-radius: 12px; text-align: center; border: 1px solid #ce93d8; height: 100%;">
            <div style="font-size: 40px; margin-bottom: 16px;">📊</div>
            <h4 style="font-size: 18px; font-weight: 700; color: #6a1b9a; margin-bottom: 12px;">Actionable Outputs</h4>
            <p style="font-size: 14px; color: #7b1fa2; line-height: 1.5;">Color legend, per-class counts, positions, and CSV export for comprehensive reporting.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 50px 20px; text-align: center; border-radius: 12px; margin: 20px 0; border: 2px solid #ffb74d;">
        <h2 style="font-size: 32px; font-weight: 700; color: #e65100; margin-bottom: 12px;">Ready to Transform Your Engineering Reviews?</h2>
        <p style="font-size: 16px; color: #bf360c; margin-bottom: 24px;">Join DPS Kuwait's engineering team in revolutionizing the way we analyze and validate electrical drawings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # More CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✨ Get Started Now", key="cta_btn", use_container_width=True):
            st.session_state.view = 'analysis'
            st.rerun()
    
    # Footer
    st.markdown("""
    <div style="background: #1a202c; color: #e2e8f0; padding: 40px 20px; text-align: center; margin-top: 40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 20px;">
                <div style="width: 32px; height: 32px; background: #e63946; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;">
                    DPS
                </div>
                <div>
                    <div style="font-size: 16px; font-weight: 600; color: #ffffff; margin: 0;">DIGITAL PROCESSING SYSTEMS</div>
                    <div style="font-size: 12px; color: #a0aec0; margin: 0;">DPS Kuwait</div>
                </div>
            </div>
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #2d3748;">
                <small style="color: #a0aec0; font-size: 12px;">Digital Processing Systems Kuwait - Electrical Symbol Detector</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_analysis_page() -> None:
    """Render the analysis page with proper layout matching the wireframe."""

    # Custom CSS
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: #f5f5f5;
    }
    .stButton > button {
        width: 100%;
    }
    div[data-testid="column"] {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    @keyframes slideInActive {
        from { transform: translateX(-10px); opacity: 0.8; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes checkmark {
        0% { transform: scale(0.5) rotate(-45deg); opacity: 0; }
        50% { transform: scale(1.2); }
        100% { transform: scale(1) rotate(0deg); opacity: 1; }
    }
    @keyframes shimmerSlide {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    @keyframes progressFill {
        0% { width: 0%; }
        100% { width: var(--progress-width, 0%); }
    }
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    .pipeline-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 8px;
    }
    .pipeline-container::-webkit-scrollbar {
        width: 6px;
    }
    .pipeline-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .pipeline-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    .pipeline-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    .pipeline-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .pipeline-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        gap: 12px;
    }
    .pipeline-icon {
        font-size: 24px;
        animation: bounce 2s infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }
    .pipeline-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    .pipeline-progress-wrapper {
        margin-bottom: 20px;
    }
    .pipeline-progress-bar {
        width: 100%;
        height: 8px;
        background: #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 8px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .pipeline-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        background-size: 200% 100%;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        animation: gradientShift 3s ease infinite;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 0%; }
    }
    .pipeline-progress-text {
        font-size: 12px;
        color: #6b7280;
        font-weight: 600;
        text-align: center;
    }
    .pipeline-steps {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        border-radius: 8px;
        background: #f9fafb;
        border-left: 4px solid #d1d5db;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .pipeline-step.pending {
        opacity: 0.6;
    }
    .pipeline-step.active {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    .pipeline-step.completed {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left-color: #10b981;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
    }
    @keyframes slideIn {
        from {
            transform: translateX(-10px);
            opacity: 0.8;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    .step-content {
        display: flex;
        align-items: center;
        gap: 12px;
        flex: 1;
    }
    .step-icon-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #f3f4f6;
        border: 2px solid #d1d5db;
        transition: all 0.4s ease;
        flex-shrink: 0;
    }
    .step-icon-wrapper.active {
        background: #dbeafe;
        border-color: #3b82f6;
        animation: pulse 1.5s ease-in-out infinite;
    }
    .step-icon-wrapper.completed {
        background: #d1fae5;
        border-color: #10b981;
        animation: scaleIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
        50% { transform: scale(1.05); box-shadow: 0 0 0 6px rgba(59, 130, 246, 0); }
    }
    @keyframes scaleIn {
        0% { transform: scale(0.5); }
        50% { transform: scale(1.15); }
        100% { transform: scale(1); }
    }
    .step-icon {
        font-size: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .step-info {
        flex: 1;
    }
    .step-name {
        font-size: 14px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 2px;
    }
    .step-status {
        font-size: 12px;
        color: #6b7280;
        font-weight: 500;
    }
    .active-text {
        color: #3b82f6;
        font-weight: 600;
        animation: blink 1.5s ease-in-out infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .step-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #d1d5db;
        transition: all 0.4s ease;
        flex-shrink: 0;
    }
    .step-indicator.active {
        background: #3b82f6;
        width: 12px;
        height: 12px;
        animation: pulse-dot 1.5s ease-in-out infinite;
    }
    .step-indicator.completed {
        background: #10b981;
    }
    @keyframes pulse-dot {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.3); }
    }
    .legend-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 16px;
        border-radius: 10px;
        margin-top: 16px;
        border: 2px solid #d0d8e0;
        max-height: 350px;
        overflow-y: auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .legend-container::-webkit-scrollbar {
        width: 6px;
    }
    .legend-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .legend-container::-webkit-scrollbar-thumb {
        background: #2196f3;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .legend-container::-webkit-scrollbar-thumb:hover {
        background: #1976d2;
    }
    .legend-title {
        font-weight: 700;
        margin-bottom: 12px;
        color: #1565c0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin: 8px 0;
        padding: 10px 12px;
        border-radius: 6px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .legend-item:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateX(4px) scale(1.02);
        background: #f0f7ff;
    }
    .legend-color-box {
        width: 22px;
        height: 22px;
        border-radius: 4px;
        margin-right: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.1);
        flex-shrink: 0;
    }
    .legend-label {
        font-size: 13px;
        color: #2c3e50;
        font-weight: 600;
    }
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa 0%, #f0f4f8 100%);
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
        transform: translateY(-2px);
        border-color: #2196f3;
    }
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Top-left corner branding with back button
    col_logo, col_text, col_back = st.columns([0.06, 0.84, 0.1])

    with col_logo:
        logo_path = APP_DIR / "logo.webp"
        if logo_path.exists():
            st.image(str(logo_path), width=40, use_container_width=False)
        else:
            st.markdown("""
            <div style="width: 40px; height: 40px; background: #e63946; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;">
                DPS
            </div>
            """, unsafe_allow_html=True)

    with col_text:
        st.markdown("""
        <div style="padding: 4px 0; margin-left: 8px;">
            <div style="font-size: 10px; color: #718096; margin: 0; font-weight: 600; letter-spacing: 0.5px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">DIGITAL PROCESSING SYSTEMS</div>
            <div style="font-size: 14px; font-weight: 700; color: #2d3748; margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">Electrical Symbol Detector</div>
        </div>
        """, unsafe_allow_html=True)

    with col_back:
        if st.button("⬅️ Back", key="back_home", help="Back to Home", use_container_width=True):
            st.session_state.view = 'landing'
            st.session_state.processing_complete = False
            st.session_state.detection_results = None
            st.rerun()

    st.divider()
    
    # Main Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Upload Section with container
        with st.container():
            st.markdown("### 📁 Upload Documents")
            st.caption("Supported formats: DWF, PDF, JPG, PNG, BMP, TIFF, WEBP")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["dwf", "pdf", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
                accept_multiple_files=False,
                key="file_uploader",
                label_visibility="collapsed"
            )
            
            # Run Analysis Button
            if uploaded_file:
                st.success(f"✅ File uploaded: **{uploaded_file.name}**")
                
                if not st.session_state.get('processing_complete', False):
                    if st.button("▶️ Run Analysis", type="primary", key="run_analysis", use_container_width=True):
                        process_file(uploaded_file)
                else:
                    st.success("✅ Analysis Complete!")
                    if st.button("🔄 Analyze New File", key="reset_analysis", use_container_width=True):
                        st.session_state.processing_complete = False
                        st.session_state.detection_results = None
                        st.rerun()
        
        st.divider()

        # Inference Results Section - Full Width
        if st.session_state.get('detection_results'):
            det = st.session_state['detection_results']

            # Detection Overview - Metrics
            st.markdown("### 📊 Detection Overview")
            metric_cols = st.columns(min(len(det['counts']), 4))
            for idx, (class_name, count) in enumerate(det['counts'].items()):
                with metric_cols[idx % 4]:
                    st.metric(class_name, count, delta=None)

            st.divider()

            # Page selector
            if det.get('images') and len(det['images']) > 1:
                col_page, col_info = st.columns([0.3, 0.7])
                with col_page:
                    page_num = st.selectbox(
                        "📄 Select Page",
                        range(1, len(det['images']) + 1),
                        format_func=lambda x: f"Page {x} of {len(det['images'])}",
                        key="page_selector",
                        label_visibility="collapsed"
                    )
                    st.session_state.selected_page = page_num
                with col_info:
                    st.info(f"📍 Viewing Page {page_num} of {len(det['images'])}")

            st.divider()

            # Display inference image with zoom/pan functionality
            st.markdown("### 🖼️ Inference Image")
            if det.get('annotated'):
                selected_idx = st.session_state.get('selected_page', 1) - 1
                if selected_idx < len(det['annotated']):
                    img_path = det['annotated'][selected_idx]
                    if Path(img_path).exists():
                        # Create interactive image viewer
                        st.markdown("""
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; border: 2px solid #e2e8f0;">
                            <p style="text-align: center; color: #718096; font-size: 12px; margin-bottom: 10px;">
                                💡 Tip: Use your mouse wheel to zoom, click and drag to pan
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.image(str(img_path), use_container_width=True, caption=f"Page {selected_idx + 1} - Detected Symbols with Annotations")
                    else:
                        st.warning("⚠️ Image file not found")
    
    with col_right:
        # Process Pipeline with Animated Progress Bar
        # Initialize pipeline status in session state if not present
        if 'pipeline_status' not in st.session_state:
            st.session_state.pipeline_status = {
                'File Upload & Validation': 'pending',
                'DWF to PDF Conversion': 'pending',
                'PDF to Image Rendering': 'pending',
                'Model Loading': 'pending',
                'Object Detection Inference': 'pending',
                'Results Processing': 'pending'
            }

        steps_list = [
            'File Upload & Validation',
            'DWF to PDF Conversion',
            'PDF to Image Rendering',
            'Model Loading',
            'Object Detection Inference',
            'Results Processing'
        ]

        # Build pipeline steps data
        pipeline_steps = []
        for step_name in steps_list:
            status = st.session_state.pipeline_status.get(step_name, 'pending')
            pipeline_steps.append({
                'name': step_name,
                'status': status
            })

        # Calculate progress percentage
        completed_count = sum(1 for step in pipeline_steps if step['status'] == 'completed')
        progress_percentage = int((completed_count / len(pipeline_steps)) * 100)

        # Display pipeline header
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.markdown("📋", unsafe_allow_html=True)
        with col2:
            st.markdown("### Process Pipeline", unsafe_allow_html=True)

        # Display progress bar
        st.progress(progress_percentage / 100, text=f"{progress_percentage}% Complete")

        # Display each step
        for step in pipeline_steps:
            status = step['status']
            step_name = step['name']

            if status == 'completed':
                icon = '✅'
                color = '#10b981'
            elif status == 'active':
                icon = '⚙️'
                color = '#3b82f6'
            else:
                icon = '📁'
                color = '#d1d5db'

            status_text = '✓ Completed' if status == 'completed' else ('Processing...' if status == 'active' else 'Pending')

            # Create step display with HTML styling
            step_html = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px;
                border-radius: 8px;
                background: {'linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)' if status == 'completed' else ('linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)' if status == 'active' else '#f9fafb')};
                border-left: 4px solid {color};
                margin: 8px 0;
                transition: all 0.4s ease;
                box-shadow: {'0 2px 8px rgba(16, 185, 129, 0.15)' if status == 'completed' else ('0 4px 12px rgba(59, 130, 246, 0.2)' if status == 'active' else 'none')};
                opacity: {'1' if status != 'pending' else '0.6'};
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 36px;
                    height: 36px;
                    border-radius: 50%;
                    background: {'#d1fae5' if status == 'completed' else ('#dbeafe' if status == 'active' else '#f3f4f6')};
                    border: 2px solid {color};
                    flex-shrink: 0;
                    font-size: 18px;
                ">
                    {icon}
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 14px; font-weight: 600; color: #1f2937; margin-bottom: 2px;">
                        {step_name}
                    </div>
                    <div style="font-size: 12px; color: {'#3b82f6' if status == 'active' else '#6b7280'}; font-weight: 500;">
                        {status_text}
                    </div>
                </div>
                <div style="
                    width: {'12px' if status == 'active' else '8px'};
                    height: {'12px' if status == 'active' else '8px'};
                    border-radius: 50%;
                    background: {color};
                    flex-shrink: 0;
                "></div>
            </div>
            """
            st.markdown(step_html, unsafe_allow_html=True)

        # Auto-refresh while processing is active
        if not st.session_state.get('processing_complete', False):
            # Check if any step is active
            is_processing = any(status == 'active' for status in st.session_state.pipeline_status.values())
            if is_processing:
                import time
                time.sleep(0.5)
                st.rerun()

        st.divider()

        # Color Legend for Symbols
        st.markdown("### 🎨 Symbol Legend")

        class_colors = {
            'Cove Light': '#FF0000',
            'Door': '#00FF00',
            'Emergency Light Fitting': '#0000FF',
            'Fluorescent Light': '#FFFF00',
            'exit': '#FF00FF',
            'Downlight': '#00FFFF',
            'Socket Outlet': '#800080'
        }

        st.markdown('<div class="legend-container">', unsafe_allow_html=True)
        st.markdown('<div class="legend-title">📍 Detection Classes & Colors</div>', unsafe_allow_html=True)

        for class_name, color in class_colors.items():
            st.markdown(f"""
            <div class="legend-item">
                <div class="legend-color-box" style="background-color: {color};"></div>
                <div class="legend-label">{class_name}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # Summary
        if st.session_state.get('detection_results'):
            st.markdown("### 📊 Summary")
            det = st.session_state['detection_results']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Pages", len(det.get('images', [])))
            with col2:
                st.metric("🔍 Classes", len(det.get('counts', {})))
            with col3:
                st.metric("📍 Detections", sum(det.get('counts', {}).values()))

            st.divider()

            # Detection Overview
            st.markdown("### 📋 Detection Overview")

            if det.get('counts'):
                import pandas as pd
                df_data = []
                for class_name, count in det['counts'].items():
                    df_data.append({
                        'Class': class_name,
                        'Count': count
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Export CSV button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name="detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )


def update_pipeline_status(step_name: str, status: str) -> None:
    """Update the pipeline status for a specific step - NO RERUN."""
    if 'pipeline_status' not in st.session_state:
        st.session_state.pipeline_status = {
            'File Upload & Validation': 'pending',
            'DWF to PDF Conversion': 'pending',
            'PDF to Image Rendering': 'pending',
            'Model Loading': 'pending',
            'Object Detection Inference': 'pending',
            'Results Processing': 'pending'
        }
    st.session_state.pipeline_status[step_name] = status


def process_file(uploaded_file) -> None:
    """Process the uploaded file and update session state."""
    if st.session_state.get('processing_complete', False):
        return

    try:
        # Initialize pipeline status
        if 'pipeline_status' not in st.session_state:
            st.session_state.pipeline_status = {
                'File Upload & Validation': 'pending',
                'DWF to PDF Conversion': 'pending',
                'PDF to Image Rendering': 'pending',
                'Model Loading': 'pending',
                'Object Detection Inference': 'pending',
                'Results Processing': 'pending'
            }

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: File Upload & Validation
        status_text.text("📤 Step 1/6: File Upload & Validation...")
        update_pipeline_status('File Upload & Validation', 'active')
        progress_bar.progress(16)

        ensure_dirs()
        uploaded_path = save_uploaded_file(uploaded_file)
        update_pipeline_status('File Upload & Validation', 'completed')

        all_images: List[Path] = []
        suffix = uploaded_path.suffix.lower()

        # Step 2: DWF to PDF Conversion (if needed)
        status_text.text("🔄 Step 2/6: DWF to PDF Conversion...")
        progress_bar.progress(32)

        if suffix == ".dwf":
            update_pipeline_status('DWF to PDF Conversion', 'active')
            pdf_paths = convert_dwf_to_pdf(uploaded_path, PDF_DIR)
            update_pipeline_status('DWF to PDF Conversion', 'completed')
        else:
            update_pipeline_status('DWF to PDF Conversion', 'completed')

        # Step 3: PDF to Image Rendering
        status_text.text("🖼️ Step 3/6: PDF to Image Rendering...")
        progress_bar.progress(48)
        update_pipeline_status('PDF to Image Rendering', 'active')

        if suffix == ".dwf":
            for p in pdf_paths:
                imgs = render_pdf_to_images(p, IMG_DIR, dpi=200)
                all_images.extend(imgs)
        elif suffix == ".pdf":
            imgs = render_pdf_to_images(uploaded_path, IMG_DIR, dpi=200)
            all_images.extend(imgs)
        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            all_images = [uploaded_path]

        update_pipeline_status('PDF to Image Rendering', 'completed')

        # Step 4: Model Loading
        status_text.text("⚙️ Step 4/6: Model Loading...")
        progress_bar.progress(64)
        update_pipeline_status('Model Loading', 'active')

        weights_path = Path(DEFAULT_WEIGHTS)
        update_pipeline_status('Model Loading', 'completed')

        # Step 5: Object Detection Inference
        status_text.text("🔍 Step 5/6: Object Detection Inference...")
        progress_bar.progress(80)
        update_pipeline_status('Object Detection Inference', 'active')

        if not weights_path.exists():
            st.warning(f"⚠️ Model weights not found. Using mock data for demo.")
            counts = {'Cove Light': 14, 'Door': 3, 'Emergency Light Fitting': 2, 'Fluorescent Light': 8}
            annotated = all_images
            page_detections = []
        else:
            counts, annotated, page_detections = run_yolo_inference(all_images, weights_path, conf=0.10, iou=0.20)

        update_pipeline_status('Object Detection Inference', 'completed')

        # Step 6: Results Processing
        status_text.text("📊 Step 6/6: Results Processing...")
        progress_bar.progress(96)
        update_pipeline_status('Results Processing', 'active')

        st.session_state.detection_results = {
            'counts': counts,
            'annotated': annotated,
            'page_detections': page_detections,
            'images': all_images
        }

        update_pipeline_status('Results Processing', 'completed')
        st.session_state.processing_complete = True
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        st.success("✅ Analysis complete! Check the results below.")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.session_state.processing_complete = False


if __name__ == "__main__":
    main()
    