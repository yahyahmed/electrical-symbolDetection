## Streamlit DWF/PDF Inference App

Run a web UI that accepts `.dwf` or `.pdf`, converts as needed, renders pages to images, performs YOLO inference with `best.pt`, and returns per-class counts plus annotated images.

### Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Notes
- Set ConvertAPI credentials via environment variable `CONVERTAPI_SECRET`. If omitted, DWF uploads will not be converted.
- Place your YOLO weights at `application/best.pt` or provide a custom path in the UI.

