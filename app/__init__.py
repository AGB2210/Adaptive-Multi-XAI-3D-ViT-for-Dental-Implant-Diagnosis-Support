"""Browser app for per-site implant screening on a CBCT scan.

    python -m app --config configs/app.yaml

A FastAPI server around `src/inference/`, plus a static page. The server holds
no model logic of its own: it stores uploads, runs jobs concurrently, and
serialises what `src/inference` returns.
"""
