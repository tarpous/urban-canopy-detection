---
title: Urban Canopy Detection
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Tree-crown detection on aerial imagery (YOLO26 + SAHI).
---

# Urban canopy detection — demo

Upload an aerial RGB image; the model tiles it (SAHI), detects tree crowns, and
returns the boxes, a crown count, and a downloadable GeoJSON. It reuses the
tested slicing / NMS / georeferencing code from
[tarpous/urban-canopy-detection](https://github.com/tarpous/urban-canopy-detection);
with no weights present it runs in a synthetic mode so the Space always boots.

Place a fine-tuned checkpoint at `weights/best.pt` (or upload it to the Space)
to serve the real detector.
