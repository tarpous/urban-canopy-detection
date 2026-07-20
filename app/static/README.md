---
title: Urban Canopy Detection
sdk: static
pinned: false
license: mit
short_description: In-browser tree-crown detection (YOLO26-s ONNX + WASM).
---

# Urban canopy detection — in-browser demo

Upload an aerial RGB image (or try a bundled NEON tile); a YOLO26-s detector,
fine-tuned on the open [NeonTreeEvaluation](https://github.com/weecology/NeonTreeEvaluation)
benchmark and exported to ONNX, runs **entirely in your browser** via ONNX
Runtime Web (WASM) — nothing is uploaded to a server. Returns crown boxes, a
count, and a downloadable GeoJSON.

Full study, metrics and code:
[github.com/tarpous/urban-canopy-detection](https://github.com/tarpous/urban-canopy-detection).
