**Benchmark:** NeonTreeEvaluation (weecology), evaluation split · **split:** site-disjoint (geographic blocking)

| Model | mAP@50 | mAP@[.5:.95] | P | R | R small | R med | R large | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| RF-DETR (fine-tuned) 🕒 | — | — | — | — | — | — | — | SAHI sliced (640/128) |
| YOLO26-s (fine-tuned) | 0.391 | 0.160 | 0.577 | 0.494 | 0.433 | 0.532 | 0.500 | tile (≤imgsz) |
| YOLO11-s (fine-tuned, lineage row) | 0.455 | 0.179 | 0.530 | 0.562 | 0.527 | 0.584 | 0.333 | tile (≤imgsz) |
| DeepForest RetinaNet (published baseline) | 0.583 | 0.223 | 0.745 | 0.615 | 0.505 | 0.682 | 0.667 | whole-image, CPU |

**SAHI effect (YOLO26-s):** whole-image mAP@50 — → sliced —.

🕒 = awaiting the T4 fine-tune run; see `notebooks/`.
