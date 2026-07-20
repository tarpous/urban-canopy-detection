**Benchmark:** NeonTreeEvaluation (weecology), evaluation split · **split:** site-disjoint (geographic blocking)

| Model | mAP@50 | mAP@[.5:.95] | P | R | R small | R med | R large | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11-s (fine-tuned, lineage row) | 0.455 | 0.179 | 0.530 | 0.562 | 0.527 | 0.584 | 0.333 | tile (≤imgsz) |
| DeepForest RetinaNet (published baseline) | 0.583 | 0.223 | 0.745 | 0.615 | 0.505 | 0.682 | 0.667 | whole-image, CPU |
| RF-DETR (fine-tuned) | 0.624 | 0.256 | 0.626 | 0.656 | 0.567 | 0.710 | 0.833 | tile (640) |
| YOLO26-s (fine-tuned) | 0.410 | 0.165 | 0.560 | 0.512 | 0.480 | 0.534 | 0.167 | tile (≤imgsz) |

**SAHI effect (YOLO11-s):** whole-image mAP@50 0.455 → sliced 0.450.
