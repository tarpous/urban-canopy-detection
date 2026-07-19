**Benchmark:** NeonTreeEvaluation (weecology), evaluation split · **split:** site-disjoint (geographic blocking)

| Model | mAP@50 | mAP@[.5:.95] | P | R | R small | R med | R large | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepForest RetinaNet (published baseline) 🕒 | — | — | — | — | — | — | — | whole-image, CPU |
| YOLO26-s (fine-tuned) 🕒 | — | — | — | — | — | — | — | SAHI sliced (640/128) |
| RF-DETR (fine-tuned) 🕒 | — | — | — | — | — | — | — | SAHI sliced (640/128) |
| YOLO11-s (fine-tuned, lineage row) 🕒 | — | — | — | — | — | — | — | SAHI sliced (640/128) |

**SAHI effect (YOLO26-s):** whole-image mAP@50 — → sliced —.

🕒 = awaiting the T4 fine-tune run; see `notebooks/`.
