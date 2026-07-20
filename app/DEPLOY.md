# Deploying the demo

The **live demo** is a free **static** Hugging Face Space that runs the detector
in the browser (ONNX Runtime Web) — [`app/static/`](static/). Hosting Gradio
Spaces on the free tier now requires HF PRO, so the static route is the free,
public, zero-backend option; `app/app.py` is kept as a Gradio variant for local
use or a PRO Space.

## Static Space (live, free)

```bash
uv run python scripts/export_onnx.py                 # writes app/static/best.onnx
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import HfApi, create_repo
api = HfApi(); repo = f"{api.whoami()['name']}/urban-canopy-detection"
create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(folder_path="app/static", repo_id=repo, repo_type="space",
                  commit_message="in-browser YOLO26 demo")
print("https://huggingface.co/spaces/" + repo)
PY
```

Requires a one-time `huggingface-cli login` (write token). The Space is public
and runs entirely client-side — nothing is uploaded to a server.

## Gradio Space (local, or a PRO Space)

Run locally with `uv run --with gradio python app/app.py`, or push `app/app.py`
+ `app/requirements.txt` + `app/README.md` (and a `weights/best.pt`) to a
Gradio Space — that requires an HF PRO subscription on the free CPU tier.
