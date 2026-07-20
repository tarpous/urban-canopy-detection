# Deploying the demo as a private Hugging Face Space

The Space is created and pushed from *your* account (`tarpous`) — this repo
stages every file it needs. One-time setup:

1. Create a token at <https://huggingface.co/settings/tokens> (write scope) and
   log in locally:

   ```bash
   uv run --with huggingface_hub huggingface-cli login
   ```

2. Create a **private** Gradio Space and push the `app/` contents to it:

   ```bash
   uv run --with huggingface_hub huggingface-cli repo create \
     urban-canopy-detection --repo-type space --space-sdk gradio --private -y

   git clone https://huggingface.co/spaces/tarpous/urban-canopy-detection hf-space
   cp app/app.py app/requirements.txt app/README.md hf-space/
   mkdir -p hf-space/weights
   cp runs/detect/runs/yolo26/weights/best.pt hf-space/weights/   # your trained YOLO26-s checkpoint
   cd hf-space && git lfs install && git lfs track "*.pt" && git add -A
   git commit -m "deploy urban canopy detection demo" && git push
   ```

The Space stays private (visible only to you) until you flip it to public in the
Space's Settings. Add the resulting URL to the repo README's Demo section.

> The app boots without `weights/best.pt` in a synthetic-box mode, so the Space
> is inspectable even before you upload a checkpoint.
