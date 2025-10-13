from __future__ import annotations
import os
import io
import sys
import asyncio
from typing import Optional

# Ensure HF mirror endpoint for faster model access if available
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from diffusers.utils import check_min_version

# Local modules (handle folder with hyphen by appending to sys.path)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_flux_dir = os.path.join(_this_dir, "FLUX-Controlnet-Inpainting")
if _flux_dir not in sys.path:
    sys.path.append(_flux_dir)

from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

# Validate diffusers version
check_min_version("0.30.2")

app = FastAPI(title="FLUX ControlNet Inpainting API")

# Global state for model/pipeline
_device = "cuda:5" if torch.cuda.is_available() else "cpu"
_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
_pipe: Optional[FluxControlNetInpaintingPipeline] = None
_pipe_lock = asyncio.Lock()

# Paths and model IDs (adapt if your paths differ)
# TRANSFORMER_REPO_PATH = "/data/fanhongxing/ckpt/FLUX.1-dev"
TRANSFORMER_REPO_PATH = "/data/ckpt/FLUX.1-dev"
CONTROLNET_MODEL_ID = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"


def _require_cuda():
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA GPU is required for this pipeline but not available.")


def _load_pipeline_if_needed():
    global _pipe
    if _pipe is not None:
        return _pipe

    _require_cuda()

    controlnet = FluxControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(
        TRANSFORMER_REPO_PATH, subfolder="transformer", torch_dtype=_dtype
    )

    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        TRANSFORMER_REPO_PATH,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=_dtype,
    ).to(_device)

    # Ensure modules use the expected dtype
    pipe.transformer.to(_dtype)
    pipe.controlnet.to(_dtype)

    _pipe = pipe
    return _pipe


def _read_image(file: UploadFile) -> Image.Image:
    try:
        data = file.file.read()
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file '{file.filename}': {e}")


@app.get("/health")
async def health():
    # Lightweight health without model load
    return JSONResponse({"status": "ok", "device": _device})


@app.post("/inpaint")
async def inpaint(
    prompt: str = Form(..., description="Text prompt to guide inpainting"),
    image: UploadFile = File(..., description="RGB input image (PNG/JPG)", alias="image"),
    mask: UploadFile = File(..., description="Mask image (white = keep/visible, black = inpaint)", alias="mask"),
    steps: int = Form(28, description="Number of inference steps"),
    guidance_scale: float = Form(3.5, description="Classifier-free guidance scale"),
    controlnet_conditioning_scale: float = Form(0.9, description="ControlNet conditioning scale"),
    true_guidance_scale: float = Form(1.0, description="True guidance scale for beta pipeline"),
    seed: Optional[int] = Form(None, description="Random seed; fixed for reproducibility if provided"),
):
    # Parse images
    base_img = _read_image(image)
    mask_img = _read_image(mask)

    # Keep original size to resize result back
    w, h = base_img.size

    # Resize both inputs to 1024x1024 as in the reference script
    target_size = (1024, 1024)
    base_resized = base_img.resize(target_size)
    mask_resized = mask_img.resize(target_size)

    # Prepare generator if seed provided
    generator = None
    if seed is not None:
        _require_cuda()
        generator = torch.Generator(device=_device).manual_seed(int(seed))

    # Run pipeline under a lock to avoid GPU memory thrash on concurrent requests
    async with _pipe_lock:
        pipe = _load_pipeline_if_needed()
        try:
            out = pipe(
                prompt=prompt,
                height=target_size[1],
                width=target_size[0],
                control_image=base_resized,
                control_mask=mask_resized,
                num_inference_steps=int(steps),
                generator=generator,
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                guidance_scale=float(guidance_scale),
                negative_prompt="",
                true_guidance_scale=float(true_guidance_scale),
            )
            result_img: Image.Image = out.images[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline inference failed: {e}")

    # Resize back to original resolution
    result_img = result_img.resize((w, h))

    # Return as PNG bytes
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    # Optional: run with `python api_server.py` for local testing
    import uvicorn

    # Do not eagerly load models on startup unless desired; they will load on first request
    uvicorn.run(
        "api_server_flux_inpainting:app",
        host="0.0.0.0",
        port=8051,
        reload=False,
        workers=1,
    )
