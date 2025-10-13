import os
import io
from typing import Dict, Any, Optional
import requests
from PIL import Image, ImageFilter

class InpaintAdapter:
    """
    A client for FLUX-Controlnet-Inpainting api_server.py.
    Expected: POST /inpaint with image + mask + prompt
    """
    def __init__(self, base_url: Optional[str] = None, timeout: int = 120, dry_run: bool = False):
        # Default ports as requested: inpainting service at 8051
        self.base_url = base_url or os.getenv("INPAINT_API_BASE", "http://0.0.0.0:8051")
        self.timeout = timeout
        self.dry_run = dry_run
        # Allow overriding endpoint path
        self.inpaint_path = os.getenv("INPAINT_API_PATH", "/inpaint")
        self._fallback_paths = [
            "/inpaint",
            "/api/inpaint",
            "/edit",
            "/run",
            "/predict",
            "/flux/inpaint",
        ]

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, extra: Optional[Dict[str, Any]] = None) -> Image.Image:
        if self.dry_run:
            # Simple demo: paste blurred region where mask=255
            img = image.convert("RGBA").copy()
            blurred = image.convert("RGBA").filter(ImageFilter.BLUR)
            mask_rgba = mask.convert("L")
            img.paste(blurred, (0, 0), mask_rgba)
            return img.convert("RGB")

        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        img_buf.seek(0)

        mask_buf = io.BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_buf.seek(0)

        files = {
            "image": ("image.png", img_buf, "image/png"),
            "mask": ("mask.png", mask_buf, "image/png"),
        }
        # include both 'prompt' and 'text' to match various servers
        data = {"prompt": prompt, "text": prompt}
        if extra:
            data.update(extra)

        paths_to_try = [self.inpaint_path] + [p for p in self._fallback_paths if p != self.inpaint_path]
        last_exc = None
        resp = None
        for p in paths_to_try:
            try:
                url = f"{self.base_url}{p}"
                resp = requests.post(url, data=data, files=files, timeout=self.timeout)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                last_exc = e
                resp = None
                continue
        if resp is None:
            if last_exc:
                raise last_exc
            raise RuntimeError("Inpaint request failed with all paths")
        out = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return out
