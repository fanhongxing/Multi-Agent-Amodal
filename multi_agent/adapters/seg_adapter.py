import os
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import json
import requests
from PIL import Image, ImageChops
import numpy as np

class SegmentationAdapter:
    """
    A client for Grounded-Segment-Anything api_server.py.

    Expected behavior (based on common patterns):
    - POST /segment with JSON or multipart containing image and text prompt or boxes
    - Returns list of masks or a combined mask
    We keep it generic and configurable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 60,
        dry_run: bool = False,
        backend: Optional[str] = None,
    ):
        """
        Parameters:
        - base_url: server base url. If None, chooses default by backend
            sam  -> env SEG_API_BASE or http://0.0.0.0:8050
            xsam -> env XSAM_API_BASE or http://0.0.0.0:8052
        - backend: 'sam' (default) or 'xsam' (X-SAM FastAPI)
            Can also be set via env SEG_BACKEND
        """
        self.backend = (backend or os.getenv("SEG_BACKEND", "sam")).strip().lower()
        if self.backend not in {"sam", "xsam"}:
            self.backend = "sam"

        if base_url:
            self.base_url = base_url
        else:
            if self.backend == "xsam":
                self.base_url = os.getenv("XSAM_API_BASE", "http://0.0.0.0:8052")
            else:
                # Default ports as requested: segmentation service at 8050
                self.base_url = os.getenv("SEG_API_BASE", "http://0.0.0.0:8050")

        self.timeout = timeout
        self.dry_run = dry_run
        # Allow overriding the endpoint path if the server differs (mainly for 'sam' server)
        self.segment_path = os.getenv("SEG_API_SEGMENT_PATH", "/predict")
        # Fallback paths commonly seen in segmentation servers (for 'sam')
        self._fallback_paths = [
            "/predict",
            "/segment",
            "/api/segment",
            "/seg",
            "/run",
            "/segment-anything",
            "/sam",
            "/grounded",
            "/grounding/segment",
        ]

    def segment(self, image: Image.Image, text: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.dry_run:
            # Create a dummy rectangular mask for demo
            w, h = image.size
            mask = Image.new("L", (w, h), 0)
            x0, y0, x1, y1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
            for x in range(x0, x1):
                for y in range(y0, y1):
                    mask.putpixel((x, y), 255)
            # Encode mask as PNG base64 to mimic server response
            buf = io.BytesIO()
            mask.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            payload = {
                "detections": [
                    {
                        "class_name": (text or "object"),
                        "confidence": 1.0,
                        "bounding_box_xyxy": [x0, y0, x1, y1],
                        "mask_png_base64": b64,
                    }
                ]
            }
            return {"mask": mask, "payload": payload}

        # Route to the selected backend
        if (self.backend or "sam") == "xsam":
            return self._segment_xsam(image, text=text, extra=extra or {})
        else:
            return self._segment_sam(image, text=text, extra=extra or {})

    # ---------------------
    # SAM-like generic client
    # ---------------------
    def _segment_sam(self, image: Image.Image, text: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        files = {"image": ("image.png", buf, "image/png")}
        # Try default path, then fallbacks if 404
        paths_to_try = [self.segment_path] + [p for p in self._fallback_paths if p != self.segment_path]
        last_exc = None
        resp = None
        for p in paths_to_try:
            try:
                url = f"{self.base_url}{p}"
                # Build payload per endpoint conventions
                if p.endswith("/predict"):
                    # Allow caller to override class names explicitly (list[str] or str JSON)
                    cls_override = extra.get("class_names") if isinstance(extra, dict) else None
                    if isinstance(cls_override, list):
                        cls_field = json.dumps(cls_override)
                    elif isinstance(cls_override, str) and cls_override.strip().startswith("["):
                        cls_field = cls_override
                    else:
                        cls_field = json.dumps([text or "object"])  # default
                    payload = {
                        "class_names": cls_field,
                        "return_masks": str(extra.get("return_masks", True)).lower(),
                    }
                    for key in ["box_threshold", "text_threshold", "nms_threshold"]:
                        if key in extra:
                            payload[key] = str(extra[key])
                    resp = requests.post(url, data=payload, files=files, timeout=self.timeout)
                else:
                    payload = {"text": text or "", "prompt": text or ""}
                    payload.update(extra)
                    resp = requests.post(url, data=payload, files=files, timeout=self.timeout)
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
            raise RuntimeError("Segmentation request failed with all paths")
        # Expecting either an image mask or base64; be robust to JSON variants
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            payload = resp.json()
            def _decode_mask_from_payload(p: Dict[str, Any]) -> Optional[Image.Image]:
                # single base64 fields
                for k in ["mask", "mask_png", "mask_base64", "mask_b64", "mask_png_base64"]:
                    v = p.get(k)
                    if isinstance(v, str) and v:
                        b64 = v
                        if "," in b64 and b64.strip().startswith("data:"):
                            b64 = b64.split(",", 1)[1]
                        try:
                            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
                            return img
                        except Exception:
                            pass
                # url fields
                for k in ["mask_url", "mask_png_url"]:
                    v = p.get(k)
                    if isinstance(v, str) and v:
                        url = v if v.startswith("http") else f"{self.base_url.rstrip('/')}/{v.lstrip('/')}"
                        try:
                            r = requests.get(url, timeout=self.timeout)
                            r.raise_for_status()
                            return Image.open(io.BytesIO(r.content)).convert("L")
                        except Exception:
                            pass
                # list of masks
                for k in ["masks", "mask_list"]:
                    v = p.get(k)
                    if isinstance(v, list) and v:
                        imgs: List[Image.Image] = []
                        for item in v:
                            if isinstance(item, str):
                                s = item
                                if "," in s and s.strip().startswith("data:"):
                                    s = s.split(",", 1)[1]
                                try:
                                    img = Image.open(io.BytesIO(base64.b64decode(s))).convert("L")
                                    imgs.append(img)
                                    continue
                                except Exception:
                                    pass
                            if isinstance(item, list):
                                try:
                                    arr = np.array(item, dtype=np.uint8)
                                    if arr.max() <= 1:
                                        arr = arr * 255
                                    imgs.append(Image.fromarray(arr, mode="L"))
                                    continue
                                except Exception:
                                    pass
                        if imgs:
                            base = imgs[0]
                            for im in imgs[1:]:
                                base = ImageChops.lighter(base, im)
                            return base
                # array-like mask
                for k in ["mask_array", "binary_mask", "mask_mat"]:
                    v = p.get(k)
                    if isinstance(v, list) and v:
                        try:
                            arr = np.array(v, dtype=np.uint8)
                            if arr.max() <= 1:
                                arr = arr * 255
                            return Image.fromarray(arr, mode="L")
                        except Exception:
                            pass
                # detections array from /predict
                dets = p.get("detections")
                if isinstance(dets, list) and dets:
                    imgs: List[Image.Image] = []
                    # Prefer masks of matching class_name
                    target = (text or "").strip().lower()
                    # Sort by confidence desc if available
                    try:
                        dets_sorted = sorted(dets, key=lambda d: float(d.get("confidence", 0)), reverse=True)
                    except Exception:
                        dets_sorted = dets
                    for det in dets_sorted:
                        cls = str(det.get("class_name", "")).strip().lower()
                        if target and cls and target not in cls and cls != target:
                            # if target provided and doesn't match, skip; fallback later if none
                            continue
                        # support common mask keys in detections
                        m = det.get("mask") or det.get("mask_base64") or det.get("mask_png_base64") or det.get("mask_png")
                        if isinstance(m, str) and m:
                            s = m
                            if "," in s and s.strip().startswith("data:"):
                                s = s.split(",", 1)[1]
                            try:
                                img = Image.open(io.BytesIO(base64.b64decode(s))).convert("L")
                                imgs.append(img)
                            except Exception:
                                pass
                    # If no matching class collected, try all
                    if not imgs:
                        for det in dets_sorted:
                            m = det.get("mask") or det.get("mask_base64") or det.get("mask_png_base64") or det.get("mask_png")
                            if isinstance(m, str) and m:
                                s = m
                                if "," in s and s.strip().startswith("data:"):
                                    s = s.split(",", 1)[1]
                                try:
                                    img = Image.open(io.BytesIO(base64.b64decode(s))).convert("L")
                                    imgs.append(img)
                                except Exception:
                                    pass
                    if imgs:
                        base = imgs[0]
                        for im in imgs[1:]:
                            base = ImageChops.lighter(base, im)
                        return base
                return None

            mask_img = _decode_mask_from_payload(payload)
            if mask_img is None:
                for key in ["data", "result", "output"]:
                    sub = payload.get(key)
                    if isinstance(sub, dict):
                        mask_img = _decode_mask_from_payload(sub)
                        if mask_img is not None:
                            break
            # Do not raise on missing mask; return payload so caller can compose masks per class
            return {"mask": mask_img, "payload": payload}
        # Otherwise assume raw image bytes (treat as mask image)
        mask_bytes = resp.content
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        return {"mask": mask_img, "payload": None}

    # ---------------------
    # X-SAM client (/predict)
    # ---------------------
    def _segment_xsam(self, image: Image.Image, text: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
        # Build prompt list
        prompt_list: List[str]
        cls_override = extra.get("class_names")
        if isinstance(cls_override, list) and cls_override:
            prompt_list = [str(x) for x in cls_override]
        elif isinstance(cls_override, str) and cls_override.strip().startswith("["):
            try:
                parsed = json.loads(cls_override)
                if isinstance(parsed, list) and parsed:
                    prompt_list = [str(x) for x in parsed]
                else:
                    prompt_list = [text or "object"]
            except Exception:
                prompt_list = [text or "object"]
        else:
            prompt_list = [text or "object"]

        task_name = str(extra.get("xsam_task", "refseg"))
        score_thr = float(extra.get("score_thr", 0.5))

        # prepare image multipart
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        files = {"image": ("image.png", buf, "image/png")}
        data = {
            "task_name": task_name,
            "prompt": json.dumps(prompt_list, ensure_ascii=False),
            "score_thr": str(score_thr),
        }
        url = f"{self.base_url.rstrip('/')}/predict"
        resp = requests.post(url, files=files, data=data, timeout=self.timeout)
        resp.raise_for_status()
        result = resp.json()

        payload: Dict[str, Any] = {}
        detections: List[Dict[str, Any]] = []

        # Prefer refseg structured result
        refseg = result.get("refseg_result") if isinstance(result, dict) else None
        if isinstance(refseg, dict):
            for d in refseg.get("detections", []) or []:
                try:
                    detections.append({
                        "class_name": d.get("class_name", "object"),
                        "bounding_box_xyxy": d.get("bounding_box_xyxy"),
                        "mask_png_base64": d.get("mask_png_base64"),
                        "confidence": d.get("confidence", 1.0),
                    })
                except Exception:
                    continue
        else:
            # Fallback: single seg image
            seg_b64 = result.get("seg_output_png_base64")
            if isinstance(seg_b64, str) and seg_b64:
                try:
                    b64 = seg_b64
                    if "," in b64 and b64.strip().startswith("data:"):
                        b64 = b64.split(",", 1)[1]
                    # Ensure it's a valid image
                    _ = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
                    detections.append({
                        "class_name": prompt_list[0] if prompt_list else (text or "object"),
                        "mask_png_base64": b64,
                        "confidence": 1.0,
                    })
                except Exception:
                    pass

        payload["detections"] = detections
        # Return payload primarily; no need to compose mask here because Orchestrator merges per class
        return {"mask": None, "payload": payload}
