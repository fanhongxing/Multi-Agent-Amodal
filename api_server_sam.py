import io
import os
import sys
import base64
import json
from typing import List, Optional, Any

import cv2
import numpy as np
import torch
import torchvision
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 'supervision' was previously imported but not used; removing to avoid optional dependency issues.


# ---------------------------
# Config
# ---------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Resolve important directories relative to this file, with env overrides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Root of the Grounded-Segment-Anything repo. Override via env var GSA_DIR if needed.
GSA_DIR = os.getenv('GSA_DIR', os.path.join(BASE_DIR, 'Grounded-Segment-Anything'))

# Ensure Python can import subpackages from GSA repo when not installed as site-packages
def _ensure_sys_path_for_gsa() -> None:
    candidates = [
        GSA_DIR,
        os.path.join(GSA_DIR, 'GroundingDINO'),
        os.path.join(GSA_DIR, 'segment_anything'),
    ]
    for p in candidates:
        try:
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
        except Exception:
            # Be permissive; import will try fallback later
            pass

# These default paths mirror the demo script; override with env vars when needed
GROUNDING_DINO_CONFIG_PATH = os.getenv(
    'GROUNDING_DINO_CONFIG_PATH',
    os.path.join(GSA_DIR, 'GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py'),
)
GROUNDING_DINO_CHECKPOINT_PATH = os.getenv(
    'GROUNDING_DINO_CHECKPOINT_PATH',
    os.path.join(GSA_DIR, 'groundingdino_swint_ogc.pth'),
)
SAM_ENCODER_VERSION = os.getenv('SAM_ENCODER_VERSION', 'vit_h')
SAM_CHECKPOINT_PATH = os.getenv('SAM_CHECKPOINT_PATH', os.path.join(GSA_DIR, 'sam_vit_h_4b8939.pth'))

# Inference thresholds
DEFAULT_BOX_THRESHOLD = float(os.getenv('BOX_THRESHOLD', '0.25'))
DEFAULT_TEXT_THRESHOLD = float(os.getenv('TEXT_THRESHOLD', '0.25'))
DEFAULT_NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', '0.8'))


# ---------------------------
# Lazy models (load once)
# ---------------------------
grounding_dino_model: Optional[Any] = None
sam_predictor: Optional[Any] = None


def _lazy_init_models():
    global grounding_dino_model, sam_predictor
    # Make sure local repo paths are importable
    _ensure_sys_path_for_gsa()
    if grounding_dino_model is None:
        # import GroundingDINO dynamically with local path fallback
        try:
            from groundingdino.util.inference import Model as GDModel  # type: ignore
        except Exception:
            import sys as _sys
            _sys.path.append(os.path.join(GSA_DIR, 'GroundingDINO'))
            from groundingdino.util.inference import Model as GDModel  # type: ignore

        grounding_dino_model = GDModel(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )
    if sam_predictor is None:
        # import SAM dynamically with local path fallback
        try:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        except Exception:
            import sys as _sys
            _sys.path.append(os.path.join(GSA_DIR, 'segment_anything'))
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore

        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)


def segment_with_sam(
    predictor: Any,
    image_rgb: np.ndarray,
    xyxy: np.ndarray,
) -> np.ndarray:
    """Run SAM with box prompts and return boolean masks [N, H, W]."""
    predictor.set_image(image_rgb)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        index = int(np.argmax(scores))
        result_masks.append(masks[index])
    return np.array(result_masks)


def encode_mask_png(mask: np.ndarray) -> str:
    """Encode a single-channel uint8 mask to PNG and then base64 string."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    success, buf = cv2.imencode('.png', mask)
    if not success:
        raise RuntimeError('Failed to encode mask to PNG')
    return base64.b64encode(buf.tobytes()).decode('utf-8')


class PredictResponse(BaseModel):
    image_width: int
    image_height: int
    detections: List[dict]


app = FastAPI(title='Grounded-SAM API', version='1.0')


@app.on_event('startup')
def startup_event():
    _lazy_init_models()


@app.post('/predict', response_model=PredictResponse)
async def predict(
    image: UploadFile = File(..., description='RGB image file'),
    class_names: str = Form(..., description='JSON array of class names, e.g. ["person","dog"]'),
    box_threshold: float = Form(DEFAULT_BOX_THRESHOLD),
    text_threshold: float = Form(DEFAULT_TEXT_THRESHOLD),
    nms_threshold: float = Form(DEFAULT_NMS_THRESHOLD),
    return_masks: bool = Form(True, description='Whether to return base64-encoded PNG masks'),
):
    """
    Perform detection+segmentation for the given classes.

    Inputs:
    - image: uploaded RGB image (any common format)
    - class_names: JSON string list of object names
    - thresholds: optional override

    Returns:
    - JSON with detections, each item includes class_name, confidence, bbox (xyxy),
      and optional mask (base64 PNG) and index.
    """
    try:
        try:
            classes: List[str] = json.loads(class_names)
            if not isinstance(classes, list) or not all(isinstance(x, str) for x in classes):
                raise ValueError
        except Exception:
            # Fallback: allow comma/semicolon/pipe/newline separated string
            sep_candidates = [',', ';', '|', '\n']
            candidates: List[str] = []
            for sep in sep_candidates:
                if sep in class_names:
                    candidates = [c.strip() for c in class_names.split(sep)]
                    break
            if not candidates:
                candidates = [class_names.strip()]
            classes = [c for c in candidates if c]
            if not classes:
                raise HTTPException(status_code=400, detail='class_names must be JSON array or comma-separated names')

        # Read image bytes -> numpy BGR
        data = await image.read()
        file_bytes = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail='Failed to decode image')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # Run detection with GroundingDINO
        assert grounding_dino_model is not None, "GroundingDINO model was not initialized"
        det = grounding_dino_model.predict_with_classes(
            image=bgr,  # GroundingDINO expects BGR (OpenCV)
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # NMS
        if len(det.xyxy) > 0:
            keep = torchvision.ops.nms(
                torch.from_numpy(det.xyxy),
                torch.from_numpy(det.confidence),
                float(nms_threshold),
            ).cpu().numpy().tolist()

            det.xyxy = det.xyxy[keep]
            det.confidence = det.confidence[keep]
            det.class_id = det.class_id[keep]

        # Masks via SAM
        masks_np: np.ndarray = np.zeros((0, H, W), dtype=bool)
        if len(det.xyxy) > 0:
            masks_np = segment_with_sam(sam_predictor, rgb, det.xyxy)

        class_count = {c: 0 for c in classes}
        results = []
        for i in range(len(det.xyxy)):
            cid = int(det.class_id[i])
            cname = classes[cid]
            bbox = det.xyxy[i].tolist()
            conf = float(det.confidence[i])
            mask_bool = masks_np[i] if i < len(masks_np) else None
            mask_u8 = (mask_bool.astype(np.uint8) * 255) if mask_bool is not None else None

            basename = f"{cname}_{class_count[cname]}"
            class_count[cname] += 1

            item = {
                'index': i,
                'basename': basename,
                'class_name': cname,
                'confidence': conf,
                'bounding_box_xyxy': bbox,
            }
            if return_masks and mask_u8 is not None:
                item['mask_png_base64'] = encode_mask_png(mask_u8)

            results.append(item)

        return JSONResponse(
            content={
                'image_width': W,
                'image_height': H,
                'detections': results,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/healthz')
def healthz():
    return {'status': 'ok', 'device': str(DEVICE)}


# Entry point for `python api_server.py`
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8050, reload=False)
