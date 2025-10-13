from typing import Tuple, List
from PIL import Image, ImageOps, ImageFilter, ImageChops
import numpy as np

# A minimal set of helpers inspired by oh-my-diff/src/utils/mask.py

def ensure_gray(mask: Image.Image) -> Image.Image:
    if mask.mode != "L":
        return mask.convert("L")
    return mask

def invert(mask: Image.Image) -> Image.Image:
    return ImageOps.invert(ensure_gray(mask))

def binary(mask: Image.Image, thr: int = 128) -> Image.Image:
    m = ensure_gray(mask)
    arr = np.array(m)
    arr = (arr >= thr).astype("uint8") * 255
    return Image.fromarray(arr, mode="L")

def dilate(mask: Image.Image, k: int = 3, iters: int = 1) -> Image.Image:
    m = ensure_gray(mask)
    out = m
    # Use MaxFilter as a simple dilation surrogate
    size = max(3, int(k) if k % 2 == 1 else int(k) + 1)
    for _ in range(max(1, iters)):
        out = out.filter(ImageFilter.MaxFilter(size=size))
    return out

def erode(mask: Image.Image, k: int = 3, iters: int = 1) -> Image.Image:
    """Simple morphological erosion using MinFilter. For binary 0/255 masks.
    k: kernel size (odd integer), roughly shrinks boundaries by ~ (k-1)/2 pixels.
    iters: how many times to apply the erosion.
    """
    m = ensure_gray(mask)
    out = m
    size = max(3, int(k) if k % 2 == 1 else int(k) + 1)
    for _ in range(max(1, iters)):
        out = out.filter(ImageFilter.MinFilter(size=size))
    return out

def combine_with(image: Image.Image, mask: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    # Visualize mask on top of image; tolerate different sizes by aligning to mask size
    m = ensure_gray(mask)
    if image.size != m.size:
        base = Image.new("RGB", m.size, (0, 0, 0))
        base.paste(image.convert("RGB"), (0, 0))
        rgb = base
    else:
        rgb = image.convert("RGB").copy()
    overlay = Image.new("RGB", rgb.size, fill)
    rgb.paste(overlay, (0, 0), m)
    return rgb

def merge_masks(masks: List[Image.Image]) -> Image.Image:
    if not masks:
        raise ValueError("masks is empty")
    base = ensure_gray(masks[0])
    for m in masks[1:]:
        base = ImageChops.lighter(base, ensure_gray(m))
    return base

def cutout_with_white(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(rgb, (0, 0), ensure_gray(mask))
    return bg

def pad_by_edges(img: Image.Image, directions: List[str], amount: float, fill=(255, 255, 255)) -> Image.Image:
    """Pad image on given edges by a relative fraction of width/height.
    directions: subset of ["top","bottom","left","right"],
    amount: relative (e.g., 0.1 adds 10% of the corresponding dimension).
    """
    w, h = img.size
    left = int(w * amount) if "left" in directions else 0
    right = int(w * amount) if "right" in directions else 0
    top = int(h * amount) if "top" in directions else 0
    bottom = int(h * amount) if "bottom" in directions else 0
    new_w, new_h = w + left + right, h + top + bottom
    out = Image.new("RGB" if img.mode != "L" else "L", (new_w, new_h), fill)
    out.paste(img, (left, top))
    return out

def safe_save(img: Image.Image, path: str) -> None:
    try:
        img.save(path)
    except Exception:
        # fallback format
        img.convert("RGB").save(path)

def _compute_padding(w: int, h: int, directions: List[str], amount: float):
    left = int(w * amount) if "left" in directions else 0
    right = int(w * amount) if "right" in directions else 0
    top = int(h * amount) if "top" in directions else 0
    bottom = int(h * amount) if "bottom" in directions else 0
    return left, right, top, bottom

def pad_by_edges_with_mask(img: Image.Image, directions: List[str], amount: float, fill=(255, 255, 255), mask_fill: int = 255):
    """Pad image and also return a mask where the padded regions are marked with mask_fill.
    Returns (padded_image, pad_mask_L, (left, top, right, bottom)).
    Offsets define how original image is placed inside padded image.
    """
    w, h = img.size
    left, right, top, bottom = _compute_padding(w, h, directions, amount)
    new_w, new_h = w + left + right, h + top + bottom
    # pad image
    mode = "RGB" if img.mode != "L" else "L"
    out = Image.new(mode, (new_w, new_h), fill)
    out.paste(img, (left, top))
    # build pad mask
    pad_mask = Image.new("L", (new_w, new_h), 0)
    if left > 0:
        Image.Image.paste(pad_mask, Image.new("L", (left, new_h), mask_fill), (0, 0))
    if right > 0:
        Image.Image.paste(pad_mask, Image.new("L", (right, new_h), mask_fill), (new_w - right, 0))
    if top > 0:
        Image.Image.paste(pad_mask, Image.new("L", (new_w, top), mask_fill), (0, 0))
    if bottom > 0:
        Image.Image.paste(pad_mask, Image.new("L", (new_w, bottom), mask_fill), (0, new_h - bottom))
    return out, pad_mask, (left, top, right, bottom)


def directional_extend_mask(
    pad_mask: Image.Image,
    directions: List[str],
    base_mask: Image.Image,
    grow_px: int = 0,
    offsets: Tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Make the padded region intrude into the original image by some pixels along given directions.

    Example need: after adding a 20% top padding, only let that top padded band intrude downwards into the original image by ~grow_px pixels, without changing the canvas size.

    Strategy:
    1. Build a base_canvas with the same size as pad_mask (place base_mask into it according to offsets).
    2. For each direction, accumulate one-way shifts inward (instead of symmetric dilation) to form inward_candidate.
       - top: shift pad_mask down by 1..grow_px
       - bottom: shift up
       - left: shift right
       - right: shift left
    3. inward_band = inward_candidate AND base_canvas AND NOT pad_mask (only pixels that intrude into original image and not in the padded area).
    4. new_mask = pad_mask OR inward_band.
    5. If grow_px <= 0, return pad_mask directly.

    Note: Use directional shifts instead of a square structuring element to avoid unnecessary spreading; union for multiple directions.
    """
    if grow_px is None or grow_px <= 0:
        return pad_mask

    pm = ensure_gray(pad_mask)
    bm = ensure_gray(base_mask)  # Only used to get original image size; not limited to the object mask
    Wp, Hp = pm.size
    Wb, Hb = bm.size

    # Build a boolean mask of the original image rectangle (not relying on object content)
    import numpy as np
    if (Wp, Hp) != (Wb, Hb):
        left = top = 0
        if offsets is not None:
            left, top, _, _ = offsets
        base_rect = np.zeros((Hp, Wp), dtype=bool)
        base_rect[top:top+Hb, left:left+Wb] = True
    else:
        base_rect = np.ones((Hp, Wp), dtype=bool)

    arr_pad = (np.array(pm) > 0)

    inward_candidate = np.zeros_like(arr_pad, dtype=bool)

    g = int(grow_px)
    g = max(1, g)

    # To avoid excessive width from repeated shifts, accumulate one-way shifts directly
    if 'top' in directions:
        # Padding on top; move inward (down)
        for s in range(1, g + 1):
            inward_candidate[s:, :] |= arr_pad[:-s, :]
    if 'bottom' in directions:
        # Move inward (up)
        for s in range(1, g + 1):
            inward_candidate[:-s, :] |= arr_pad[s:, :]
    if 'left' in directions:
        # Move inward (right)
        for s in range(1, g + 1):
            inward_candidate[:, s:] |= arr_pad[:, :-s]
    if 'right' in directions:
        # Move inward (left)
        for s in range(1, g + 1):
            inward_candidate[:, :-s] |= arr_pad[:, s:]

    inward_band = inward_candidate & base_rect & (~arr_pad)
    if not inward_band.any():
        return pm  # No intrusion, return as-is

    final = np.array(pm)
    final[inward_band] = 255
    return Image.fromarray(final, mode="L")
