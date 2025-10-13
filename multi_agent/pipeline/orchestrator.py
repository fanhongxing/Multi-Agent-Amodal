from typing import Optional, Dict, Any, List
import os
import json
from PIL import Image, ImageChops
from ..adapters.seg_adapter import SegmentationAdapter
from ..adapters.inpaint_adapter import InpaintAdapter
from ..adapters.gpt_adapter import GPTAdapter
from ..utils.mask_adapter import binary, invert, dilate, erode, combine_with, cutout_with_white, pad_by_edges_with_mask, safe_save
from ..utils.mask_adapter import directional_extend_mask


class Orchestrator:
    """
    End-to-end single-output pipeline.

    Steps:
    1) Ask GPT for occluders and inpaint prompt (unless provided)
    2) Segment target and occluders; merge masks
    3) Boundary analysis to decide padding and inward-grow
    4) Build inpaint mask and call inpainting
    5) Save artifacts into DEBUG_DIR (env) / results/<seg_text> (default)
    """

    def __init__(
        self,
        seg: Optional[SegmentationAdapter] = None,
        ip: Optional[InpaintAdapter] = None,
        gpt: Optional[GPTAdapter] = None,
    ) -> None:
        self.seg = seg or SegmentationAdapter()
        self.ip = ip or InpaintAdapter()
        self.gpt = gpt or GPTAdapter()

    def run(
        self,
        image: Image.Image,
        seg_text: str,
        prompt: Optional[str] = None,
        mask_thr: int = 128,
        invert_mask: bool = False,
        dilate_k: int = 0,
        dilate_iters: int = 1,
        boundary_mode: str = "boundary_bbox",
        bbox_json_path: Optional[str] = None,
        occluded_object: Optional[str] = None,
        extended_pad_dilate_px: Optional[int] = None,
        edge_grow_ratio: Optional[float] = None,
        visible_target_erode_px: Optional[int] = 5,
        debug_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Debug dirs (default to results/<seg_text>)
        if debug_dir is None:
            # If DEBUG_DIR is provided in env, honor it; otherwise use results/<seg-text>
            env_dir = os.getenv("DEBUG_DIR")
            if env_dir:
                debug_dir = env_dir
            else:
                # sanitize seg_text to safe folder name
                try:
                    import re
                    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", (seg_text or "").strip().lower())
                    if not safe_name:
                        safe_name = "default"
                except Exception:
                    safe_name = (seg_text or "default")
                debug_dir = os.path.join("results", safe_name)
        os.makedirs(debug_dir, exist_ok=True)
        initial_dir = os.path.join(debug_dir, "initial")
        os.makedirs(initial_dir, exist_ok=True)

        # Persist basic run params
        run_params = {
            "seg_text": seg_text,
            "mask_thr": mask_thr,
            "invert_mask": invert_mask,
            "dilate_k": dilate_k,
            "dilate_iters": dilate_iters,
            "boundary_mode": boundary_mode,
            "occluded_object": occluded_object or seg_text,
        }
        if extended_pad_dilate_px is not None:
            run_params["extended_pad_dilate_px"] = extended_pad_dilate_px
        if edge_grow_ratio is not None:
            run_params["edge_grow_ratio"] = edge_grow_ratio
        if visible_target_erode_px is not None:
            run_params["visible_target_erode_px"] = visible_target_erode_px
        with open(os.path.join(debug_dir, "run_params.json"), "w", encoding="utf-8") as f:
            json.dump(run_params, f, ensure_ascii=False, indent=2)

        # 1) GPT: occluding list and object prompt
        occluding_names = self.gpt.gen_inpaint_prompt_from_image(image, seg_text, prompt_type="occluding_object") or "[]"
        with open(os.path.join(initial_dir, "occluding_objects.txt"), "w", encoding="utf-8") as f:
            f.write(str(occluding_names))

        desc = prompt or self.gpt.gen_inpaint_prompt_from_image(image, seg_text, prompt_type="prompt") or ""
        tail = "pure white background."
        if desc.strip():
            if not desc.strip().lower().endswith(tail):
                if desc.rstrip()[-1:] not in ".!?":
                    desc = desc.rstrip() + "."
                desc = desc + " " + tail
        with open(os.path.join(initial_dir, "object_description.txt"), "w", encoding="utf-8") as f:
            f.write(str(desc))

        # Parse occluders text -> list
        class_names: List[str] = [seg_text]
        raw_txt = str(occluding_names)

        def _extract_bracket_list(s: str) -> str:
            starts = [i for i, ch in enumerate(s) if ch == "["]
            for st in starts:
                depth = 0
                for i in range(st, len(s)):
                    if s[i] == "[":
                        depth += 1
                    elif s[i] == "]":
                        depth -= 1
                        if depth == 0:
                            return s[st : i + 1]
            return s

        candidate = _extract_bracket_list(raw_txt)
        occluders_raw: List[str] = []
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                occluders_raw = [str(x) for x in parsed]
        except Exception:
            tmp = candidate.strip().strip("[]").strip()
            if tmp:
                import re
                parts = re.split(r"[,，、;；\n\r\t•·\-–—]+", tmp)
                occluders_raw = [p.strip().strip('\"\'') for p in parts if p.strip()]

        target_key = (occluded_object or seg_text).lower()
        NEG = {"none", "no", "no occlusion", "no occlusions", "n/a", "na", "null", "nothing"}
        seen = set()
        occluders_final: List[str] = []
        for name in occluders_raw:
            n = name.strip().strip(".")
            if not n:
                continue
            nl = n.lower()
            if nl in NEG or nl == target_key:
                continue
            if nl not in seen:
                seen.add(nl)
                occluders_final.append(n)

        with open(os.path.join(initial_dir, "occluding_objects_parsed.json"), "w", encoding="utf-8") as f:
            json.dump(occluders_raw, f, ensure_ascii=False, indent=2)
        with open(os.path.join(initial_dir, "occluding_objects_postfilter.json"), "w", encoding="utf-8") as f:
            json.dump(occluders_final, f, ensure_ascii=False, indent=2)

        class_names += occluders_final

        # 2) Segmentation and mask merge
        from ..utils.mask_adapter import merge_masks

        def _collect(payload: Dict[str, Any], all_names: List[str]):
            dets = payload.get("detections") or []
            masks_target: List[Image.Image] = []
            masks_occ: List[Image.Image] = []
            bbox_t = None
            tkey = target_key
            occ_keys = [n.lower() for n in all_names if n != seg_text]
            for d in dets:
                try:
                    cls = str(d.get("class_name", "")).lower()
                except Exception:
                    cls = ""
                m = d.get("mask") or d.get("mask_base64") or d.get("mask_png_base64") or d.get("mask_png")
                mask_img = None
                if isinstance(m, str) and m:
                    import base64, io
                    if m.startswith("data:"):
                        m = m.split(",", 1)[1]
                    try:
                        from PIL import Image as PILImage
                        mask_img = PILImage.open(io.BytesIO(base64.b64decode(m))).convert("L")
                    except Exception:
                        mask_img = None
                if cls == tkey and bbox_t is None:
                    bbox_t = d.get("bounding_box_xyxy")
                if mask_img is not None:
                    if cls == tkey:
                        masks_target.append(mask_img)
                    elif cls in occ_keys:
                        masks_occ.append(mask_img)
            return masks_target, masks_occ, dets, bbox_t

        w, h = image.size
        target_mask = None
        occluding_mask = None
        bbox_target = None
        attempts: List[Dict[str, Any]] = []

        res_joint = self.seg.segment(image, text=seg_text, extra={"class_names": class_names, "return_masks": True})
        payload_joint = res_joint.get("payload") or {}
        with open(os.path.join(initial_dir, "detections.json"), "w", encoding="utf-8") as f:
            json.dump(payload_joint, f, ensure_ascii=False, indent=2)
        mt, mo, dets, bbox_t = _collect(payload_joint, class_names)
        attempts.append({
            "attempt": 1,
            "mode": "joint",
            "detections_count": len(dets),
            "target_masks_count": len(mt),
            "occluder_masks_count": len(mo),
            "success": bool(mt),
        })
        if mt:
            target_mask = merge_masks(mt)
            occluding_mask = merge_masks(mo) if mo else Image.new("L", image.size, 0)
            bbox_target = bbox_t
        else:
            res_t = self.seg.segment(image, text=seg_text, extra={"class_names": [seg_text], "return_masks": True})
            payload_t = res_t.get("payload") or {}
            with open(os.path.join(initial_dir, "detections_fallback_target.json"), "w", encoding="utf-8") as f:
                json.dump(payload_t, f, ensure_ascii=False, indent=2)
            mt2, _, dets_t, bbox_t2 = _collect(payload_t, [seg_text])
            attempts.append({
                "attempt": 2,
                "mode": "target_only",
                "detections_count": len(dets_t),
                "target_masks_count": len(mt2),
                "occluder_masks_count": 0,
                "success": bool(mt2),
            })
            if mt2:
                target_mask = merge_masks(mt2)
                occluding_mask = Image.new("L", image.size, 0)
                bbox_target = bbox_t2
            else:
                occ_list = [c for c in class_names if c != seg_text]
                if occ_list:
                    res_o = self.seg.segment(image, text=seg_text, extra={"class_names": occ_list, "return_masks": True})
                    payload_o = res_o.get("payload") or {}
                    with open(os.path.join(initial_dir, "detections_fallback_occluders.json"), "w", encoding="utf-8") as f:
                        json.dump(payload_o, f, ensure_ascii=False, indent=2)
                    _, mo2, dets_o, _ = _collect(payload_o, class_names)
                    attempts.append({
                        "attempt": 3,
                        "mode": "occluders_only",
                        "detections_count": len(dets_o),
                        "target_masks_count": 0,
                        "occluder_masks_count": len(mo2),
                        "success": bool(mo2),
                    })
                    occluding_mask = merge_masks(mo2) if mo2 else Image.new("L", image.size, 0)
                # If still missing target mask, bail out
                if target_mask is None:
                    raise RuntimeError("No target mask from segmentation after fallbacks")

        try:
            with open(os.path.join(initial_dir, "segmentation_attempts.json"), "w", encoding="utf-8") as f:
                json.dump(attempts, f, ensure_ascii=False, indent=2)
            # also update run_params
            rp_path = os.path.join(debug_dir, "run_params.json")
            rp = {}
            if os.path.exists(rp_path):
                try:
                    rp = json.load(open(rp_path, "r", encoding="utf-8"))
                except Exception:
                    rp = {}
            rp["segmentation_attempts"] = len(attempts)
            json.dump(rp, open(rp_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Save masks/overlays
        safe_save(target_mask, os.path.join(initial_dir, "target_mask.png"))
        safe_save(occluding_mask, os.path.join(initial_dir, "occluding_mask.png"))
        safe_save(combine_with(image, target_mask), os.path.join(initial_dir, "target_overlay.png"))
        safe_save(combine_with(image, occluding_mask), os.path.join(initial_dir, "occluding_overlay.png"))

        # Target cutout
        target_cut = cutout_with_white(image, target_mask)
        safe_save(target_cut, os.path.join(initial_dir, "target_cutout_white.png"))

        # Save bbox json for boundary_bbox if available
        bbox_json_path_out = os.path.join(initial_dir, "bbox_info.json")
        if bbox_target is not None:
            info = {f"{target_key}_0": {"bounding_box_xyxy": bbox_target, "image_width": w, "image_height": h}}
            with open(bbox_json_path_out, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

        # 3) Boundary analysis -> padding plan
        prompt_name = "boundary_bbox" if boundary_mode == "boundary_bbox" else "boundary"
        bb = self.gpt.gen_inpaint_prompt_from_image(
            image,
            seg_text,
            prompt_type=prompt_name,
            bbox_json_path=(bbox_json_path or (bbox_json_path_out if bbox_target is not None else None)),
            occluded_object=target_key,
        )
        with open(os.path.join(initial_dir, "boundary_bbox.json.txt"), "w", encoding="utf-8") as f:
            f.write(str(bb))

        def _extract_json_object(s: str) -> dict:
            s = (s or "").strip()
            if s.startswith("```"):
                try:
                    first = s.find("```")
                    if first != -1:
                        rest = s[first + 3 :]
                        nl = rest.find("\n")
                        if nl != -1:
                            rest = rest[nl + 1 :]
                        end = rest.find("```")
                        if end != -1:
                            s = rest[:end]
                except Exception:
                    pass
            # Try to locate a top-level JSON object
            start_idxs = [i for i, ch in enumerate(s) if ch == "{"]
            for st in start_idxs:
                depth = 0
                for i in range(st, len(s)):
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                        if depth == 0:
                            frag = s[st : i + 1]
                            try:
                                return json.loads(frag)
                            except Exception:
                                break
            try:
                return json.loads(s)
            except Exception:
                return {}

        obj = _extract_json_object(str(bb))
        pad_dirs: List[str] = []
        pad_amount: float = 0.0
        if isinstance(obj, dict):
            dirs = obj.get("extension_direction") or obj.get("directions") or []
            amt = obj.get("extension_amount") or obj.get("amount") or 0.0
            if isinstance(dirs, str):
                dirs = [dirs]
            norm = []
            for d in dirs:
                dl = str(d).strip().lower()
                if dl in {"left", "right", "top", "bottom"}:
                    norm.append(dl)
            pad_dirs = norm
            if isinstance(amt, (int, float)):
                pad_amount = float(amt)
            elif isinstance(amt, str):
                t = amt.strip().lower()
                if t.endswith("%"):
                    try:
                        pad_amount = float(t[:-1]) / 100.0
                    except Exception:
                        pad_amount = 0.0
                else:
                    try:
                        pad_amount = float(t)
                    except Exception:
                        pad_amount = 0.0
        try:
            with open(os.path.join(initial_dir, "boundary_bbox_parsed.json"), "w", encoding="utf-8") as f:
                json.dump({"extension_direction": pad_dirs, "extension_amount": pad_amount}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Compute extended_pad_dilate_px from ratio if not given
        if edge_grow_ratio is not None and (extended_pad_dilate_px is None or extended_pad_dilate_px == 0):
            if pad_dirs and pad_amount > 0:
                w0, h0 = image.size
                horiz = int(w0 * pad_amount) if any(d in pad_dirs for d in ["left", "right"]) else 0
                vert = int(h0 * pad_amount) if any(d in pad_dirs for d in ["top", "bottom"]) else 0
                base_extent = max(horiz, vert, 0)
                px = int(round(base_extent * float(edge_grow_ratio)))
                extended_pad_dilate_px = max(0, px)
                try:
                    rp_path = os.path.join(debug_dir, "run_params.json")
                    rp = json.load(open(rp_path, "r", encoding="utf-8"))
                except Exception:
                    rp = {}
                rp["edge_grow_ratio"] = edge_grow_ratio
                rp["extended_pad_dilate_px"] = extended_pad_dilate_px
                try:
                    json.dump(rp, open(rp_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                except Exception:
                    pass

        # visible target erosion kernel
        try:
            if visible_target_erode_px is not None and visible_target_erode_px >= 0:
                ki = 2 * int(visible_target_erode_px) + 1
            else:
                ki = 3
            if ki < 3:
                ki = 3
            if ki % 2 == 0:
                ki += 1
            visible_target_erode_k = ki
        except Exception:
            visible_target_erode_k = 3

        # 4) Apply padding to target cut, build pad_mask
        pad_mask = None
        pad_mask_raw = None
        pad_offsets = (0, 0, 0, 0)
        if pad_dirs and pad_amount > 0:
            target_cut, pad_mask, pad_offsets = pad_by_edges_with_mask(
                target_cut, pad_dirs, pad_amount, fill=(255, 255, 255)
            )
            pad_mask_raw = pad_mask.copy() if pad_mask is not None else None
            if extended_pad_dilate_px and pad_mask is not None:
                pad_mask = directional_extend_mask(
                    pad_mask, pad_dirs, target_mask, grow_px=int(extended_pad_dilate_px or 0), offsets=pad_offsets
                )
            safe_save(target_cut, os.path.join(initial_dir, "target_cutout_white_padded.png"))
            if pad_mask is not None:
                safe_save(pad_mask, os.path.join(initial_dir, "boundary_pad_mask.png"))

        # Build occluder inpaint base mask
        occ_pad_mask = None
        occ_mask_pre = binary(occluding_mask, thr=mask_thr)
        occ_mask_bin = occ_mask_pre.copy()
        if invert_mask:
            occ_mask_bin = invert(occ_mask_bin)
        if dilate_k and dilate_k > 0:
            occ_mask_bin = dilate(occ_mask_bin, k=dilate_k, iters=max(1, int(dilate_iters)))
        occ_mask_post = occ_mask_bin.copy()
        safe_save(occ_mask_bin, os.path.join(initial_dir, "occluding_mask_bin.png"))
        if pad_dirs and pad_amount > 0:
            occ_padded, occ_pad_mask, _ = pad_by_edges_with_mask(occ_mask_bin, pad_dirs, pad_amount, fill=0, mask_fill=255)
            safe_save(occ_pad_mask, os.path.join(initial_dir, "occ_pad_mask.png"))
            occ_mask_expanded = ImageChops.lighter(occ_padded, occ_pad_mask)
            if pad_mask is not None:
                pm_bin = binary(pad_mask, thr=1)
                occ_mask_expanded = ImageChops.lighter(occ_mask_expanded, pm_bin)
            occ_mask_bin = occ_mask_expanded
        safe_save(occ_mask_bin, os.path.join(initial_dir, "occluding_mask_padded.png"))

        # Visual overlay for regions
        try:
            base = target_cut.convert("RGB")
            W, H = base.size
            origW, origH = image.size
            left_off, top_off, _, _ = pad_offsets if pad_offsets else (0, 0, 0, 0)

            from PIL import Image as PILImage

            def place_mask(m: Image.Image | None) -> Image.Image:
                if m is None:
                    return PILImage.new("L", (W, H), 0)
                if m.size == (W, H):
                    return m
                if (W, H) != (origW, origH) and m.size == (origW, origH):
                    canvas = PILImage.new("L", (W, H), 0)
                    canvas.paste(m, (left_off, top_off))
                    return canvas
                canvas = PILImage.new("L", (W, H), 0)
                canvas.paste(m, (0, 0))
                return canvas

            occ_pre_canvas = place_mask(occ_mask_pre)
            occ_post_canvas = place_mask(occ_mask_post)
            pad_edge_canvas = place_mask(occ_pad_mask if (pad_dirs and pad_amount > 0) else None)
            tgt_pad_raw_canvas = place_mask(pad_mask_raw)
            tgt_pad_dil_canvas = place_mask(pad_mask)

            import numpy as np

            arr_pre = np.array(occ_pre_canvas) > 0
            arr_post = np.array(occ_post_canvas) > 0
            arr_pad_edge = np.array(pad_edge_canvas) > 0
            arr_tgt_pad_raw = np.array(tgt_pad_raw_canvas) > 0
            arr_tgt_pad_dil = np.array(tgt_pad_dil_canvas) > 0

            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            px = overlay.load()
            for y in range(H):
                for x in range(W):
                    if arr_post[y, x] and not arr_pre[y, x]:
                        px[x, y] = (255, 165, 0, 140)
                    elif arr_pre[y, x]:
                        px[x, y] = (255, 0, 0, 140)
                    elif arr_pad_edge[y, x]:
                        px[x, y] = (0, 255, 255, 140)
                    elif arr_tgt_pad_dil[y, x] and not arr_tgt_pad_raw[y, x]:
                        px[x, y] = (255, 0, 255, 140)
                    elif arr_tgt_pad_raw[y, x]:
                        px[x, y] = (0, 255, 0, 140)

            final_vis = base.convert("RGBA")
            final_vis.alpha_composite(overlay)
            safe_save(final_vis.convert("RGB"), os.path.join(initial_dir, "final_regions_overlay.png"))

            legend = {
                "red": "Original occluder region (not dilated)",
                "orange": "Occluder dilation added region",
                "cyan": "Edge extension added region",
                "green": "Target edge extension original region",
                "magenta": "Target edge extension dilation added region",
            }
            with open(os.path.join(initial_dir, "final_regions_overlay_legend.json"), "w", encoding="utf-8") as f:
                json.dump(legend, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 5) Build final inpaint mask and inpaint
        occ_inpaint_mask = occ_mask_bin
        try:
            target_bin_final = binary(target_mask, thr=mask_thr) if target_mask is not None else None
            if target_bin_final is not None:
                cw, ch = occ_mask_bin.size
                tb_final = Image.new("L", (cw, ch), 0)
                if target_bin_final.size == (cw, ch):
                    tb_final = target_bin_final
                else:
                    try:
                        left_off, top_off, _, _ = pad_offsets
                    except Exception:
                        left_off, top_off = 0, 0
                    try:
                        tb_final.paste(target_bin_final, (left_off, top_off))
                    except Exception:
                        try:
                            tb_final.paste(target_bin_final, (0, 0))
                        except Exception:
                            pass
                try:
                    tb_final_eroded = erode(binary(tb_final, thr=1), k=visible_target_erode_k, iters=1)
                except Exception:
                    tb_final_eroded = tb_final
                pm_keep = binary(pad_mask, thr=1) if pad_mask is not None else None
                tb_to_sub_final = tb_final_eroded
                if pm_keep is not None and pm_keep.size != (cw, ch):
                    try:
                        pm_keep = pm_keep.resize((cw, ch), resample=Image.NEAREST)
                    except Exception:
                        pm_keep = None
                if pm_keep is not None:
                    tb_to_sub_final = ImageChops.multiply(tb_final_eroded, invert(pm_keep))
                occ_inpaint_mask = ImageChops.subtract(occ_mask_bin, tb_to_sub_final)
        except Exception:
            pass
        try:
            safe_save(occ_inpaint_mask, os.path.join(debug_dir, "inpaint_mask_final.png"))
        except Exception:
            pass

        final_out = self.ip.inpaint(target_cut, occ_inpaint_mask, desc or "")
        safe_save(final_out, os.path.join(debug_dir, "final_out.png"))

        return {
            "prompt": desc or "",
            "occluding_objects": occluding_names,
            "bbox": bbox_target,
            "target_mask": target_mask,
            "occluding_mask": occ_mask_bin,
            "output": final_out,
        }
