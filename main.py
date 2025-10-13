import argparse
from PIL import Image
from multi_agent.pipeline.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--seg-text", dest="seg_text", required=True, help="segmentation prompt, e.g., 'red cup'")
    parser.add_argument("--prompt", default=None, help="inpaint prompt; if absent, GPT will help")
    parser.add_argument("--dry-run", action="store_true", help="simulate adapters when services are missing")
    # segmentation backend selection
    parser.add_argument("--seg-backend", choices=["sam", "xsam"], default=None, help="choose segmentation backend: 'sam' (default) or 'xsam'")
    parser.add_argument("--seg-url", default=None, help="override segmentation server base url; e.g., http://0.0.0.0:8050 for sam, http://0.0.0.0:8052 for xsam")
    # mask options
    parser.add_argument("--mask-thr", type=int, default=128, help="binarize threshold (0-255)")
    parser.add_argument("--invert-mask", action="store_true", help="invert mask before inpainting")
    parser.add_argument("--dilate-k", type=int, default=7, help="MaxFilter kernel size for dilation (odd >=3), 0 to skip")
    parser.add_argument("--dilate-iters", type=int, default=2, help="dilation iterations")
    # edge extension inward grow controls
    parser.add_argument("--edge-grow-ratio", type=float, default=None, help="Edge extension inward-grow ratio (relative to padding extent). For example, 0.15 means add 15% of the extension width inward.")
    parser.add_argument("--edge-grow-px", type=int, default=10, help="Edge extension inward-grow pixels (absolute). Takes precedence over the ratio if provided.")
    # boundary mode selector
    parser.add_argument("--boundary-mode", choices=["boundary", "boundary_bbox"], default="boundary_bbox", help="boundary analysis prompt type")
    parser.add_argument("--bbox-json-path", default=None, help="detections json path for boundary_bbox")
    parser.add_argument("--occluded-object", default=None, help="object name used for boundary_bbox analysis, e.g., 'bird'")
    # visible target erosion before subtracting from occluders
    parser.add_argument("--visible-target-erode-px", dest="visible_target_erode_px", type=int, default=5, help="Shrink target-visible region by N pixels before subtracting from occluder mask (default: 5). Set 0 to disable shrink.")
    args = parser.parse_args()

    # Allow toggling dry-run via env or arg: we recreate adapters accordingly
    from multi_agent.adapters.seg_adapter import SegmentationAdapter
    from multi_agent.adapters.inpaint_adapter import InpaintAdapter
    from multi_agent.adapters.gpt_adapter import GPTAdapter

    seg = SegmentationAdapter(dry_run=args.dry_run, backend=args.seg_backend, base_url=args.seg_url)
    ip = InpaintAdapter(dry_run=args.dry_run)
    gpt = GPTAdapter()

    orch = Orchestrator(seg=seg, ip=ip, gpt=gpt)

    image = Image.open(args.image).convert("RGB")

    # We only know pad_amount after boundary analysis, so we cannot directly compute the ratio here.
    # Strategy: pass-through either the absolute pixels or the ratio into orchestrator via kwargs.
    # If ratio is provided (and px not), orchestrator will compute the final inward-grow pixels after parsing pad_amount.

    extra_kwargs = {}
    if args.edge_grow_ratio is not None and args.edge_grow_ratio < 0:
        args.edge_grow_ratio = 0
    if args.edge_grow_px is not None and args.edge_grow_px < 0:
        args.edge_grow_px = 0
    if args.edge_grow_px is not None:
        extra_kwargs["extended_pad_dilate_px"] = args.edge_grow_px
    elif args.edge_grow_ratio is not None:
    # Record the ratio; the actual pixels will be computed inside orchestrator after boundary parsing
        extra_kwargs["edge_grow_ratio"] = args.edge_grow_ratio

    # pass visible-target erode control
    if args.visible_target_erode_px is not None and args.visible_target_erode_px < 0:
        args.visible_target_erode_px = 0
    extra_kwargs["visible_target_erode_px"] = args.visible_target_erode_px

    result = orch.run(
        image,
        seg_text=args.seg_text,
        prompt=args.prompt,
        mask_thr=args.mask_thr,
        invert_mask=args.invert_mask,
        dilate_k=args.dilate_k,
        dilate_iters=args.dilate_iters,
        boundary_mode=args.boundary_mode,
        bbox_json_path=args.bbox_json_path,
        occluded_object=args.occluded_object,
        **extra_kwargs,
    )

if __name__ == "__main__":
    main()
