import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image
from multi_agent.pipeline.orchestrator import Orchestrator
from multi_agent.adapters.seg_adapter import SegmentationAdapter
from multi_agent.adapters.inpaint_adapter import InpaintAdapter
from multi_agent.adapters.gpt_adapter import GPTAdapter

def run():
    # make a simple image
    img = Image.new("RGB", (256, 256), (200, 200, 200))
    orch = Orchestrator(
        seg=SegmentationAdapter(dry_run=True),
        ip=InpaintAdapter(dry_run=True),
        gpt=GPTAdapter(backend="mock"),
    )
    # simple mode
    out = orch.run(img, seg_text="dog", prompt=None, extended_pad_dilate_px=3)
    assert out["output"].size == (256, 256)
    print("dry-run ok; prompt:", out["prompt"])

    # probabilistic 模式已移除

if __name__ == "__main__":
    run()
