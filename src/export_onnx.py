#!/usr/bin/env python
"""Export trained RTMDet checkpoint to ONNX."""
import argparse, torch, sys
import functools
from pathlib import Path
from mmengine.config import Config
from mmdet.apis import init_detector

def resolve_ckpt(path_like: str) -> str:
    """
    If `path_like` is an existing file → return as is.
    If it's a directory → read `last_checkpoint` inside it or take
    the highest-numbered epoch_XXX.pth.
    """
    p = Path(path_like)
    if p.is_file():
        return str(p)

    # treat as directory
    last_file = p / "last_checkpoint"
    if last_file.is_file():
        return str((p / last_file.read_text().strip()).resolve())

    ckpts = sorted(p.glob("epoch_*.pth"))
    if not ckpts:
        sys.exit(f"[ERROR] no .pth found in {p}")
    return str(ckpts[-1])           # biggest epoch number

class RTMDetONNX(torch.nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det
        # force raw tensors, no NMS
        self.det.forward = functools.partial(det.forward, mode='tensor')

    def forward(self, x):
        cls, box = self.det(x)           # tuple of 3 + 3 tensors
        # (B, C, H, W)  →  (B, H*W, C)   then concatenate scales
        cls = torch.cat([c.flatten(2).permute(0, 2, 1) for c in cls], dim=1)
        # (B, 4, H, W)  →  (B, H*W, 4)
        box = torch.cat([b.flatten(2).permute(0, 2, 1) for b in box], dim=1)
        return cls, box

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/rtmdet_car.py')
    ap.add_argument(
        "--ckpt", default="../models",
        help="path to .pth file **or** directory with checkpoints",
    )
    ap.add_argument("--out", default="../models/rtmdet_car.onnx")
    args = ap.parse_args()

    ckpt_path = resolve_ckpt(args.ckpt)
    model = init_detector(Config.fromfile(args.cfg), ckpt_path, device="cuda")
    model.eval()

    wrapper = RTMDetONNX(model).eval().cuda()
    dummy = torch.randn(1, 3, 640, 640, device="cuda")

    torch.onnx.export(
        wrapper, dummy, args.out,
        opset_version=12, do_constant_folding=True,
        input_names=['images'],
        output_names=['pred_logits', 'pred_boxes'],
        dynamic_axes={'images': {0: 'batch'},
                      'pred_logits': {0: 'batch'},
                      'pred_boxes': {0: 'batch'}},
        verbose=True
    )
    print("Exported →", args.out)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=['CUDAExecutionProvider'])
    print([o.name for o in sess.get_outputs()])  # → ['pred_logits', 'pred_boxes']

if __name__ == "__main__":
    main()
