import argparse, sys, warnings
from pathlib import Path

from mmengine.config import Config

from mmdet.apis import init_detector, inference_detector

import cv2, numpy as np, torch

# ----------------------------------------------------------------------
# Draw MMDetection result on an image (v2.x *или* v3.x).
# ----------------------------------------------------------------------
def _unpack(result):
    """Convert result to flat (N, 4+score+label) ndarray."""
    if hasattr(result, 'pred_instances'):          # v3: DetDataSample
        inst = result.pred_instances
        b = inst.bboxes.cpu().numpy()
        s = inst.scores.cpu().numpy()[:, None]
        l = inst.labels.cpu().numpy()[:, None]
        return np.hstack([b, s, l])
    else:                                          # v2: list[ndarray] per class
        if isinstance(result, tuple):              # masks / det
            result = result[0]
        outs = []
        for cls, dets in enumerate(result):
            if dets.size:
                lbl = np.full((dets.shape[0], 1), cls, dtype=np.float32)
                outs.append(np.hstack([dets, lbl]))
        return np.vstack(outs) if outs else np.empty((0, 6), np.float32)

def draw_result(img_path, result, class_names, out_file, score_thr=0.3):
    mat = cv2.imread(img_path)
    dets = _unpack(result)
    for x1, y1, x2, y2, score, label in dets:
        if score < score_thr:
            continue
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(mat, p1, p2, (0, 255, 0), 2)
        txt = f'{class_names[int(label)]} {score:.2f}'
        cv2.putText(mat, txt, (p1[0], max(p1[1] - 2, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(out_file, mat)

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def resolve_ckpt(path_like: str) -> str:
    """Return .pth if path_like is file or top/latest in dir."""
    p = Path(path_like)
    if p.is_file():
        return str(p)
    last = p / "last_checkpoint"
    if last.is_file():
        return str((p / last.read_text().strip()).resolve())
    ckpts = sorted(p.glob("epoch_*.pth"))
    if not ckpts:
        sys.exit(f"[ERROR] no .pth found in {p}")
    return str(ckpts[-1])


def visualise(img_bgr: np.ndarray, dets: np.ndarray, names: list[str], out: str, thr: float = 0.4):
    """Draw boxes [x1,y1,x2,y2,score,label]."""
    h, w = img_bgr.shape[:2]
    dst_sz = 640
    scale = min(dst_sz / w, dst_sz / h)
    dets[:, :4] /= scale # TODO make division on copy of dets (to not modify arguments)
    for x1, y1, x2, y2, s, cls in dets:
        if s < thr:
            continue
        color = (0, 255, 0)
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img_bgr, f"{names[int(cls)]}:{s:.2f}", (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    cv2.imwrite(out, img_bgr)
    print("saved", out)


def filter_bboxes(dets: np.ndarray, thr: float = 0.4):
    filtered_dets = []
    for x1, y1, x2, y2, s, cls in dets:
        if s < thr:
            continue
        filtered_dets.append((x1, y1, x2, y2, s, cls))
    return np.array(filtered_dets)


def print_bboxes(img_bgr: np.ndarray, dets: np.ndarray):
    h, w = img_bgr.shape[:2]
    dst_sz = 640
    scale = min(dst_sz / w, dst_sz / h)
    dets[:, :4] /= scale  # TODO make division on copy of dets (to not modify arguments)
    ntotal = 0
    print("{} - number of bboxes (ignoring their scores/confidences)".format(len(dets)))
    for x1, y1, x2, y2, s, cls in dets:
        print(s, x1, y1, x2, y2)
        ntotal += 1
    print(ntotal)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/rtmdet_car.py")
    ap.add_argument("--ckpt", default="../models", help=".pth or dir with checkpoints")
    ap.add_argument("--img", required=True, help="image for sanity check")
    ap.add_argument("--out", default="../models/rtmdet_car.onnx")
    ap.add_argument("--score_thr", type=float, default=0.4)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--diff_thr", type=float, default=0.5, help="max avg abs diff allowed")
    args = ap.parse_args()

    # ------- load model & run native inference --------------------------------
    ckpt = resolve_ckpt(args.ckpt)
    cfg = Config.fromfile(args.cfg)

    model = init_detector(cfg, ckpt, device="cuda")
    model.eval()

    # visualise PyTorch inference
    res_pt = inference_detector(model, args.img)
    draw_result(
        args.img, res_pt,
        class_names=model.dataset_meta['classes'],
        out_file=Path(args.out).with_suffix("").with_name(Path(args.out).stem + "_pt.jpg")
    )

    # filter output bboxes w.r.t. confidence threshold
    res_pt = filter_bboxes(_unpack(res_pt))

    img_bgr = cv2.imread(args.img)
    print("PyTorch inference results 1: {} bboxes:".format(res_pt.shape))
    print_bboxes(img_bgr, res_pt)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
