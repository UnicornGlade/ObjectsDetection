"""RTMDet‑ONNX inference with proper bbox decode
================================================

This script runs **pure‑ONNX** inference for a model exported the way we
wrapped it earlier (raw regression distances + class logits).  It:

* letterbox‑resizes each input image to 640×640 exactly as during training;
* decodes the raw `(l,t,r,b)` distances back to absolute **xyxy** boxes
  (logic equivalent to `distance2bbox` in MMDetection);
* rescales boxes to original image size, applies score‑filter + NMS;
* draws detections and either shows them in a window or saves to `--out-dir`;
* has a NumPy fallback for NMS, so it works even without a compiled
  `torchvision` build.

Usage
-----
```bash
python infer_onnx.py --dir path/to/images \
                     --model rtmdet_car.onnx \
                     --out-dir results       # optional, saves jpgs
```
Add `--no-gui` if you run on a headless machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

import cv2  # type: ignore
import numpy as np
import onnxruntime as ort

# -----------------------------------------------------------------------------
# Constants (adapt if your model differs)
# -----------------------------------------------------------------------------
CLS: Tuple[str, ...] = ("car", "bus", "truck")          # class names in model order
STRIDES: Tuple[int, int, int] = (8, 16, 32)               # 3 heads of RTMDet‑Ti
FM_SHAPES: Tuple[Tuple[int, int], ...] = (
    (80, 80), (40, 40), (20, 20)
)  # H×W per head for 640×640 input
COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 56, 56), (255, 157, 151), (255, 112, 31)
)  # BGR  -> red / orange / ...

# -----------------------------------------------------------------------------
# Pre‑processing helpers
# -----------------------------------------------------------------------------

def preprocess(img: np.ndarray, new_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize with upper‑left letterbox padding.

    Returns
    -------
    blob : np.ndarray
        CHW float32 RGB image, values in [0,1].
    r : float
        Scale factor applied w.r.t. the larger side.
    orig_shape : tuple[int, int]
        (w0, h0) of the original image.
    """
    h0, w0 = img.shape[:2]
    r = new_size / max(h0, w0)

    resized = cv2.resize(img, (int(w0 * r), int(h0 * r)))
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized

    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob, r, (w0, h0)


def scale_coords(boxes: np.ndarray, r: float, w0: int, h0: int) -> np.ndarray:
    """Map boxes from 640×640 letterboxed back to original image size."""
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)
    return boxes


# -----------------------------------------------------------------------------
# Decode raw distances -> absolute XYXY  (main fix)
# -----------------------------------------------------------------------------

def _make_grids() -> Tuple[np.ndarray, np.ndarray]:
    """Pre‑compute grid centres (x,y) and stride for every location."""
    centers: List[np.ndarray] = []
    strides: List[np.ndarray] = []
    for (h, w), s in zip(FM_SHAPES, STRIDES):
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        centers.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        strides.append(np.full((h * w, 1), s, dtype=np.float32))
    return np.concatenate(centers, axis=0).astype(np.float32), np.concatenate(strides, axis=0)


GRID_CENTERS, STRIDE_PER_LOC = _make_grids()  # (8400,2) & (8400,1)


def decode_rtm(boxes: np.ndarray) -> np.ndarray:
    """Decode (l,t,r,b) *distances* to absolute **xyxy** boxes in 640×640."""
    ltrb = boxes.astype(np.float32) * STRIDE_PER_LOC  # broadcast (N,4)*(N,1)
    cxcy = (GRID_CENTERS + 0.5) * STRIDE_PER_LOC      # centre of each anchor

    x1 = cxcy[:, 0] - ltrb[:, 0]
    y1 = cxcy[:, 1] - ltrb[:, 1]
    x2 = cxcy[:, 0] + ltrb[:, 2]
    y2 = cxcy[:, 1] + ltrb[:, 3]
    return np.stack((x1, y1, x2, y2), axis=1)


# -----------------------------------------------------------------------------
# NMS  (torchvision fast‑path + NumPy fallback)
# -----------------------------------------------------------------------------
try:
    import torch  # type: ignore
    from torchvision.ops import nms as _torch_nms  # type: ignore

    def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.55) -> np.ndarray:
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        scores_t = torch.as_tensor(scores, dtype=torch.float32)
        keep = _torch_nms(boxes_t, scores_t, iou_thr)
        return keep.cpu().numpy()

except Exception:  # noqa: BLE001

    def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.55) -> np.ndarray:
        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_thr]
        return np.array(keep, dtype=np.int32)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    ap = argparse.ArgumentParser("RTMDet‑ONNX inference")
    ap.add_argument("--dir", required=True, help="directory with images")
    ap.add_argument("--model", default="../models/rtmdet_car.onnx")
    ap.add_argument("--out-dir", help="save visualisations here (optional)")
    ap.add_argument("--thr", type=float, default=0.3, help="score threshold")
    ap.add_argument("--no-gui", action="store_true", help="disable cv2.imshow")
    args = ap.parse_args()

    # Collect images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths: List[str] = []
    for ext in exts:
        img_paths += [str(p) for p in Path(args.dir).rglob(ext)]
    img_paths.sort()
    if not img_paths:
        sys.exit(f"[ERROR] No images found in {args.dir}")

    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ONNX Runtime session
    providers = [
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]
    try:
        sess = ort.InferenceSession(args.model, providers=[p[0] for p in providers])
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[ERROR] Failed to load ONNX model: {exc}")

    for path in img_paths:
        if "P1000101_1400494888594.JPG" not in path:
            continue

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"[WARN] Can't read {path}")
            continue

        blob, r, (w0, h0) = preprocess(img_bgr)
        logits, raw_boxes = sess.run(None, {"images": blob[np.newaxis]})
        logits = logits.squeeze(0).astype(np.float32)      # (8400, C)
        raw_boxes = raw_boxes.squeeze(0).astype(np.float32)  # (8400,4)

        # Decode distances -> xyxy in letterboxed space
        boxes = decode_rtm(raw_boxes)

        # Convert logits to class + score
        scores = logits.max(1).astype(np.float32)          # (8400,)
        cls_ids = logits.argmax(1).astype(np.int32)

        mask = scores > args.thr
        boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
        if boxes.size == 0:
            continue

        # NMS
        keep = nms(boxes, scores, 0.55)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

        # Rescale back to original image size
        boxes = scale_coords(boxes, r, w0, h0).astype(int)

        # Draw detections
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cls_ids):
            color = COLORS[cid % len(COLORS)]
            color = (0, 0, 255)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 200)
            label = f"{CLS[cid]} {score:.2f}"
            cv2.putText(
                img_bgr,
                label,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                200,
                cv2.LINE_AA,
            )

        fname = Path(path).name
        if args.out_dir:
            out_path = Path(args.out_dir) / fname
            cv2.imwrite(str(out_path), img_bgr)
            print(f"[INFO] Saved {out_path}")

        if not args.no_gui:
            try:
                for i in range(2):
                    img_bgr = cv2.pyrDown(img_bgr)
                cv2.imshow("result", img_bgr)
                if cv2.waitKey(0) & 0xFF == 27:  # ESC to quit early
                    break
            except cv2.error as exc:  # noqa: BLE001
                print("[INFO] cv2.imshow not available, switching to --no-gui:", exc)
                args.no_gui = True

    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
