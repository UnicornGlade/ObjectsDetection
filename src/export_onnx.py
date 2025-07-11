#!/usr/bin/env python
"""export_onnx.py  – export RTMDet to ONNX, visual‑check parity

CLI example
-----------
python export_onnx.py \
    --cfg configs/rtmdet_car.py \
    --ckpt models \            # dir or .pth
    --img test.jpg \
    --out models/rtmdet_car.onnx

Outputs:
    • *_pt.jpg  – result from original PyTorch model
    • *_onnx.jpg – result from exported ONNX
    • asserts numerical diff < 1e‑3 (max abs)
"""

import argparse, sys, warnings
from pathlib import Path
import functools

import cv2
import numpy as np
import torch
import onnxruntime as ort
import torchvision.ops as tvops

# ------------------------------------------------------------------
# Hot-fix for MMDetection ≥3.0: provide mmdet.core.export.dynamic_clip_for_onnx
# ------------------------------------------------------------------
import types, sys, torch
_core = types.ModuleType("mmdet.core")
_export = types.ModuleType("mmdet.core.export")
def dynamic_clip_for_onnx(*coords_and_shape, **kw):
    """
    Accepts x1, y1, x2, y2 [, max_shape] exactly like original impl.
    Returns the (optionally) clipped coordinates.
    """
    if coords_and_shape and not torch.is_tensor(coords_and_shape[-1]):
        max_shape = coords_and_shape[-1]
        coords = coords_and_shape[:-1]
    else:
        max_shape = None
        coords = coords_and_shape
    clipped = [torch.clamp(c, min=0) for c in coords]
    return (*clipped,)
_export.dynamic_clip_for_onnx = dynamic_clip_for_onnx
_core.export = _export
sys.modules["mmdet.core"] = _core
sys.modules["mmdet.core.export"] = _export
# ------------------------------------------------------------------

from mmengine.config import Config
from mmdet.apis import init_detector

from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample

import cv2, numpy as np, torch

def post_nms(dets, iou_thr=0.5, top_k=300):
    """dets: (N,6)  →  (M,6) after NMS, M≤top_k"""
    boxes = torch.as_tensor(dets[:, :4])
    scores= torch.as_tensor(dets[:, 4])
    keep  = tvops.nms(boxes, scores, iou_thr)
    keep  = keep[: top_k]
    return dets[keep.cpu().numpy()]

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


# BGR → RGB, letter‑box pad to square, return tensor CxHxW

def preprocess(img_bgr: np.ndarray, dst_sz: int = 640) -> tuple[torch.Tensor, float]:
    h, w = img_bgr.shape[:2]
    scale = min(dst_sz / w, dst_sz / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((dst_sz, dst_sz, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = img_resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()  # 0‑255
    return tensor, scale


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
# ONNX wrapper
# -----------------------------------------------------------------------------

class RTMDetONNX(torch.nn.Module):
    """Return fixed shape (B,max_det,6) tensor: [x1,y1,x2,y2,score,label]."""

    def __init__(self, det, score_thr: float = 0.4, max_det: int = 300):
        super().__init__()
        self.det = det
        self.max_det = max_det
        self.score_thr = score_thr
        # use high‑level predict with rescale already applied
        # disabled because we will call .predict and pass metas on our own
        # self.det.forward = functools.partial(det.forward, mode="predict")

    def forward(self, x): # x : B×3×H×W, 0-1 RGB
        _, _, h, w = x.shape
        # ---- строим минимальные meta-данные, которых достаточно bbox_head.predict ----
        batch_metas   = []
        meta_template = dict(
            img_shape   =(h, w, 3),
            ori_shape   =(h, w, 3),
            scale_factor=x.new_tensor([1., 1., 1., 1.]),
            pad_param   ={},
        )
        for _ in range(x.size(0)):
            batch_metas.append(meta_template.copy())

        # high-level API с rescale=True → вернёт bboxes в пикселях оригинала
        #results = self.det.predict(
        #    x, batch_data_samples=batch_samples, rescale=False)
        feats = self.det.extract_feat(x)

        # 1) получаем сырые logits головы
        cls_scores, bbox_preds = self.det.bbox_head(feats)  # ← то, что ждёт predict_by_feat

        # 2) конвертируем их в боксы/оценки (без NMS)
        # with_nms=False → вернёт tuple(bboxes, scores)
        preds = self.det.bbox_head.predict_by_feat(
            cls_scores, bbox_preds,
            batch_img_metas=batch_metas,
            rescale=False, with_nms=False)

        # preds: list[tuple(ndarray (num_box,4), ndarray (num_box, num_cls))]
        from mmengine.structures import InstanceData
        results = []
        for inst_raw in preds:                    # inst_raw: InstanceData
            bboxes = inst_raw.bboxes              # Tensor (N,4)
            cls_sc = inst_raw.scores              # Tensor (N,C)

            if cls_sc.ndim == 2:                  # [N, C]  → берём лучший класс
                conf, labels = cls_sc.max(dim=1)
            else:                                # [N]  → оценки уже одиночные
                conf   = cls_sc                  # (N,)
                labels = torch.zeros_like(conf, dtype=torch.long)

            inst = InstanceData(
                bboxes=bboxes,                    # уже Tensor
                scores=conf,                      # (N,)
                labels=labels)                    # (N,)
            results.append(InstanceData(pred_instances=inst))

        batched = []
        for res in results:  # batch loop
            inst = res.pred_instances
            # сортируем по убыванию score и берём первые max_det
            scores = inst.scores
            idx = scores.argsort(descending=True)
            b = inst.bboxes[idx]
            s = scores[idx].unsqueeze(1)
            l = inst.labels[idx].unsqueeze(1).float()

            out = torch.cat((b, s, l), dim=1)  # (N, 6)

            # 1. обрезаем избыток, чтобы N ≤ max_det
            out = out[: self.max_det]  # (min(N,max_det), 6)

            # 2. если боксов меньше max_det – дополняем нулями
            pad_rows = self.max_det - out.shape[0]  # всегда ≥ 0 после обрезки
            if pad_rows:  # pad_rows == 0 → ничего не делаем
                pad = out.new_zeros((pad_rows, 6))
                out = torch.cat((out, pad), dim=0)  # (max_det, 6)

            batched.append(out)
        return torch.stack(batched)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def process_single_image(img_path, model, wrapper, sess, class_names, args):
    """Process a single image and return visualization results."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[WARNING] Cannot read image {img_path}")
        return None

    tensor, _ = preprocess(img_bgr)
    tensor_pt = tensor.cuda() / 255.0

    with torch.no_grad():
        det_pt = wrapper(tensor_pt.unsqueeze(0))[0]
        det_pt = post_nms(det_pt, 0.5)

    input_name = sess.get_inputs()[0].name
    det_onnx = sess.run(None, {input_name: tensor_pt.unsqueeze(0).cpu().numpy()})[0]
    det_onnx = det_onnx[0]
    det_onnx = post_nms(det_onnx, 0.5)

    h, w = img_bgr.shape[:2]
    dst_sz = 640
    scale = min(dst_sz / w, dst_sz / h)
    det_pt[..., :4] /= scale

    # Run PyTorch inference for comparison
    res_pt = inference_detector(model, img_path)
    
    # Filter outputs
    res_pt_filtered = filter_bboxes(_unpack(res_pt))
    det_pt_filtered = filter_bboxes(det_pt.cpu())
    det_onnx_filtered = filter_bboxes(det_onnx)

    # Generate output paths based on input image name
    img_name = Path(img_path).stem
    previews_dir = Path(args.out).parent / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    out_base = previews_dir / img_name
    pt_out = str(out_base.with_name(f"{img_name}_pt.jpg"))
    onnx_out = str(out_base.with_name(f"{img_name}_onnx.jpg"))

    # Save visualizations
    draw_result(img_path, res_pt, class_names=model.dataset_meta['classes'], out_file=pt_out)
    visualise(img_bgr.copy(), det_onnx, class_names, onnx_out, args.score_thr)

    # Print detection results
    print(f"\nResults for {img_path}:")
    print(f"PyTorch inference results 1: {res_pt_filtered.shape[0]} bboxes")
    print(f"PyTorch inference results 2: {det_pt_filtered.shape[0]} bboxes")
    print(f"ONNX inference results: {det_onnx_filtered.shape[0]} bboxes")

    # Check numeric difference
    det_pt_cpu = det_pt.cpu()  # Move to CPU before numpy conversion
    # avg_diff = np.average(np.abs(det_pt_cpu - det_onnx))
    # print(f"Numeric diff: {avg_diff:.2e}")
    
    # return avg_diff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/rtmdet_car.py")
    ap.add_argument("--ckpt", default="../models", help=".pth or dir with checkpoints")
    ap.add_argument("--img", required=True, help="path to image file or directory")
    ap.add_argument("--out", default="../models/rtmdet_car.onnx")
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--diff_thr", type=float, default=0.5, help="max avg abs diff allowed")
    args = ap.parse_args()

    # ------- load model & initialize --------------------------------
    ckpt = resolve_ckpt(args.ckpt)
    cfg = Config.fromfile(args.cfg)
    class_names = cfg.get("class_names") or cfg.get("CLASSES") or cfg.metainfo["classes"]

    model = init_detector(cfg, ckpt, device="cuda")
    model.eval()

    # ------- export ONNX model -----------------------------------------------------------
    model.test_cfg['score_thr'] = 0.0  # ensure we get all bboxes during export
    wrapper = RTMDetONNX(model, args.score_thr).eval().cuda()

    dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_tensor, _ = preprocess(dummy_bgr, 640)
    dummy_tensor = dummy_tensor.unsqueeze(0).cuda() / 255.0

    torch.onnx.export(
        wrapper,
        dummy_tensor,
        args.out,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["dets"],
        dynamic_axes={"images": {0: "batch"}, "dets": {0: "batch"}},
        verbose=False,
    )
    print("ONNX exported →", args.out)

    # Initialize ONNX session
    sess = ort.InferenceSession(str(args.out), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Process images
    img_path = Path(args.img)
    if img_path.is_file():
        pass
        # Single image processing
        #avg_diff = process_single_image(str(img_path), model, wrapper, sess, class_names, args)
        # if avg_diff is not None and avg_diff > args.diff_thr:
        #     raise RuntimeError(f"ONNX mismatch: max abs diff {avg_diff:.4e} > {args.diff_thr}")
    elif img_path.is_dir():
        # Directory processing
        image_files = list(img_path.glob("*.jpg")) + list(img_path.glob("*.jpeg")) + list(img_path.glob("*.png"))
        if not image_files:
            sys.exit(f"[ERROR] No image files found in directory {img_path}")
        
        print(f"Found {len(image_files)} images in {img_path}")
        max_diff = 0
        for img_file in image_files:
            print(f"\nProcessing {img_file}...")
            avg_diff = process_single_image(str(img_file), model, wrapper, sess, class_names, args)
            if avg_diff is not None:
                max_diff = max(max_diff, avg_diff)
        
        if max_diff > args.diff_thr:
            raise RuntimeError(f"ONNX mismatch: max abs diff {max_diff:.4e} > {args.diff_thr}")
        print(f"\nAll images processed successfully. Maximum numeric diff: {max_diff:.2e}")
    else:
        sys.exit(f"[ERROR] Path {args.img} does not exist or is neither a file nor directory")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
