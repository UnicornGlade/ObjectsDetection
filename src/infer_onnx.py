"""Pure‑ONNX inference for RT‑DETR."""        
import argparse, cv2, numpy as np, onnxruntime as ort
from utils.box import nms, scale_coords, COLORS

CLS = ('car', 'bus', 'truck')

def preprocess(img, new_size=640):
    h0, w0 = img.shape[:2]
    r = new_size / max(h0, w0)
    resized = cv2.resize(img, (int(w0 * r), int(h0 * r)))
    pad = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    pad[: resized.shape[0], : resized.shape[1]] = resized
    img = pad[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB→CHW
    return img.astype(np.float32) / 255.0, r, (w0, h0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../models/rtdetr_car.onnx')
    ap.add_argument('--img', required=True)
    ap.add_argument('--save', default=None)
    ap.add_argument('--thr', type=float, default=0.3)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider'])
    img0 = cv2.imread(args.img)
    blob, r, (w0, h0) = preprocess(img0)
    preds = sess.run(None, {'images': blob[np.newaxis, ...]})
    logits, boxes = preds
    probs = logits.squeeze()
    boxes = boxes.squeeze()

    # filter + NMS
    scores = probs.max(1)
    cls_ids = probs.argmax(1)
    keep = scores > args.thr
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    keep = nms(boxes, scores, 0.55)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

    # scale back
    boxes = scale_coords(boxes, r, w0, h0)

    # draw
    for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cls_ids):
        cv2.rectangle(img0, (x1, y1), (x2, y2), COLORS[cid], 2)
        cv2.putText(img0, f'{CLS[cid]} {score:.2f}', (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cid], 2)

    if args.save:
        cv2.imwrite(args.save, img0)
    cv2.imshow('result', img0)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
