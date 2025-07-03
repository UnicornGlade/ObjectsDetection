import cv2, numpy as np, torch

COLORS = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]

def coco2yolo(bbox, w, h, cls_name):
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f'{cls_name} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}'

def read_yolo(txt_path):
    boxes, labels = [], []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = parts
            boxes.append([float(xc), float(yc), float(bw), float(bh)])
            labels.append(int(cls))
    return boxes, labels

def alb2yolo(box, lbl):
    return f'{lbl} ' + ' '.join(f'{v:.6f}' for v in box)

def nms(boxes, scores, iou_thr=0.55):
    keep = torch.ops.torchvision.nms(torch.tensor(boxes), torch.tensor(scores), iou_thr)
    return keep.cpu().numpy()

def scale_coords(boxes, r, w0, h0):
    boxes[:, [0, 2]] -= (640 - w0 * r) / 2
    boxes[:, [1, 3]] -= (640 - h0 * r) / 2
    boxes /= r
    boxes = boxes.round().astype(int)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w0)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h0)
    return boxes

def draw_boxes(path, dets, classes, coco2yolo):
    img = cv2.imread(path)
    for d in dets:
        x1, y1, x2, y2 = map(int, d['bbox'])
        cls_name = classes[coco2yolo[d['cls']]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, cls_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img
