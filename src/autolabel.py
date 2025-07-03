"""
Rough auto‑labeling of downward‑looking traffic images.
"""
import argparse, os, cv2, torch
from mmengine.config import Config
import tempfile, requests, pathlib
from mmdet.apis import init_detector, inference_detector
from utils.paths import ensure_dir
from utils.box import coco2yolo, draw_boxes

# they should be pre-downloaded in setup.py
CFG_FILE = 'pretrained/rtmdet_tiny_8xb32-300e_coco.py'
CKPT_FILE = 'pretrained/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

CLASSES = ['car', 'bus', 'truck']
COCO2YOLO = {2: 0, 5: 1, 7: 2}

def fetch_config(path_or_url: str) -> str:
     if path_or_url.startswith(('http://', 'https://')):
         r = requests.get(path_or_url, timeout=30)
         r.raise_for_status()
         tmp = tempfile.NamedTemporaryFile('w+', suffix='.py', delete=False)
         tmp.write(r.text)
         tmp.close()
         return tmp.name
     return path_or_url

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--score_thr', type=float, default=0.25)
    cfg = parser.parse_args()

    model = init_detector(CFG_FILE, CKPT_FILE, device='cuda')

    classes_path = os.path.join(cfg.out_dir, 'classes.txt')
    if not os.path.exists(classes_path):
        with open(classes_path, 'w') as f:
            f.write('\n'.join(CLASSES))
            print(f'classes.txt written → {classes_path}')

    img_names = [n for n in os.listdir(cfg.img_dir) if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for name in img_names:
        full = os.path.join(cfg.img_dir, name)
        result = inference_detector(model, full)
        dets = []
        bboxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores
        labels = result.pred_instances.labels

        yolo_lines = []
        for box, score, lbl in zip(bboxes, scores, labels):
            if score < cfg.score_thr:
                continue
            if int(lbl) not in COCO2YOLO:
                continue
            dets.append({
                'bbox': box.cpu().numpy().tolist(),
                'score': float(score),
                'cls': int(lbl)
            })

            h, w = cv2.imread(full).shape[:2]

            # YOLO-format
            x1, y1, x2, y2 = box.tolist()
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            class_id = COCO2YOLO[int(lbl)]
            yolo_lines.append(f'{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}')

        with open(os.path.join(cfg.out_dir, os.path.splitext(name)[0] + '.txt'), 'w') as f:
            f.writelines('\n'.join(yolo_lines))

        # preview
        prev_dir = ensure_dir(os.path.join(cfg.out_dir, '..', 'preview'))
        prev_img = draw_boxes(full, dets, CLASSES, COCO2YOLO)
        cv2.imwrite(os.path.join(prev_dir, name), prev_img)
        print(f'[{name}] → {len(dets)} boxes')

if __name__ == '__main__':
    main()
