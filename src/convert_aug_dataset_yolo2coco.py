#!/usr/bin/env python
"""
Convert YOLO txt dataset (one .txt per image) ➜ COCO JSON.

Example
-------
python convert_aug_dataset_yolo2coco.py \
    --src data/dataset01_aug \
    --dst data/dataset01_aug_coco \
    --train-ratio 0.8 \
    --seed 42
"""
from __future__ import annotations
import argparse, json, random, shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import re
import cv2
from tqdm import tqdm

# ---- your class list --------------------------------------------------------
# YOLO id : name ; COCO ids must start at 1, so we add +1 later.
YOLO_CLASSES = ['car', 'bus', 'truck']

# ------------------------------------------------------------------------------------------------------------------
def yolo_txt_to_boxes(txt: Path, W: int, H: int) -> List[Dict]:
    """Read YOLO-format .txt and return list[{cls_id,x,y,w,h,area}] in pixels."""
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts)
        xc, yc, bw, bh = xc * W, yc * H, bw * W, bh * H
        out.append(dict(
            cls_id=int(cls),
            x=xc - bw / 2, y=yc - bh / 2, w=bw, h=bh,
            area=bw * bh))
    return out


def build_coco(records: List[Dict], cats: List[Dict]) -> Dict:
    coco, ann_id = dict(images=[], annotations=[], categories=cats), 1
    for r in records:
        coco['images'].append(r['img'])
        for b in r['ann']:
            coco['annotations'].append(dict(
                id=ann_id, image_id=r['img']['id'],
                category_id=b['cls_id'] + 1,
                bbox=[b['x'], b['y'], b['w'], b['h']],
                area=b['area'], iscrowd=0))
            ann_id += 1
    return coco


# ------------------------------------------------------------------------------------------------------------------
def group_key(p: Path) -> str:
    """
    Derive 'original' name → grouping key.

    Handles "<name>.jpg"  , "<name>_aug3.jpg", "<prefix>_<name>_aug15.png".
    The rule: cut the first "_aug\d*" suffix if present, else use full stem.
    """
    stem = p.stem
    m = re.match(r'^(.*?)(_aug\d+)?$', stem)
    return m.group(1) if m else stem


def split_groups(imgs: List[Path], ratio: float, seed: int):
    """Return (train_imgs, val_imgs) with group-preserving split."""
    # build {key: [paths]}
    groups = {}
    for p in imgs:
        groups.setdefault(group_key(p), []).append(p)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    k = int(len(keys) * ratio)
    train_keys = set(keys[:k])

    train = list(itertools.chain.from_iterable(groups[k] for k in train_keys))
    val   = list(itertools.chain.from_iterable(groups[k] for k in keys[k:]))
    return train, val


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with images + .txt")
    ap.add_argument("--dst", required=True, help="output COCO folder")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in src.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    train_imgs, val_imgs = split_groups(img_paths, args.train_ratio, args.seed)

    def process(subset: List[Path], name: str):
        records = []
        for img_path in tqdm(subset, desc=name):
            shutil.copyfile(img_path, dst / img_path.name)

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            boxes = yolo_txt_to_boxes(img_path.with_suffix(".txt"), W, H)
            records.append(dict(
                img=dict(id=len(records) + 1,
                         file_name=img_path.name, width=W, height=H),
                ann=boxes))

        cats = [dict(id=i + 1, name=n, supercategory="vehicle")
                for i, n in enumerate(YOLO_CLASSES)]
        coco = build_coco(records, cats)
        coco['info'] = dict(description='auto-converted YOLO → COCO',
                            version='1.1',
                            date_created=datetime.now().strftime('%Y-%m-%d'))
        coco['licenses'] = [dict(name='unknown', id=0)]

        (dst / f"instances_{name}.json").write_text(
            json.dumps(coco, indent=2), encoding='utf-8')
        print(f"[✓] instances_{name}.json written ({len(records)} images)")

    process(train_imgs, "train")
    process(val_imgs,   "val")


if __name__ == "__main__":
    import itertools, json
    main()
