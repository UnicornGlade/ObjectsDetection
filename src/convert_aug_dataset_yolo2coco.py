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
import argparse, json, os, shutil, random, glob, itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import cv2
from tqdm import tqdm

# ---- your class list --------------------------------------------------------
# YOLO id : name ; COCO ids must start at 1, so we add +1 later.
YOLO_CLASSES = ['car', 'bus', 'truck']


# --------------------------------------------------------------------------- #
def yolo_txt_to_boxes(txt_path: Path, W: int, H: int) -> List[Dict]:
    """Read YOLO txt and return list of dicts: {cls_id, x, y, w, h} in pixels."""
    boxes = []
    if not txt_path.exists():
        return boxes
    with txt_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            xc *= W
            yc *= H
            bw *= W
            bh *= H
            x1 = xc - bw / 2
            y1 = yc - bh / 2
            boxes.append(
                dict(
                    cls_id=int(cls),
                    x=float(x1),
                    y=float(y1),
                    w=float(bw),
                    h=float(bh),
                    area=float(bw * bh),
                )
            )
    return boxes


def build_coco_dict(
    records: List[Dict], categories: List[Dict]
) -> Dict:
    """records = [{'img': {...}, 'ann': [...]}, ...]"""
    coco = dict(
        images=[],
        annotations=[],
        categories=categories,
    )
    ann_id = 1
    for r in records:
        coco["images"].append(r["img"])
        for box in r["ann"]:
            coco["annotations"].append(
                dict(
                    id=ann_id,
                    image_id=r["img"]["id"],
                    category_id=box["cls_id"] + 1,  # +1 => 1..N
                    bbox=[box["x"], box["y"], box["w"], box["h"]],
                    area=box["area"],
                    iscrowd=0,
                )
            )
            ann_id += 1
    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with images + .txt")
    ap.add_argument("--dst", required=True, help="output COCO folder")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # gather images
    img_paths = sorted(
        p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    random.seed(args.seed)
    random.shuffle(img_paths)

    split_idx = int(len(img_paths) * args.train_ratio)
    train_imgs = img_paths[:split_idx]
    val_imgs = img_paths[split_idx:]

    def process_subset(subset: List[Path], subset_name: str):
        records = []
        for idx, img_path in enumerate(tqdm(subset, desc=subset_name)):
            # copy image
            dst_img = dst / img_path.name
            shutil.copyfile(img_path, dst_img)

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            txt_path = img_path.with_suffix(".txt")
            boxes = yolo_txt_to_boxes(txt_path, W, H)

            records.append(
                dict(
                    img=dict(
                        id=len(records) + 1,
                        file_name=img_path.name,
                        width=W,
                        height=H,
                    ),
                    ann=boxes,
                )
            )

        # categories section
        categories = [
            dict(id=i + 1, name=n, supercategory="vehicle")
            for i, n in enumerate(YOLO_CLASSES)
        ]
        coco = build_coco_dict(records, categories)

        coco['info'] = dict(
            description='auto-converted YOLO → COCO',
            version='1.0',
            date_created=datetime.now().strftime('%Y-%m-%d'))
        coco['licenses'] = [dict(name='unknown', id=0)]

        json_out = dst / f"instances_{subset_name}.json"
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        print(f"[✓] {json_out} written ({len(records)} images)")

    process_subset(train_imgs, "train")
    process_subset(val_imgs, "val")


if __name__ == "__main__":
    main()