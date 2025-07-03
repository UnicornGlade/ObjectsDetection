"""
Create K augmented versions for every annotated image in YOLO format.
"""
import argparse, glob, os, cv2
import albumentations as A
from tqdm import tqdm
from utils.box import read_yolo, alb2yolo
from utils.paths import ensure_dir

TRANSFORMS = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Original dataset dir')
    ap.add_argument('--dst', required=True, help='Dest dir for augmented set')
    ap.add_argument('-k', '--copies', type=int, default=5)
    args = ap.parse_args()

    img_files = glob.glob(os.path.join(args.src, '*.jpg')) + glob.glob(os.path.join(args.src, '*.png'))
    dst_dir = ensure_dir(args.dst)

    for img_path in tqdm(img_files):
        base = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = img_path.rsplit('.', 1)[0] + '.txt'
        orig_img = cv2.imread(img_path)
        h, w = orig_img.shape[:2]
        boxes, labels = read_yolo(ann_path)
        for i in range(args.copies):
            augmented = TRANSFORMS(image=orig_img, bboxes=boxes, class_labels=labels)
            img_out = augmented['image']
            boxes_out = augmented['bboxes']
            name = f'{base}_aug{i}.jpg'
            cv2.imwrite(os.path.join(dst_dir, name), img_out)
            with open(os.path.join(dst_dir, f'{base}_aug{i}.txt'), 'w') as f:
                for box, lbl in zip(boxes_out, labels):
                    f.write(alb2yolo(box, lbl) + '\n')

if __name__ == '__main__':
    main()
