"""
Create tile-based dataset for RTMDet training.
Generates 640x640 tiles from original high-resolution images without downscaling.
Includes optional data augmentation functionality.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import shutil
import glob
import albumentations as A

class TileGenerator:
    def __init__(self, 
                 tile_size: int = 640,
                 overlap_ratio: float = 0.5,
                 min_objects_per_tile: int = 0,
                 min_object_area_ratio: float = 0.3,
                 enable_augmentation: bool = True,
                 augmentation_copies: int = 1,
                 balance_negatives: bool = True,
                 negative_ratio: float = 1.0):
        """
        Initialize tile generator.
        
        Args:
            tile_size: Size of square tiles (default 640x640)
            overlap_ratio: Overlap between adjacent tiles (0.5 = 50%)
            min_objects_per_tile: Minimum objects required to keep a tile
            min_object_area_ratio: Minimum portion of object that must be in tile
            enable_augmentation: Enable data augmentation (requires albumentations)
            augmentation_copies: Number of augmented copies per image
            balance_negatives: Balance number of negative and positive samples
            negative_ratio: Ratio of negative to positive samples (1.0 = 1:1)
        """
        self.tile_size = tile_size
        self.overlap = int(tile_size * overlap_ratio)
        self.stride = tile_size - self.overlap
        self.min_objects_per_tile = min_objects_per_tile
        self.min_object_area_ratio = min_object_area_ratio
        self.enable_augmentation = enable_augmentation
        self.augmentation_copies = augmentation_copies
        self.balance_negatives = balance_negatives
        self.negative_ratio = negative_ratio
        
        # Initialize augmentation transforms if enabled
        if self.enable_augmentation:
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.HueSaturationValue(10, 15, 10, p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.GaussNoise(std_range=[0.01, 0.03], p=0.4), # the last one because it models ISO noise on camera sensor (so it is independently executed on per-pixel basis)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        print(f"TileGenerator initialized:")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Overlap: {self.overlap} pixels ({overlap_ratio*100:.1f}%)")
        print(f"  Stride: {self.stride} pixels")
        print(f"  Min objects per tile: {min_objects_per_tile}")
        print(f"  Min object area ratio: {min_object_area_ratio}")
        if self.enable_augmentation:
            print(f"  🎨 Augmentation enabled: {augmentation_copies} copies per image")
        else:
            print(f"  🎨 Augmentation disabled")
        if self.balance_negatives:
            print(f"  ⚖️ Negative balancing enabled: {negative_ratio:.1f}:1 ratio")
        else:
            print(f"  ⚖️ Negative balancing disabled")

    def load_yolo_annotations(self, annotation_path: str) -> List[Dict]:
        """Load YOLO format annotations."""
        annotations = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        annotations.append({
                            'class_id': int(class_id),
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
        return annotations

    def yolo_to_bbox(self, yolo_ann: Dict, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert YOLO format to bounding box coordinates."""
        x_center = yolo_ann['x_center'] * img_width
        y_center = yolo_ann['y_center'] * img_height
        width = yolo_ann['width'] * img_width
        height = yolo_ann['height'] * img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return x1, y1, x2, y2

    def bbox_intersection_area(self, bbox1: Tuple[int, int, int, int], 
                              bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate intersection area between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there's an intersection
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        return (x2_i - x1_i) * (y2_i - y1_i)

    def bbox_area(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate area of bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def get_tile_positions(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """Generate tile positions for sliding window."""
        positions = []
        
        # Calculate number of tiles in each dimension
        x_positions = list(range(0, img_width - self.tile_size + 1, self.stride))
        y_positions = list(range(0, img_height - self.tile_size + 1, self.stride))
        
        # Add final positions to cover the entire image
        if x_positions[-1] + self.tile_size < img_width:
            x_positions.append(img_width - self.tile_size)
        if y_positions[-1] + self.tile_size < img_height:
            y_positions.append(img_height - self.tile_size)
        
        for y in y_positions:
            for x in x_positions:
                positions.append((x, y))
        
        return positions

    def process_tile(self, image: np.ndarray, annotations: List[Dict], 
                    tile_x: int, tile_y: int, img_width: int, img_height: int) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single tile and its annotations."""
        # Extract tile from image
        tile_image = image[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size]
        
        # Define tile bounding box
        tile_bbox = (tile_x, tile_y, tile_x + self.tile_size, tile_y + self.tile_size)
        
        # Process annotations for this tile
        tile_annotations = []
        
        for ann in annotations:
            # Convert YOLO to bbox coordinates
            obj_bbox = self.yolo_to_bbox(ann, img_width, img_height)
            
            # Calculate intersection with tile
            intersection_area = self.bbox_intersection_area(obj_bbox, tile_bbox)
            object_area = self.bbox_area(obj_bbox)
            
            # Check if enough of the object is in the tile
            if intersection_area > 0 and intersection_area / object_area >= self.min_object_area_ratio:
                # Calculate object coordinates relative to tile
                obj_x1, obj_y1, obj_x2, obj_y2 = obj_bbox
                
                # Clip to tile boundaries
                rel_x1 = max(0, obj_x1 - tile_x)
                rel_y1 = max(0, obj_y1 - tile_y)
                rel_x2 = min(self.tile_size, obj_x2 - tile_x)
                rel_y2 = min(self.tile_size, obj_y2 - tile_y)
                
                # Convert back to YOLO format for the tile
                tile_width = rel_x2 - rel_x1
                tile_height = rel_y2 - rel_y1
                tile_x_center = (rel_x1 + rel_x2) / 2 / self.tile_size
                tile_y_center = (rel_y1 + rel_y2) / 2 / self.tile_size
                tile_norm_width = tile_width / self.tile_size
                tile_norm_height = tile_height / self.tile_size
                
                tile_annotations.append({
                    'class_id': ann['class_id'],
                    'x_center': tile_x_center,
                    'y_center': tile_y_center,
                    'width': tile_norm_width,
                    'height': tile_norm_height,
                    'bbox': [rel_x1, rel_y1, tile_width, tile_height],  # COCO format
                    'area': tile_width * tile_height
                })
        
        return tile_image, tile_annotations

    def create_coco_annotation(self, ann_id: int, image_id: int, category_id: int, 
                              bbox: List[float], area: float) -> Dict:
        """Create COCO format annotation."""
        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(category_id) + 1,  # COCO categories start from 1
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }

    def create_coco_image(self, image_id: int, filename: str, width: int, height: int) -> Dict:
        """Create COCO format image entry."""
        return {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        }

    def read_yolo_for_augmentation(self, annotation_path: str) -> Tuple[List[List[float]], List[int]]:
        """Read YOLO annotations in format expected by albumentations."""
        boxes, labels = [], []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = parts
                        boxes.append([float(xc), float(yc), float(bw), float(bh)])
                        labels.append(int(cls))
        return boxes, labels

    def alb2yolo(self, box: List[float], label: int) -> str:
        """Convert albumentations box format back to YOLO string."""
        return f'{label} ' + ' '.join(f'{v:.6f}' for v in box)

    def augment_image(self, image: np.ndarray, boxes: List[List[float]], labels: List[int], 
                     output_dir: str, base_filename: str, aug_idx: int) -> Tuple[str, str]:
        """Apply augmentation to image and save augmented version."""
        if not self.enable_augmentation or not boxes:
            return None, None
        
        try:
            # Apply augmentation
            augmented = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            aug_image = augmented['image']
            aug_boxes = augmented['bboxes']
            aug_labels = augmented['class_labels']
            
            # Save augmented image
            aug_filename = f"{base_filename}_aug{aug_idx}.jpg"
            aug_image_path = os.path.join(output_dir, aug_filename)
            cv2.imwrite(aug_image_path, aug_image)
            
            # Save augmented annotations
            aug_ann_filename = f"{base_filename}_aug{aug_idx}.txt"
            aug_ann_path = os.path.join(output_dir, "temp_annotations", aug_ann_filename)
            os.makedirs(os.path.join(output_dir, "temp_annotations"), exist_ok=True)
            
            with open(aug_ann_path, 'w') as f:
                for box, lbl in zip(aug_boxes, aug_labels):
                    f.write(self.alb2yolo(box, lbl) + '\n')
            
            return aug_image_path, aug_ann_path
            
        except Exception as e:
            print(f"Warning: Augmentation failed for {base_filename}: {e}")
            return None, None

    def process_single_image_to_tiles(self, image: np.ndarray, annotations: List[Dict], 
                                    output_dir: str, base_filename: str, suffix: str = "") -> Tuple[List[Dict], List[Dict], int, int, int, int]:
        """Process a single image (original or augmented) and generate tiles with balanced negatives."""
        img_height, img_width = image.shape[:2]
        
        # Get tile positions
        tile_positions = self.get_tile_positions(img_width, img_height)
        
        # Separate positive and negative tiles
        positive_tiles = []  # Tiles with objects
        negative_tiles = []  # Tiles without objects
        
        for tile_x, tile_y in tile_positions:
            # Process tile
            tile_image, tile_annotations = self.process_tile(
                image, annotations, tile_x, tile_y, img_width, img_height
            )
            
            tile_data = {
                'image': tile_image,
                'annotations': tile_annotations,
                'position': (tile_x, tile_y),
                'filename': f"{base_filename}{suffix}_tile_{tile_x}_{tile_y}.jpg"
            }
            
            # Classify as positive or negative
            if len(tile_annotations) > 0:
                positive_tiles.append(tile_data)
            else:
                negative_tiles.append(tile_data)
        
        # Balance negatives if enabled
        selected_negatives = []
        if self.balance_negatives and negative_tiles:
            # Calculate how many negatives to include
            num_negatives_needed = int(len(positive_tiles) * self.negative_ratio)
            
            if num_negatives_needed > 0:
                # Randomly sample negatives
                import random
                if len(negative_tiles) >= num_negatives_needed:
                    selected_negatives = random.sample(negative_tiles, num_negatives_needed)
                else:
                    # Use all available negatives if not enough
                    selected_negatives = negative_tiles
        else:
            # Use all negative samples without balancing their number with positive samples
            selected_negatives = negative_tiles
        
        # Combine positive and selected negative tiles
        all_selected_tiles = positive_tiles + selected_negatives
        
        # Save tiles and create COCO entries
        coco_images = []
        coco_annotations = []
        
        for tile_data in all_selected_tiles:
            # Save tile image
            tile_path = os.path.join(output_dir, tile_data['filename'])
            cv2.imwrite(tile_path, tile_data['image'])
            
            # Create COCO image entry
            image_id = len(coco_images) + 1
            coco_images.append(self.create_coco_image(
                image_id, tile_data['filename'], self.tile_size, self.tile_size
            ))
            
            # Create COCO annotations (only for positive tiles)
            for tile_ann in tile_data['annotations']:
                ann_id = len(coco_annotations) + 1
                coco_annotations.append(self.create_coco_annotation(
                    ann_id, image_id, tile_ann['class_id'], 
                    tile_ann['bbox'], tile_ann['area']
                ))
        
        return (coco_images, coco_annotations, len(tile_positions), 
                len(positive_tiles), len(selected_negatives), len(all_selected_tiles))

    def process_image(self, image_path: str, annotation_path: str, 
                     output_dir: str, base_filename: str) -> Tuple[List[Dict], List[Dict]]:
        """Process a single image and generate tiles, with optional augmentation."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return [], []
        
        # Load annotations
        annotations = self.load_yolo_annotations(annotation_path)
        if not annotations:
            print(f"Warning: No annotations found for {image_path}")
            return [], []
        
        all_coco_images = []
        all_coco_annotations = []
        total_tile_count = 0
        total_kept_tiles = 0
        
        # Process original image
        (coco_images, coco_annotations, tile_count, 
         positive_tiles, negative_tiles, kept_tiles) = self.process_single_image_to_tiles(
            image, annotations, output_dir, base_filename, ""
        )
        
        # Update IDs for original image tiles
        for img in coco_images:
            img['id'] = len(all_coco_images) + 1
            all_coco_images.append(img)
        
        for ann in coco_annotations:
            ann['id'] = len(all_coco_annotations) + 1
            # Update image_id to match the current image list
            ann['image_id'] = len(all_coco_images) - len(coco_images) + ann['image_id']
            all_coco_annotations.append(ann)
        
        total_tile_count += tile_count
        total_kept_tiles += kept_tiles
        total_positive_tiles = positive_tiles
        total_negative_tiles = negative_tiles
        
        # Process augmented images if augmentation is enabled
        if self.enable_augmentation:
            # Read annotations in albumentations format
            boxes, labels = self.read_yolo_for_augmentation(annotation_path)
            
            if boxes:  # Only augment if there are annotations
                for aug_idx in range(self.augmentation_copies):
                    try:
                        # Apply augmentation
                        augmented = self.transforms(image=image, bboxes=boxes, class_labels=labels)
                        aug_image = augmented['image']
                        aug_boxes = augmented['bboxes']
                        aug_labels = augmented['class_labels']
                        
                        # Convert augmented annotations back to our format
                        aug_annotations = []
                        for box, label in zip(aug_boxes, aug_labels):
                            aug_annotations.append({
                                'class_id': label,
                                'x_center': box[0],
                                'y_center': box[1],
                                'width': box[2],
                                'height': box[3]
                            })
                        
                        # Process augmented image to tiles
                        (aug_coco_images, aug_coco_annotations, aug_tile_count, 
                         aug_positive_tiles, aug_negative_tiles, aug_kept_tiles) = self.process_single_image_to_tiles(
                            aug_image, aug_annotations, output_dir, base_filename, f"_aug{aug_idx}"
                        )
                        
                        # Update IDs for augmented image tiles
                        for img in aug_coco_images:
                            img['id'] = len(all_coco_images) + 1
                            all_coco_images.append(img)
                        
                        for ann in aug_coco_annotations:
                            ann['id'] = len(all_coco_annotations) + 1
                            # Update image_id to match the current image list
                            ann['image_id'] = len(all_coco_images) - len(aug_coco_images) + ann['image_id']
                            all_coco_annotations.append(ann)
                        
                        total_tile_count += aug_tile_count
                        total_kept_tiles += aug_kept_tiles
                        
                    except Exception as e:
                        print(f"Warning: Augmentation {aug_idx} failed for {base_filename}: {e}")
                        continue
        
        print(f"  Generated {total_tile_count} tiles, kept {total_kept_tiles} tiles")
        return all_coco_images, all_coco_annotations

    def create_tiles_dataset(self, input_dir: str, output_dir: str, 
                           train_ratio: float = 0.8) -> None:
        """Create complete tiles dataset from YOLO format dataset with proper image-level splitting."""
        print(f"\nCreating tiles dataset:")
        print(f"  Input directory: {input_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Train/Val split: {train_ratio:.1%}/{1-train_ratio:.1%}")
        print("  🔒 FIXED: Prevents data leakage by splitting images first, then processing tiles")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files with annotations (deduplicate by stem name)
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files_dict = {}
        
        for ext in image_extensions:
            for image_path in Path(input_dir).glob(f"*{ext}"):
                # Only include images that have corresponding annotation files
                annotation_path = image_path.with_suffix('.txt')
                if annotation_path.exists():
                    stem_name = image_path.stem
                    # Keep only one file per stem name (prefer JPG over other formats)
                    if stem_name not in image_files_dict or ext.upper() == '.JPG':
                        image_files_dict[stem_name] = image_path
        
        image_files = sorted(image_files_dict.values())
        print(f"Found {len(image_files)} unique images with annotations")
        
        if not image_files:
            print("No images with annotations found!")
            return
        
        # Load class names
        classes_file = os.path.join(input_dir, "classes.txt")
        class_names = ['car']  # Default classes
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
        
        print(f"Classes: {class_names}")
        
        # CRITICAL FIX: Split images first to prevent data leakage
        import random
        random.seed(42)  # For reproducible splits
        random.shuffle(image_files)
        
        train_count = int(len(image_files) * train_ratio)
        train_image_files = image_files[:train_count]
        val_image_files = image_files[train_count:]
        
        print(f"Image-level split:")
        print(f"  Train images: {len(train_image_files)}")
        print(f"  Val images: {len(val_image_files)}")
        
        # Process train and validation sets separately
        train_coco_images = []
        train_coco_annotations = []
        val_coco_images = []
        val_coco_annotations = []
        
        # Process training images
        print(f"\nProcessing training images...")
        for image_path in tqdm(train_image_files, desc="Train images"):
            base_filename = image_path.stem
            annotation_path = image_path.with_suffix('.txt')
            
            coco_images, coco_annotations = self.process_image(
                str(image_path), str(annotation_path), output_dir, base_filename
            )
            
            # Add to train lists with updated IDs
            for img in coco_images:
                img['id'] = len(train_coco_images) + 1
                train_coco_images.append(img)
            
            for ann in coco_annotations:
                ann['id'] = len(train_coco_annotations) + 1
                # Update image_id to match the train image list
                ann['image_id'] = len(train_coco_images) - len(coco_images) + ann['image_id']
                train_coco_annotations.append(ann)
        
        # Process validation images
        print(f"\nProcessing validation images...")
        for image_path in tqdm(val_image_files, desc="Val images"):
            base_filename = image_path.stem
            annotation_path = image_path.with_suffix('.txt')
            
            coco_images, coco_annotations = self.process_image(
                str(image_path), str(annotation_path), output_dir, base_filename
            )
            
            # Add to val lists with updated IDs
            for img in coco_images:
                img['id'] = len(val_coco_images) + 1
                val_coco_images.append(img)
            
            for ann in coco_annotations:
                ann['id'] = len(val_coco_annotations) + 1
                # Update image_id to match the val image list
                ann['image_id'] = len(val_coco_images) - len(coco_images) + ann['image_id']
                val_coco_annotations.append(ann)
        
        print(f"\nDataset generation complete!")
        print(f"Train tiles: {len(train_coco_images)}, annotations: {len(train_coco_annotations)}")
        print(f"Val tiles: {len(val_coco_images)}, annotations: {len(val_coco_annotations)}")
        print(f"Total tiles: {len(train_coco_images) + len(val_coco_images)}")
        
        # Create COCO format datasets
        categories = [
            {"id": i + 1, "name": name, "supercategory": "vehicle"}
            for i, name in enumerate(class_names)
        ]
        
        train_coco = {
            "images": train_coco_images,
            "annotations": train_coco_annotations,
            "categories": categories,
            "info": {
                "description": f"Tile-based dataset ({self.tile_size}x{self.tile_size}) - TRAIN",
                "version": "1.0",
                "date_created": "2025-01-11"
            },
            "licenses": [{"name": "unknown", "id": 0}]
        }
        
        val_coco = {
            "images": val_coco_images,
            "annotations": val_coco_annotations,
            "categories": categories,
            "info": {
                "description": f"Tile-based dataset ({self.tile_size}x{self.tile_size}) - VAL",
                "version": "1.0",
                "date_created": "2025-01-11"
            },
            "licenses": [{"name": "unknown", "id": 0}]
        }
        
        # Save COCO format files
        train_json_path = os.path.join(output_dir, "instances_train.json")
        val_json_path = os.path.join(output_dir, "instances_val.json")
        
        with open(train_json_path, 'w') as f:
            json.dump(train_coco, f, indent=2)
        
        with open(val_json_path, 'w') as f:
            json.dump(val_coco, f, indent=2)
        
        print(f"Saved train annotations: {train_json_path}")
        print(f"Saved val annotations: {val_json_path}")
        
        # Generate statistics for combined dataset
        all_images = train_coco_images + val_coco_images
        all_annotations = train_coco_annotations + val_coco_annotations
        self.print_statistics(all_images, all_annotations, class_names)
        
        # Verify no data leakage
        train_source_images = {img['file_name'].split('_tile_')[0] for img in train_coco_images}
        val_source_images = {img['file_name'].split('_tile_')[0] for img in val_coco_images}
        overlap = train_source_images.intersection(val_source_images)
        
        if overlap:
            print(f"⚠️  WARNING: Data leakage detected! {len(overlap)} source images appear in both sets")
            print(f"Overlapping images: {sorted(overlap)}")
        else:
            print(f"✅ No data leakage: Train and validation sets use completely different source images")

    def print_statistics(self, images: List[Dict], annotations: List[Dict], class_names: List[str]) -> None:
        """Print dataset statistics."""
        print(f"\n{'='*50}")
        print("DATASET STATISTICS")
        print(f"{'='*50}")
        
        print(f"Total tiles: {len(images)}")
        print(f"Total annotations: {len(annotations)}")
        print(f"Average annotations per tile: {len(annotations)/len(images):.2f}")
        
        # Positive vs Negative tiles
        positive_tiles = len([img for img in images if any(ann['image_id'] == img['id'] for ann in annotations)])
        negative_tiles = len(images) - positive_tiles
        
        if self.balance_negatives:
            print(f"\n⚖️ Balanced Dataset:")
            print(f"  Positive tiles (with objects): {positive_tiles}")
            print(f"  Negative tiles (background): {negative_tiles}")
            print(f"  Positive:Negative ratio: {positive_tiles/negative_tiles:.2f}:1" if negative_tiles > 0 else "  All tiles are positive")
        
        # Class distribution
        class_counts = {}
        for ann in annotations:
            class_id = int(ann['category_id']) - 1  # Convert back to 0-based
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            percentage = count / len(annotations) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Object size statistics
        areas = [ann['area'] for ann in annotations]
        if areas:
            print(f"\nObject size statistics (pixels²):")
            print(f"  Min area: {min(areas):.0f}")
            print(f"  Max area: {max(areas):.0f}")
            print(f"  Mean area: {np.mean(areas):.0f}")
            print(f"  Median area: {np.median(areas):.0f}")
        
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Create tile-based dataset for RTMDet training with optional augmentation")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory with YOLO format dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for tile-based dataset")
    parser.add_argument("--tile_size", type=int, default=640,
                       help="Size of square tiles (default: 640)")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                       help="Overlap ratio between tiles (default: 0.5)")
    parser.add_argument("--min_objects", type=int, default=0,
                       help="Minimum objects per tile to keep (default: 0)")
    parser.add_argument("--min_area_ratio", type=float, default=0.3,
                       help="Minimum object area ratio in tile (default: 0.3)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Train/validation split ratio (default: 0.8)")
    
    # Augmentation arguments
    parser.add_argument("--no_augs", action="store_true",
                       help="Disable data augmentation")
    parser.add_argument("--aug_copies", type=int, default=5,
                       help="Number of augmented copies per image (default: 5)")
    
    # Negative balancing arguments
    parser.add_argument("--no_balance_negatives", action="store_true",
                       help="Disable negative sample balancing")
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                       help="Ratio of negative to positive samples (default: 1.0)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    if args.tile_size <= 0:
        print("Error: Tile size must be positive!")
        return
    
    if not (0.0 <= args.overlap_ratio < 1.0):
        print("Error: Overlap ratio must be between 0.0 and 1.0!")
        return
    
    if not (0.0 < args.train_ratio < 1.0):
        print("Error: Train ratio must be between 0.0 and 1.0!")
        return
    
    # Determine augmentation setting (enabled by default, disabled with --no_augs)
    enable_augmentation = not args.no_augs
    
    if enable_augmentation and args.aug_copies <= 0:
        print("Error: Number of augmentation copies must be positive!")
        return
    
    # Determine negative balancing setting (enabled by default, disabled with --no_balance_negatives)
    balance_negatives = not args.no_balance_negatives
    
    if args.negative_ratio <= 0:
        print("Error: Negative ratio must be positive!")
        return
    
    # Create tile generator
    generator = TileGenerator(
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        min_objects_per_tile=args.min_objects,
        min_object_area_ratio=args.min_area_ratio,
        enable_augmentation=enable_augmentation,
        augmentation_copies=args.aug_copies,
        balance_negatives=balance_negatives,
        negative_ratio=args.negative_ratio
    )
    
    # Generate tiles dataset
    generator.create_tiles_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio
    )
    
    print(f"\n✅ Tile-based dataset created successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Ready for RTMDet training with {args.tile_size}x{args.tile_size} tiles")


if __name__ == "__main__":
    main()
