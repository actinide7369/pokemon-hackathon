#!/usr/bin/env python3
"""
Specific improvements for identified issues:
1. Charizard false positive reduction via strict color & shape filtering
2. Bulbasaur confidence boosting with spatial clustering
3. Mewtwo multi-pose detection enhancement
4. Adaptive confidence thresholds per class

Author: Final CV System v4.0
"""

import torch
import numpy as np
import cv2
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict
import shutil
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== POKEMON CLASS MAPPING ======================
POKEMON_CLASSES = {
    0: "Pikachu",
    1: "Charizard", 
    2: "Bulbasaur",
    3: "Mewtwo"
}

POKEMON_CLASS_IDS = {v: k for k, v in POKEMON_CLASSES.items()}

# Enhanced class-specific configuration with issue-specific fixes
CLASS_CONFIG = {
    "Pikachu": {
        "conf_threshold": 0.25,
        "nms_threshold": 0.4,
        "max_aspect_ratio": 1.5,
        "min_area": 400,
        "max_area": 40000
    },
    "Charizard": {
        "conf_threshold": 0.45,  # Raised to reduce false positives
        "nms_threshold": 0.25,  # Stricter NMS
        "max_aspect_ratio": 2.0,
        "min_area": 600,
        "max_area": 50000,
        "use_strict_color_filter": True,  # Enhanced color filtering
        "use_shape_validation": True,  # Additional shape validation
        "orange_ratio_threshold": 0.35,  # Minimum orange pixels ratio
        "edge_density_threshold": 0.15  # Minimum edge density for real Charizard
    },
    "Bulbasaur": {
        "conf_threshold": 0.15,  # Lower threshold
        "nms_threshold": 0.6,  # Higher to keep more detections for clustering
        "max_aspect_ratio": 1.8,
        "min_area": 500,
        "max_area": 45000,
        "use_clustering": True,  # Enable spatial clustering
        "cluster_eps": 50,  # DBSCAN epsilon for clustering
        "boost_clustered_confidence": True,  # Boost confidence for clustered detections
        "confidence_boost_factor": 1.5  # Multiply confidence by this for clustered detections
    },
    "Mewtwo": {
        "conf_threshold": 0.20,  # Lowered for better pose detection
        "nms_threshold": 0.35,
        "max_aspect_ratio": 3.0,  # Higher for different poses
        "min_area": 400,
        "max_area": 60000,
        "use_pose_variants": True,  # Enable pose-specific detection
        "use_gradient_matching": True  # Match gradient patterns for Mewtwo shape
    }
}

# ====================== ADVANCED FILTERING & VALIDATION ======================
class AdvancedFilter:
    """Advanced filtering for reducing false positives"""
    
    @staticmethod
    def validate_charizard(image: np.ndarray, bbox: List[float]) -> bool:
        """
        Strict validation for Charizard to reduce false positives
        """
        x, y, w, h = [int(v) for v in bbox]
        roi = image[max(0, y):min(image.shape[0], y+h), 
                   max(0, x):min(image.shape[1], x+w)]
        
        if roi.size == 0:
            return False
        
        # 1. Color validation - check for orange/red dominance
        orange_ratio = AdvancedFilter._calculate_orange_ratio(roi)
        if orange_ratio < CLASS_CONFIG["Charizard"]["orange_ratio_threshold"]:
            logger.debug(f"Charizard rejected: orange_ratio={orange_ratio:.2f}")
            return False
        
        # 2. Edge density check - real Charizard has distinct edges
        edge_density = AdvancedFilter._calculate_edge_density(roi)
        if edge_density < CLASS_CONFIG["Charizard"]["edge_density_threshold"]:
            logger.debug(f"Charizard rejected: edge_density={edge_density:.2f}")
            return False
        
        # 3. Shape validation - check for wing-like structures
        has_wings = AdvancedFilter._detect_wing_structure(roi)
        if not has_wings:
            logger.debug("Charizard rejected: no wing structure detected")
            return False
        
        return True
    
    @staticmethod
    def _calculate_orange_ratio(roi: np.ndarray) -> float:
        """Calculate ratio of orange/red pixels in ROI"""
        if len(roi.shape) == 2:
            return 0.0
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Define orange/red color ranges
        lower_orange = np.array([180, 50, 0])
        upper_orange = np.array([255, 180, 100])
        lower_red = np.array([150, 0, 0])
        upper_red = np.array([255, 100, 100])
        
        # Create masks
        mask_orange = cv2.inRange(roi_rgb, lower_orange, upper_orange)
        mask_red = cv2.inRange(roi_rgb, lower_red, upper_red)
        mask_combined = cv2.bitwise_or(mask_orange, mask_red)
        
        # Calculate ratio
        total_pixels = roi.shape[0] * roi.shape[1]
        orange_pixels = np.sum(mask_combined > 0)
        
        return orange_pixels / total_pixels if total_pixels > 0 else 0
    
    @staticmethod
    def _calculate_edge_density(roi: np.ndarray) -> float:
        """Calculate edge density in ROI"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    @staticmethod
    def _detect_wing_structure(roi: np.ndarray) -> bool:
        """Detect wing-like structures using contour analysis"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:  # Need at least 2 major contours for wings
            return False
        
        # Check for symmetric large contours (wings)
        areas = [cv2.contourArea(c) for c in contours]
        areas.sort(reverse=True)
        
        if len(areas) >= 2:
            # Check if two largest contours are similar in size (wings)
            ratio = areas[1] / areas[0] if areas[0] > 0 else 0
            return 0.3 < ratio < 1.0  # Wings should be somewhat similar in size
        
        return False
    
    @staticmethod
    def validate_mewtwo(image: np.ndarray, bbox: List[float]) -> bool:
        """
        Validation for Mewtwo considering different poses
        """
        x, y, w, h = [int(v) for v in bbox]
        roi = image[max(0, y):min(image.shape[0], y+h), 
                   max(0, x):min(image.shape[1], x+w)]
        
        if roi.size == 0:
            return False
        
        # Check for purple/pink colors
        purple_ratio = AdvancedFilter._calculate_purple_ratio(roi)
        
        # Check for elongated body structure (tall/thin)
        aspect_ratio = h / w if w > 0 else 0
        
        # Mewtwo is typically tall and has purple coloring
        return purple_ratio > 0.15 or aspect_ratio > 1.2
    
    @staticmethod
    def _calculate_purple_ratio(roi: np.ndarray) -> float:
        """Calculate ratio of purple/pink pixels in ROI"""
        if len(roi.shape) == 2:
            return 0.0
        
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Define purple/pink color ranges
        lower_purple = np.array([100, 0, 100])
        upper_purple = np.array([255, 150, 255])
        
        mask = cv2.inRange(roi_rgb, lower_purple, upper_purple)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        purple_pixels = np.sum(mask > 0)
        
        return purple_pixels / total_pixels if total_pixels > 0 else 0

# ====================== SPATIAL CLUSTERING FOR BULBASAUR ======================
class SpatialClusterer:
    """Spatial clustering to handle multiple close Bulbasaur detections"""
    
    @staticmethod
    def cluster_and_boost(detections: List[Dict], pokemon_class: str) -> List[Dict]:
        """
        Cluster nearby detections and boost confidence for clustered ones
        """
        if not detections or pokemon_class != "Bulbasaur":
            return detections
        
        config = CLASS_CONFIG[pokemon_class]
        
        if not config.get("use_clustering", False):
            return detections
        
        # Extract centers for clustering
        centers = np.array([[d['center'][0], d['center'][1]] for d in detections])
        
        # Apply DBSCAN clustering
        eps = config.get("cluster_eps", 50)
        clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
        
        # Count detections per cluster
        cluster_counts = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            cluster_counts[label].append(idx)
        
        # Boost confidence for clusters with multiple detections
        boosted_detections = []
        processed_clusters = set()
        
        for label, indices in cluster_counts.items():
            if label in processed_clusters:
                continue
            
            cluster_dets = [detections[i] for i in indices]
            
            if len(cluster_dets) > 1:
                # Multiple detections in cluster - merge and boost
                best_det = max(cluster_dets, key=lambda x: x['confidence'])
                
                # Calculate cluster center
                cluster_center = np.mean([centers[i] for i in indices], axis=0)
                
                # Boost confidence
                boost_factor = config.get("confidence_boost_factor", 1.5)
                boosted_conf = min(best_det['confidence'] * boost_factor * (1 + 0.1 * len(cluster_dets)), 0.95)
                
                # Create boosted detection
                boosted_det = best_det.copy()
                boosted_det['confidence'] = boosted_conf
                boosted_det['center'] = cluster_center.tolist()
                boosted_det['clustered'] = True
                boosted_det['cluster_size'] = len(cluster_dets)
                
                boosted_detections.append(boosted_det)
                logger.debug(f"Bulbasaur cluster: {len(cluster_dets)} detections merged, confidence boosted to {boosted_conf:.3f}")
            else:
                # Single detection - keep as is but mark
                det = cluster_dets[0].copy()
                det['clustered'] = False
                det['cluster_size'] = 1
                boosted_detections.append(det)
            
            processed_clusters.add(label)
        
        return boosted_detections

# ====================== POSE-AWARE DETECTION FOR MEWTWO ======================
class PoseAwareDetection:
    """Handle different poses for Mewtwo detection"""
    
    @staticmethod
    def apply_pose_variants(image: np.ndarray) -> List[np.ndarray]:
        """Generate pose variants for better Mewtwo detection"""
        variants = [image]  # Original
        
        # Add contrast enhanced version (helps with different lighting)
        enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
        variants.append(enhanced)
        
        # Add edge-enhanced version (helps with pose outlines)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined = cv2.addWeighted(image, 0.7, edge_colored, 0.3, 0)
        variants.append(combined)
        
        return variants

# ====================== OPTIMIZED DETECTOR WITH FIXES ======================
class FinalPokemonDetector:
    """Final detector with all specific fixes"""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.advanced_filter = AdvancedFilter()
        self.clusterer = SpatialClusterer()
        self.pose_detector = PoseAwareDetection()
        
        logger.info(f"Final detector initialized with model: {model_path}")
    
    def detect_pokemon(self, image_path: str, target_pokemon: str) -> List[Dict]:
        """
        Detect specific Pokemon with all optimizations
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        config = CLASS_CONFIG[target_pokemon]
        all_detections = []
        
        # Special handling for Mewtwo - use pose variants
        if target_pokemon == "Mewtwo" and config.get("use_pose_variants", False):
            image_variants = self.pose_detector.apply_pose_variants(image)
        else:
            image_variants = [image]
        
        for idx, img_variant in enumerate(image_variants):
            # Save temp image
            temp_path = f"temp_variant_{idx}.jpg"
            cv2.imwrite(temp_path, img_variant)
            
            # Run detection with class-specific threshold
            results = self.model(
                temp_path, 
                conf=config['conf_threshold'],
                iou=config['nms_threshold'],
                device=self.device,
                verbose=False
            )
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_name = POKEMON_CLASSES.get(class_id, "Unknown")
                        
                        # Filter for target class
                        if class_name != target_pokemon:
                            continue
                        
                        # Get bbox
                        x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                        bbox = [x_center - width/2, y_center - height/2, width, height]
                        
                        # Apply class-specific validation
                        valid = True
                        
                        # Strict Charizard validation
                        if target_pokemon == "Charizard" and config.get("use_strict_color_filter", False):
                            valid = self.advanced_filter.validate_charizard(image, bbox)
                        
                        # Mewtwo validation
                        elif target_pokemon == "Mewtwo":
                            valid = self.advanced_filter.validate_mewtwo(image, bbox)
                        
                        # Size validation for all
                        area = width * height
                        if area < config.get("min_area", 0) or area > config.get("max_area", float('inf')):
                            valid = False
                        
                        if valid:
                            detection = {
                                'bbox': bbox,
                                'center': [float(x_center), float(y_center)],
                                'confidence': box.conf.item(),
                                'class_id': class_id,
                                'class_name': class_name,
                                'variant_idx': idx
                            }
                            all_detections.append(detection)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Apply NMS across all variants
        merged_detections = self._merge_variant_detections(all_detections, config['nms_threshold'])
        
        # Special handling for Bulbasaur - clustering and confidence boosting
        if target_pokemon == "Bulbasaur":
            merged_detections = self.clusterer.cluster_and_boost(merged_detections, target_pokemon)
        
        # Final filtering by confidence (after boosting)
        final_threshold = config['conf_threshold']
        if target_pokemon == "Bulbasaur":
            # Lower final threshold for Bulbasaur since we've boosted clustered detections
            final_threshold *= 0.8
        
        final_detections = [d for d in merged_detections if d['confidence'] >= final_threshold]
        
        return final_detections
    
    def _merge_variant_detections(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Merge detections from different variants"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
                else:
                    # Merge confidence if from different variants
                    if det['variant_idx'] != best['variant_idx']:
                        best['confidence'] = min(best['confidence'] * 1.1, 0.99)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in xywh format"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to xyxy
        box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
        box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        x_left = max(box1_xyxy[0], box2_xyxy[0])
        y_top = max(box1_xyxy[1], box2_xyxy[1])
        x_right = min(box1_xyxy[2], box2_xyxy[2])
        y_bottom = min(box1_xyxy[3], box2_xyxy[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def get_final_targets(self, image_path: str, target_pokemon: str, 
                         use_heuristics: bool = True) -> List[List[float]]:
        """
        Get final targeting coordinates with all optimizations
        """
        # Detect with all fixes
        detections = self.detect_pokemon(image_path, target_pokemon)
        
        if not detections:
            logger.warning(f"No {target_pokemon} detected in {image_path}")
            return []
        
        logger.info(f"Detected {len(detections)} {target_pokemon} candidates")
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        coordinates = []
        
        if use_heuristics:
            # Use 2-miss advantage
            added_buffer = 0
            max_buffer = 2
            
            for i, det in enumerate(detections):
                # Always add high confidence detections
                if det['confidence'] > 0.7:
                    coordinates.append(det['center'])
                    
                    # Add buffer shot for very high confidence
                    if det['confidence'] > 0.85 and added_buffer < max_buffer:
                        buffer_center = [
                            det['center'][0] + random.uniform(-3, 3),
                            det['center'][1] + random.uniform(-3, 3)
                        ]
                        coordinates.append(buffer_center)
                        added_buffer += 1
                        logger.debug(f"Added buffer shot for high-confidence {target_pokemon}")
                
                # For Bulbasaur, be more aggressive with clustered detections
                elif target_pokemon == "Bulbasaur" and det.get('clustered', False):
                    coordinates.append(det['center'])
                    logger.debug(f"Added clustered Bulbasaur (cluster_size={det.get('cluster_size', 1)})")
                
                # Medium confidence - add if we have buffer remaining
                elif det['confidence'] > 0.4 and added_buffer < max_buffer:
                    coordinates.append(det['center'])
                    added_buffer += 1
                
                # Lower confidence - only if plenty of buffer
                elif det['confidence'] > 0.3 and len(coordinates) < 8:
                    coordinates.append(det['center'])
        else:
            # No heuristics - just take top detections
            for det in detections[:10]:
                coordinates.append(det['center'])
        
        # Convert to COCO format (already in correct format from YOLO)
        final_coordinates = [[float(x), float(y)] for x, y in coordinates[:10]]
        
        logger.info(f"Generated {len(final_coordinates)} shots for {target_pokemon}")
        for i, (x, y) in enumerate(final_coordinates):
            conf = detections[i]['confidence'] if i < len(detections) else 0
            logger.debug(f"  Shot {i+1}: ({x:.1f}, {y:.1f}) conf={conf:.3f}")
        
        return final_coordinates

# ====================== ENHANCED TRAINING WITH FIXES ======================
class FinalTrainingPipeline:
    """Training pipeline with specific fixes for each Pokemon"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.base_augmentation = self._get_enhanced_augmentation()
    
    def _get_enhanced_augmentation(self):
        """Get augmentation pipeline with specific fixes"""
        return A.Compose([
            # Strong rotation for Bulbasaur
            A.RandomRotate90(p=0.7),
            A.Rotate(limit=180, p=0.8, border_mode=cv2.BORDER_CONSTANT),
            
            # Color augmentation for Charizard false positive reduction
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.7),
            
            # Perspective changes for Mewtwo poses
            A.Perspective(scale=(0.05, 0.2), p=0.5),
            A.Affine(
                scale=(0.6, 1.4),
                translate_percent=(-0.3, 0.3),
                rotate=(-45, 45),
                shear=(-20, 20),
                p=0.6
            ),
            
            # Heavy augmentation for all
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Noise for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
            ], p=0.4),
            
            # Blur variations
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1),
                A.GaussianBlur(blur_limit=(3, 9), p=1),
            ], p=0.3),
            
            # Cutout for occlusion handling
            A.CoarseDropout(
                max_holes=10,
                max_height=40,
                max_width=40,
                min_holes=1,
                min_height=10,
                min_width=10,
                p=0.4
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def train_final_model(self, model_size: str = 'x') -> str:
        """Train final model with all optimizations"""
        logger.info(f"Training final YOLOv8{model_size} model with enhanced pipeline...")
        
        # Prepare dataset (assuming it's already in YOLO format)
        yolo_yaml = self.dataset_path / "yolo_dataset" / "pokemon.yaml"
        
        if not yolo_yaml.exists():
            raise FileNotFoundError(f"YOLO dataset not found: {yolo_yaml}")
        
        model = YOLO(f"yolov8{model_size}.pt")
        
        # Train with optimized parameters
        results = model.train(
            data=str(yolo_yaml),
            epochs=100,     #In hackathon I stopped it around ~67 epochs cuz my model wasnt improving that much anwyays and I thought it will lead to overfitting
            imgsz=640,
            batch=8 if model_size == 'x' else 16,
            optimizer="AdamW",
            lr0=0.0005,  # Lower learning rate for stability
            lrf=0.001,
            momentum=0.937,
            weight_decay=0.001,
            warmup_epochs=10,
            warmup_momentum=0.8,
            box=7.5,
            cls=1.0,  # Higher classification loss weight
            dfl=1.5,
            degrees=180.0,  # Full rotation augmentation
            translate=0.3,
            scale=0.7,
            shear=20.0,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.3,
            copy_paste=0.4,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,
            crop_fraction=1.0,  # No cropping to preserve full Pokemon
            name="pokemon_final_model",
            device=0 if torch.cuda.is_available() else "cpu",
            amp=True,
            patience=50,
            save=True,
            exist_ok=True,
            val=True,
            plots=True
        )
        
        model_path = "runs/detect/pokemon_final_model/weights/best.pt"
        logger.info(f"Final model saved to {model_path}")
        return model_path

# ====================== MAIN EXECUTION ======================
def main():
    """Main function to run the final optimized pipeline"""
    logger.info("🚀 Starting Final Pokemon Detection System")
    
    # Configuration
    dataset_path = r"D:\IIT Madras\Competitions\dataset\dataset"
    
    try:
        # Option 1: Train new model with fixes
        pipeline = FinalTrainingPipeline(dataset_path)
        model_path = pipeline.train_final_model(model_size='x')  # Use 'x' for best results
        
        # Option 2: Use existing model (update path)
        # model_path = "runs/detect/pokemon_final_model/weights/best.pt"
        
        # Initialize final detector
        detector = FinalPokemonDetector(model_path)
        
        # Test on sample images
        test_image = str(Path(dataset_path) / "images" / "img_00362.png")  # Update with actual image
        
        # Test each Pokemon with specific fixes
        for pokemon_name in POKEMON_CLASSES.values():
            logger.info(f"\nTesting {pokemon_name} detection:")
            
            try:
                coordinates = detector.get_final_targets(
                    test_image,
                    pokemon_name,
                    use_heuristics=True
                )
                
                if coordinates:
                    logger.info(f"  ✅ {pokemon_name}: {len(coordinates)} targets")
                    for i, (x, y) in enumerate(coordinates[:3]):  # Show first 3
                        logger.info(f"     Target {i+1}: ({x:.1f}, {y:.1f})")
                else:
                    logger.info(f"  ❌ {pokemon_name}: No targets detected")
                    
            except Exception as e:
                logger.error(f"Error detecting {pokemon_name}: {e}")
        
        logger.info("\n✅ Final system ready for competition!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
