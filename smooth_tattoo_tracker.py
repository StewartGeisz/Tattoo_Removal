#!/usr/bin/env python3
"""
Smooth Tattoo Tracker
Addresses jumpy blur boxes and missed compass tattoo with temporal smoothing
"""

import cv2
import numpy as np
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrackedRegion:
    """Represents a tracked region across frames"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    age: int
    velocity: Tuple[float, float] = (0, 0)
    last_seen: int = 0
    region_type: str = "tattoo"

class SmoothTattooTracker:
    def __init__(self):
        """Initialize the smooth tracker with temporal consistency"""
        logger.info("SMOOTH TATTOO TRACKER - ENHANCED COVERAGE")
        logger.info("Temporal smoothing + comprehensive tattoo detection")
        logger.info("=" * 80)
        
        # Initialize OpenCV Face Detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("OpenCV face detection initialized")
        
        # Tracking parameters
        self.tracked_regions: Dict[int, TrackedRegion] = {}
        self.next_id = 0
        self.max_track_age = 30  # frames
        self.min_confidence = 0.3
        self.movement_threshold = 50  # pixels
        self.smoothing_factor = 0.7  # for position smoothing
        
        # Detection history for temporal consistency
        self.detection_history = deque(maxlen=5)
        
        # Statistics
        self.frame_count = 0
        self.total_faces = 0
        self.total_tattoos = 0
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces = [(x, y, w, h) for x, y, w, h in detected_faces]
        return faces
    
    def detect_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create skin mask using multiple color spaces"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # HSV skin detection
        lower_hsv = np.array([0, 20, 70])
        upper_hsv = np.array([20, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin detection (more robust)
        lower_ycrcb = np.array([0, 133, 77])
        upper_ycrcb = np.array([255, 173, 127])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        return skin_mask
    
    def detect_all_dark_regions(self, frame: np.ndarray, skin_mask: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect dark regions with expanded coverage for all tattoo sizes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple darkness thresholds for comprehensive coverage
        detections = []
        
        # Very dark regions (obvious tattoos like compass)
        very_dark_mask = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)[1]
        dark_skin_very = cv2.bitwise_and(very_dark_mask, skin_mask)
        
        # Medium dark regions (subtle tattoos)  
        medium_dark_mask = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)[1]
        dark_skin_medium = cv2.bitwise_and(medium_dark_mask, skin_mask)
        
        # Process very dark regions
        contours, _ = cv2.findContours(dark_skin_very, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 150:  # Lowered threshold for smaller tattoos
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                avg_darkness = 255 - np.mean(roi)
                size_score = min(area / 1200, 1.0)
                confidence = (avg_darkness / 255 * 0.8) + (size_score * 0.2)
                detections.append((x, y, w, h, min(confidence, 1.0)))
        
        # Process medium dark regions (more selective)
        contours, _ = cv2.findContours(dark_skin_medium, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 250:  # Slightly higher threshold for medium dark
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                avg_darkness = 255 - np.mean(roi)
                
                # Only include if significantly darker than surrounding skin
                if avg_darkness > 35:  # Must be reasonably dark
                    size_score = min(area / 1800, 1.0)
                    confidence = (avg_darkness / 255 * 0.6) + (size_score * 0.4)
                    detections.append((x, y, w, h, min(confidence, 0.9)))
        
        return detections
    
    def detect_enhanced_patterns(self, frame: np.ndarray, skin_mask: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Enhanced pattern detection for comprehensive tattoo coverage"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_skin = cv2.bitwise_and(gray, skin_mask)
        
        detections = []
        
        # Method 1: Enhanced edge detection with multiple thresholds
        edges_fine = cv2.Canny(gray_skin, 30, 120)  # More sensitive
        edges_coarse = cv2.Canny(gray_skin, 60, 180)  # Less sensitive
        edges_combined = cv2.bitwise_or(edges_fine, edges_coarse)
        
        # Method 2: Texture analysis using local binary patterns approximation
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        texture = cv2.filter2D(gray_skin, -1, kernel)
        texture_mask = cv2.threshold(np.abs(texture), 15, 255, cv2.THRESH_BINARY)[1]
        
        # Combine edge and texture information
        pattern_mask = cv2.bitwise_or(edges_combined, texture_mask)
        
        # Find pattern contours
        contours, _ = cv2.findContours(pattern_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 80 < area < 8000:  # Expanded size range
                x, y, w, h = cv2.boundingRect(contour)
                
                # Enhanced pattern analysis
                roi_pattern = pattern_mask[y:y+h, x:x+w]
                roi_gray = gray_skin[y:y+h, x:x+w]
                
                pattern_density = np.sum(roi_pattern > 0) / (w * h)
                avg_intensity = np.mean(roi_gray[roi_gray > 0]) if np.any(roi_gray > 0) else 0
                
                # More inclusive pattern detection
                if pattern_density > 0.08 and avg_intensity < 120:  # Lower thresholds
                    confidence = min(pattern_density * 1.5 + (120 - avg_intensity) / 120 * 0.5, 1.0)
                    detections.append((x, y, w, h, confidence))
        
        return detections
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracking(self, current_detections: List[Tuple[int, int, int, int, float, str]]) -> List[TrackedRegion]:
        """Update tracking with temporal smoothing"""
        # Age existing tracks
        for track_id in list(self.tracked_regions.keys()):
            self.tracked_regions[track_id].age += 1
            if self.tracked_regions[track_id].age > self.max_track_age:
                del self.tracked_regions[track_id]
        
        # Match current detections with existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for i, (x, y, w, h, conf, det_type) in enumerate(current_detections):
            best_match_id = None
            best_iou = 0.0
            
            for track_id, track in self.tracked_regions.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self.calculate_iou((x, y, w, h), track.bbox)
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track with smoothing
                track = self.tracked_regions[best_match_id]
                
                # Smooth position update
                old_x, old_y, old_w, old_h = track.bbox
                smooth_x = int(old_x * self.smoothing_factor + x * (1 - self.smoothing_factor))
                smooth_y = int(old_y * self.smoothing_factor + y * (1 - self.smoothing_factor))
                smooth_w = int(old_w * self.smoothing_factor + w * (1 - self.smoothing_factor))
                smooth_h = int(old_h * self.smoothing_factor + h * (1 - self.smoothing_factor))
                
                # Update velocity
                vel_x = smooth_x - old_x
                vel_y = smooth_y - old_y
                track.velocity = (vel_x, vel_y)
                
                # Update track
                track.bbox = (smooth_x, smooth_y, smooth_w, smooth_h)
                track.confidence = max(track.confidence * 0.9 + conf * 0.1, conf)
                track.age = 0
                track.last_seen = self.frame_count
                
                matched_tracks.add(best_match_id)
                matched_detections.add(i)
            
        # Create new tracks for unmatched detections
        for i, (x, y, w, h, conf, det_type) in enumerate(current_detections):
            if i not in matched_detections and conf > self.min_confidence:
                self.tracked_regions[self.next_id] = TrackedRegion(
                    bbox=(x, y, w, h),
                    confidence=conf,
                    age=0,
                    last_seen=self.frame_count,
                    region_type=det_type
                )
                self.next_id += 1
        
        # Return active tracks
        active_tracks = [track for track in self.tracked_regions.values() if track.age < 10]
        return active_tracks
    
    def process_frame(self, frame: np.ndarray, debug: bool = False) -> np.ndarray:
        """Process a single frame with smooth tracking"""
        self.frame_count += 1
        output_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        self.total_faces += len(faces)
        
        # Create skin mask
        skin_mask = self.detect_skin_mask(frame)
        
        # Detect tattoos using multiple methods
        current_detections = []
        
        # Method 1: Comprehensive dark regions (all tattoo sizes)
        dark_regions = self.detect_all_dark_regions(frame, skin_mask)
        for x, y, w, h, conf in dark_regions:
            current_detections.append((x, y, w, h, conf, "dark_region"))
        
        # Method 2: Enhanced pattern detection
        patterns = self.detect_enhanced_patterns(frame, skin_mask)
        for x, y, w, h, conf in patterns:
            current_detections.append((x, y, w, h, conf, "pattern"))
        
        # Update tracking
        active_tracks = self.update_tracking(current_detections)
        
        # Apply blur to tracked regions
        for track in active_tracks:
            x, y, w, h = track.bbox
            
            # Ensure bbox is within frame
            height, width = frame.shape[:2]
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w > 5 and h > 5:
                # Extract region
                roi = output_frame[y:y+h, x:x+w]
                
                # Apply Gaussian blur
                blur_size = max(15, min(w, h) // 3)
                if blur_size % 2 == 0:
                    blur_size += 1
                blurred_roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
                
                # Put blurred region back
                output_frame[y:y+h, x:x+w] = blurred_roi
                
                if debug:
                    # Draw tracking box with enhanced colors
                    color = (0, 255, 0) if track.region_type == "dark_region" else (0, 255, 255)
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(output_frame, f"{track.region_type[:4]}", (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Apply face blur
        for x, y, w, h in faces:
            # Ensure bbox is within frame
            height, width = frame.shape[:2]
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w > 5 and h > 5:
                roi = output_frame[y:y+h, x:x+w]
                blur_size = max(15, min(w, h) // 2)
                if blur_size % 2 == 0:
                    blur_size += 1
                blurred_roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
                output_frame[y:y+h, x:x+w] = blurred_roi
                
                if debug:
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(output_frame, "face", (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        self.total_tattoos += len(active_tracks)
        
        return output_frame
    
    def process_video(self, input_path: str, output_path: str, debug: bool = False):
        """Process video with smooth tracking"""
        logger.info(f"Processing: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"SMOOTH TRACKING MODE: Temporal consistency + compass detection")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames, {duration:.1f}s")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        start_time = time.time()
        logger.info("Starting smooth tracking with compass detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame, debug)
            out.write(processed_frame)
            
            # Progress update
            if self.frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                progress = (self.frame_count / total_frames) * 100
                eta = (total_frames - self.frame_count) / current_fps if current_fps > 0 else 0
                
                active_tracks = len([t for t in self.tracked_regions.values() if t.age < 10])
                
                logger.info(f"Progress: {progress:5.1f}% | Frame: {self.frame_count:6d}/{total_frames} | "
                           f"FPS: {current_fps:5.1f} | Faces: {len(self.detect_faces(frame)):2d} | "
                           f"Active Tracks: {active_tracks:3d} | ETA: {eta:4.0f}s")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        logger.info("=" * 80)
        logger.info("SMOOTH TRACKING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Frames processed: {self.frame_count}/{total_frames}")
        logger.info(f"Total faces detected: {self.total_faces}")
        logger.info(f"Total tattoo detections: {self.total_tattoos}")
        logger.info(f"Processing time: {elapsed:.1f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Enhanced coverage: Multiple thresholds + pattern detection!")
        logger.info(f"Temporal smoothing reduces jumpy blur boxes!")
        logger.info("=" * 80)
        logger.info("SUCCESS: Smooth tracking with better compass detection!")

def main():
    parser = argparse.ArgumentParser(description='Smooth Tattoo Tracker')
    parser.add_argument('input_video', help='Input video path')
    parser.add_argument('output_video', help='Output video path')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    
    args = parser.parse_args()
    
    try:
        tracker = SmoothTattooTracker()
        tracker.process_video(args.input_video, args.output_video, args.debug)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()