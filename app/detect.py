import cv2
import time
from datetime import datetime, timedelta
from ultralytics import YOLO
import easyocr
from collections import deque
import difflib
import re
import numpy as np
import logging
import os
import json
from pathlib import Path
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.sensor import get_sensor_manager
import queue
logger = logging.getLogger(__name__)
# Load models
model = YOLO("models/yolov8n.pt")
reader = easyocr.Reader(['en', 'vi'])  # Support both English and Vietnamese

# Cache for container codes and license plates
container_cache = deque(maxlen=100)
plate_cache = deque(maxlen=100)

# Detection parameters
MOTION_THRESHOLD = 400
CONFIDENCE_THRESHOLD = 0.35
OCR_CONFIDENCE_THRESHOLD = 0.65
OCR_COOLDOWN = 0.5
EMAIL_ALERT_THRESHOLD = 0.45  # Confidence threshold for email alerts

# Vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Vietnamese license plate pattern
VN_PLATE_PATTERN = r'^[0-9]{2}[A-Z][0-9]{4,5}$'

# Container code pattern
CONTAINER_PATTERN = r'^[A-Z]{4}\d{7}$'

# Email configuration (simulated)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'sender_password': 'your-app-password',
    'recipient_email': 'manager@company.com'
}

# 1. Thêm cấu hình cho kích thước container
CONTAINER_HEIGHT_RATIO = 0.5  # Tỷ lệ chiều cao container so với xe tải
MIN_CONTAINER_SIZE = (100, 50)  # Kích thước tối thiểu của container

# Thêm cấu hình cho lưu ảnh và thống kê
SAVE_DIR = "storage/containers"  # Thư mục lưu ảnh container
STATS_FILE = "storage/stats/detection_stats.json"  # File lưu thống kê
ALERT_THRESHOLD = 10  # Số lần lỗi liên tiếp trước khi cảnh báo
STATS_INTERVAL = 3600  # Thời gian tính thống kê (1 giờ)

# Tạo thư mục lưu trữ nếu chưa tồn tại
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)

# Class để quản lý thống kê và cảnh báo
class DetectionStats:
    def __init__(self):
        self.stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0,
            "last_hour_stats": {},
            "camera_status": {},
            "latest_detections": {
                "ENTRANCE": None,
                "EXIT": None
            },
            "runtime_params": {
                "MOTION_THRESHOLD": MOTION_THRESHOLD,
                "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
                "OCR_CONFIDENCE_THRESHOLD": OCR_CONFIDENCE_THRESHOLD,
                "OCR_COOLDOWN": OCR_COOLDOWN
            }
        }
        self.load_stats()

    def load_stats(self):
        """Load stats from file if exists"""
        try:
            if os.path.exists(STATS_FILE):
                with open(STATS_FILE, 'r') as f:
                    loaded_stats = json.load(f)
                    # Ensure latest_detections exists
                    if "latest_detections" not in loaded_stats:
                        loaded_stats["latest_detections"] = {
                            "ENTRANCE": None,
                            "EXIT": None
                        }
                    self.stats.update(loaded_stats)
        except Exception as e:
            logger.error(f"Error loading stats: {str(e)}")

    def save_stats(self):
        """Save stats to file"""
        try:
            os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
            with open(STATS_FILE, 'w') as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving stats: {str(e)}")

    def get_latest_detection(self, camera_name: str) -> dict:
        """Get the latest detection for a specific camera"""
        try:
            return self.stats["latest_detections"].get(camera_name)
        except Exception as e:
            logger.error(f"Error getting latest detection: {str(e)}")
            return None

    def update_detection(self, camera_name, success, confidence=0, detection_type="container"):
        """Update detection statistics"""
        try:
            current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
            if current_hour not in self.stats["last_hour_stats"]:
                self.stats["last_hour_stats"][current_hour] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_confidence": 0,
                    "container_detections": 0,
                    "plate_detections": 0
                }

            self.stats["total_detections"] += 1
            hour_stats = self.stats["last_hour_stats"][current_hour]
            hour_stats["total"] += 1

            if success:
                self.stats["successful_detections"] += 1
                hour_stats["successful"] += 1
                if detection_type == "container":
                    hour_stats["container_detections"] += 1
                else:
                    hour_stats["plate_detections"] += 1
                
                current_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (current_avg * (self.stats["successful_detections"] - 1) + confidence) / self.stats["successful_detections"]
                
                # Update latest detection
                self.stats["latest_detections"][camera_name] = {
                    "code": "",  # This will be updated by the capture endpoint
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "detection_type": detection_type
                }
            else:
                self.stats["failed_detections"] += 1
                hour_stats["failed"] += 1

            self.save_stats()
        except Exception as e:
            logger.error(f"Error updating detection stats: {str(e)}")

    def update_camera_status(self, camera_name, status, error_message=None):
        self.stats["camera_status"][camera_name] = {
            "status": status,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_message": error_message
        }
        self.save_stats()

    def get_runtime_params(self):
        return self.stats["runtime_params"]

    def update_runtime_params(self, params):
        self.stats["runtime_params"].update(params)
        self.save_stats()
        return self.stats["runtime_params"]

# Khởi tạo đối tượng thống kê
detection_stats = DetectionStats()

def send_alert_email(subject, message):
    """Simulate sending alert email"""
    try:
        # Print email details instead of actually sending
        print("\n" + "="*50)
        print("SIMULATED EMAIL ALERT")
        print("="*50)
        print(f"From: {EMAIL_CONFIG['sender_email']}")
        print(f"To: {EMAIL_CONFIG['recipient_email']}")
        print(f"Subject: {subject}")
        print("-"*50)
        print("Message:")
        print(message)
        print("="*50 + "\n")
        
        logger.info("Email alert simulated successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to simulate email: {str(e)}")
        return False

def is_valid_plate(text):
    """Validate Vietnamese license plate format"""
    return bool(re.match(VN_PLATE_PATTERN, text))

def is_valid_container_code(text):
    """Validate container code format"""
    return bool(re.match(CONTAINER_PATTERN, text))

def preprocess_image(image, detection_type="container"):
    """Enhanced image preprocessing for different detection types"""
    if image.size == 0:
        return None

    # Resize if too large
    max_width = 800
    if image.shape[1] > max_width:
        ratio = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(image.shape[0] * ratio)))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if detection_type == "plate":
        # Enhanced preprocessing for license plates
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
        sharp = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)
        binary = cv2.adaptiveThreshold(
            sharp, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        # Standard preprocessing for container codes
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
        sharp = cv2.addWeighted(contrast, 1.8, blurred, -0.8, 0)
        binary = cv2.adaptiveThreshold(
            sharp, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

    return binary

def detect_and_display(rtsp_url, upload_callback, camera_name="CAM", stop_event=None, frame_queue=None, detection_type="container", display_mode="web"):
    """Enhanced detection function supporting both container codes and license plates"""
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"[{camera_name}] Cannot connect to camera")
            return

        # Initialize variables
        frame_prev = None
        movement_detected = False
        last_ocr_time = 0
        best_ocr = {"text": "", "confidence": 0}
        error_count = 0
        last_success_time = time.time()
        last_upload_time = time.time()
        UPLOAD_INTERVAL = 5  # Upload to Excel every 5 seconds if new detection

        logger.info(f"Camera {camera_name} started in {display_mode} mode")

        while not (stop_event and stop_event.is_set()):
            try:
                ret, frame = cap.read()
                if not ret:
                    error_count += 1
                    if error_count >= ALERT_THRESHOLD:
                        logger.error(f"[{camera_name}] Camera not responding after {ALERT_THRESHOLD} attempts")
                        detection_stats.update_camera_status(camera_name, "error", "Camera not responding")
                        send_alert_email(
                            f"Camera Alert: {camera_name}",
                            f"Camera {camera_name} is not responding. Please check the connection."
                        )
                    continue

                error_count = 0
                detection_stats.update_camera_status(camera_name, "running")

                # Create display frame copy
                display_frame = frame.copy()
                frame_height, frame_width = frame.shape[:2]

                # Motion detection
                if frame_prev is not None:
                    # Convert frames to grayscale
                    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
                    gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate absolute difference
                    frame_diff = cv2.absdiff(gray_prev, gray_current)
                    
                    # Apply threshold
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    
                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw red boxes for motion
                    for contour in contours:
                        if cv2.contourArea(contour) > MOTION_THRESHOLD:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
                            movement_detected = True

                # Add timestamp
                cv2.putText(display_frame,
                           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2)

                # Put frame in queue for web streaming
                if frame_queue:
                    try:
                        frame_queue.put(display_frame, block=False)
                    except queue.Full:
                        pass  # Skip frame if queue is full

                # Process frame for detection if needed
                if time.time() - last_ocr_time > OCR_COOLDOWN:
                    # Run YOLO detection
                    results = model(frame)[0]
                    vehicle_regions = []

                    # Find vehicles in frame
                    for box in results.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if class_id in VEHICLE_CLASSES and conf > CONFIDENCE_THRESHOLD:
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                            
                            if detection_type == "plate":
                                # For license plates, focus on the front/rear of the vehicle
                                plate_height = (by2 - by1) // 4
                                plate_y1 = by1 + plate_height
                                plate_y2 = by1 + plate_height * 2
                                vehicle_regions.append({
                                    'box': (bx1, plate_y1, bx2, plate_y2),
                                    'conf': conf
                                })
                            else:
                                # For containers, focus on the upper part
                                container_height = (by2 - by1) // 2
                                container_y1 = max(0, by1 - container_height)
                                vehicle_regions.append({
                                    'box': (bx1, container_y1, bx2, by1),
                                    'conf': conf
                                })

                    # Process each vehicle region
                    for vehicle in vehicle_regions:
                        tx1, ty1, tx2, ty2 = vehicle['box']
                        region = frame[ty1:ty2, tx1:tx2]
                        
                        if region.size == 0:
                            continue

                        processed = preprocess_image(region, detection_type)
                        ocr_result = reader.readtext(processed)
                        
                        for result in ocr_result:
                            text, ocr_conf = result[1].strip().upper(), result[2]
                            
                            is_valid = (is_valid_plate(text) if detection_type == "plate" 
                                      else is_valid_container_code(text))
                            
                            if is_valid and ocr_conf > OCR_CONFIDENCE_THRESHOLD:
                                if ocr_conf > best_ocr["confidence"]:
                                    best_ocr = {
                                        "text": text,
                                        "confidence": ocr_conf,
                                        "box": (tx1, ty1, tx2, ty2)
                                    }
                                
                                # Draw detection box
                                color = (0, 255, 0) if ocr_conf > EMAIL_ALERT_THRESHOLD else (0, 0, 255)
                                cv2.rectangle(display_frame, 
                                            (tx1, ty1), 
                                            (tx2, ty2),
                                            color, 3)
                                
                                # Draw text background
                                text_size = cv2.getTextSize(f"{text} ({ocr_conf:.2f})", 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                                          0.8, 2)[0]
                                cv2.rectangle(display_frame,
                                            (tx1, ty1 - 35),
                                            (tx1 + text_size[0], ty1 - 5),
                                            (0, 0, 0),
                                            -1)
                                
                                # Display text
                                cv2.putText(display_frame,
                                          f"{text} ({ocr_conf:.2f})",
                                          (tx1, ty1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.8, (255, 255, 255), 2)
                                
                                # Send alert if confidence is low
                                if ocr_conf < EMAIL_ALERT_THRESHOLD:
                                    send_alert_email(
                                        f"Low Confidence Alert: {camera_name}",
                                        f"Low confidence detection ({ocr_conf:.2f}) for {text} on {camera_name}"
                                    )
                                
                                # Save detection image
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"{camera_name}_{text}_{timestamp}.jpg"
                                filepath = os.path.join(SAVE_DIR, filename)
                                
                                detection_img = frame[ty1:ty2, tx1:tx2]
                                cv2.imwrite(filepath, detection_img)

                                # Update stats
                                detection_stats.update_detection(
                                    camera_name, 
                                    True, 
                                    ocr_conf,
                                    detection_type
                                )

                                # Prepare data for upload
                                data = {
                                    "code": text,
                                    "confidence": round(ocr_conf * 100, 2),
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "image_path": filepath,
                                    "detection_type": detection_type
                                }
                                
                                # Auto-upload to Excel if enough time has passed
                                current_time = time.time()
                                if current_time - last_upload_time >= UPLOAD_INTERVAL:
                                    upload_callback([data], camera_name)
                                    last_upload_time = current_time
                                
                                logger.info(f"\nDetection successful:")
                                logger.info(f"Type: {detection_type}")
                                logger.info(f"Code: {text}")
                                logger.info(f"Confidence: {ocr_conf:.2f}")
                                logger.info(f"Image saved: {filepath}\n")

                    last_ocr_time = time.time()

                # Update previous frame
                frame_prev = frame.copy()

            except Exception as e:
                logger.error(f"[{camera_name}] Frame processing error: {str(e)}")
                detection_stats.update_detection(camera_name, False)
                error_count += 1

    except Exception as e:
        logger.error(f"[{camera_name}] Error: {str(e)}")
        detection_stats.update_camera_status(camera_name, "error", str(e))
    finally:
        if cap:
            cap.release()
        logger.info(f"Camera {camera_name} stopped")

def is_similar(text, cache):
    """Check if text is similar to any recent detection"""
    for item in cache:
        sim = difflib.SequenceMatcher(None, item["text"], text).ratio()
        if sim > 0.9 and time.time() - item["ts"] < 120:
            return True
    return False

class CameraManager:
    def __init__(self, camera_name, rtsp_url):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.reconnect_attempts = 0
        self.max_reconnects = 5
        self.reconnect_delay = 10  # seconds
        
    async def handle_connection_loss(self):
        while self.reconnect_attempts < self.max_reconnects:
            logger.warning(f"Attempting to reconnect {self.camera_name}...")
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                if cap.isOpened():
                    return cap
            except Exception as e:
                logger.error(f"Reconnection failed: {str(e)}")
            self.reconnect_attempts += 1
            await asyncio.sleep(self.reconnect_delay)
        raise ConnectionError(f"Failed to reconnect {self.camera_name}")
