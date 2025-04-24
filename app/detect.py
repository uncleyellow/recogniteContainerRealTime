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

# Load models
model = YOLO("models/yolov8n.pt")
reader = easyocr.Reader(['en'])

# Cache for container codes to avoid duplicates
container_cache = deque(maxlen=100)

# Movement detection parameters
MOTION_THRESHOLD = 400
CONFIDENCE_THRESHOLD = 0.35
OCR_CONFIDENCE_THRESHOLD = 0.65
OCR_COOLDOWN = 0.5

# Chỉ nhận diện xe tải (truck), không nhận diện ô tô con
TRUCK_CLASSES = [7]  # 7: truck trong YOLOv8
CONTAINER_CLASSES = [7]  # Chỉ tìm container trên xe tải

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
            "runtime_params": {
                "MOTION_THRESHOLD": MOTION_THRESHOLD,
                "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
                "OCR_CONFIDENCE_THRESHOLD": OCR_CONFIDENCE_THRESHOLD,
                "OCR_COOLDOWN": OCR_COOLDOWN
            }
        }
        self.load_stats()

    def load_stats(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    self.stats = json.load(f)
            except:
                pass

    def save_stats(self):
        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def update_detection(self, camera_name, success, confidence=0):
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        if current_hour not in self.stats["last_hour_stats"]:
            self.stats["last_hour_stats"] = {current_hour: {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "avg_confidence": 0
            }}

        self.stats["total_detections"] += 1
        hour_stats = self.stats["last_hour_stats"][current_hour]
        hour_stats["total"] += 1

        if success:
            self.stats["successful_detections"] += 1
            hour_stats["successful"] += 1
            # Cập nhật confidence trung bình
            current_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (current_avg * (self.stats["successful_detections"] - 1) + confidence) / self.stats["successful_detections"]
        else:
            self.stats["failed_detections"] += 1
            hour_stats["failed"] += 1

        self.save_stats()

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

# 2. Cải thiện hàm preprocess_image
def preprocess_image(image):
    # Thêm kiểm tra kích thước ảnh
    if image.size == 0 or image.shape[0] < MIN_CONTAINER_SIZE[1] or image.shape[1] < MIN_CONTAINER_SIZE[0]:
        return None
        
    # Resize ảnh nếu quá lớn
    max_width = 800
    if image.shape[1] > max_width:
        ratio = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(image.shape[0] * ratio)))

    # Các bước xử lý hiện tại...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

# 3. Cải thiện hàm is_valid_container_code
def is_valid_container_code(text):
    # Thêm các pattern phổ biến khác của mã container
    patterns = [
        r'[A-Z]{4}\d{7}',  # Pattern hiện tại
        r'[A-Z]{4}[0-9]{6}[A-Z]',  # Pattern thay thế
        r'[A-Z]{3}[UJZ]\d{7}'  # Pattern cho container đặc biệt
    ]
    
    # Kiểm tra các pattern tiêu chuẩn trước
    if any(re.fullmatch(pattern, text) for pattern in standard_patterns):
        return True
        
    # Thêm điều kiện cho các text khác
    # Text phải có ít nhất 4 ký tự và chỉ chứa chữ cái, số và khoảng trắng
    if len(text) >= 4 and re.match(r'^[A-Z0-9\s]+$', text):
        return True
        
    return False

# 4. Thêm xử lý lỗi và logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('container_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_similar(text):
    for item in container_cache:
        sim = difflib.SequenceMatcher(None, item["text"], text).ratio()
        if sim > 0.9 and time.time() - item["ts"] < 120:
            return True
    return False

def detect_and_display(rtsp_url, upload_callback, camera_name="CAM", stop_event=None, frame_queue=None, display_mode="web"):
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"[{camera_name}] Không thể kết nối đến camera")
            return

        # Thêm retry logic cho kết nối camera
        retry_count = 0
        max_retries = 3
        while not cap.isOpened() and retry_count < max_retries:
            logger.warning(f"[{camera_name}] Thử kết nối lại lần {retry_count + 1}")
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            retry_count += 1

        # Initialize variables
        frame_prev = None
        movement_detected = False
        last_ocr_time = 0
        best_ocr = {"text": "", "confidence": 0}

        # Thêm biến đếm lỗi
        error_count = 0
        last_success_time = time.time()

        while not (stop_event and stop_event.is_set()):
            try:
                ret, frame = cap.read()
                if not ret:
                    error_count += 1
                    if error_count >= ALERT_THRESHOLD:
                        logger.error(f"[{camera_name}] Camera không hoạt động sau {ALERT_THRESHOLD} lần thử")
                        detection_stats.update_camera_status(camera_name, "error", "Camera không phản hồi")
                    continue
                
                error_count = 0  # Reset error count khi đọc frame thành công
                detection_stats.update_camera_status(camera_name, "running")

                # Tạo bản sao của frame để vẽ
                display_frame = frame.copy()
                frame_height, frame_width = frame.shape[:2]

                # Chuyển frame sang grayscale để phát hiện chuyển động
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if frame_prev is None:
                    frame_prev = frame_gray
                    continue

                # Phát hiện chuyển động với độ nhạy cao hơn
                frame_diff = cv2.absdiff(frame_prev, frame_gray)
                frame_prev = frame_gray
                
                blur = cv2.GaussianBlur(frame_diff, (5, 5), 0)
                thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=4)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                current_time = time.time()
                has_movement = False

                # Chạy YOLO trên toàn bộ frame để phát hiện xe tải
                results = model(frame)[0]
                truck_regions = []

                # Tìm các xe tải trong frame
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if class_id in TRUCK_CLASSES and conf > CONFIDENCE_THRESHOLD:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        
                        # Tính toán vùng container (phía trên thùng xe)
                        container_height = (by2 - by1) // 2  # Chiều cao container khoảng 1/2 xe
                        container_y1 = max(0, by1 - container_height)  # Mở rộng lên trên
                        
                        truck_regions.append({
                            'truck_box': (bx1, by1, bx2, by2),
                            'container_box': (bx1, container_y1, bx2, by1),  # Vùng container phía trên
                            'conf': conf
                        })
                        
                        # Vẽ box cho xe tải
                        cv2.rectangle(display_frame, 
                                    (bx1, by1), 
                                    (bx2, by2),
                                    (0, 165, 255), 2)  # Màu cam cho xe tải
                        cv2.putText(display_frame,
                                  f"Truck ({conf:.2f})",
                                  (bx1, by1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (0, 165, 255), 2)

                # Xử lý từng vùng có chuyển động
                for contour in contours:
                    if cv2.contourArea(contour) < MOTION_THRESHOLD:
                        continue

                    has_movement = True
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Vẽ khung chuyển động
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Kiểm tra từng xe tải
                    for truck in truck_regions:
                        tx1, ty1, tx2, ty2 = truck['container_box']  # Lấy vùng container
                        
                        # Kiểm tra nếu chuyển động nằm trong vùng container
                        if (x >= tx1 and x + w <= tx2 and 
                            y >= ty1 and y + h <= ty2):
                            
                            # Cắt vùng container để OCR
                            container_region = frame[ty1:ty2, tx1:tx2]
                            
                            if container_region.size == 0:
                                continue

                            # Thực hiện OCR nếu đủ thời gian
                            if current_time - last_ocr_time >= OCR_COOLDOWN:
                                # Tiền xử lý ảnh
                                processed = preprocess_image(container_region)
                                
                                # Thực hiện OCR
                                ocr_result = reader.readtext(processed)
                                
                                for result in ocr_result:
                                    text, ocr_conf = result[1].strip().upper(), result[2]
                                    
                                    if is_valid_container_code(text) and ocr_conf > OCR_CONFIDENCE_THRESHOLD:
                                        if ocr_conf > best_ocr["confidence"]:
                                            best_ocr = {
                                                "text": text,
                                                "confidence": ocr_conf,
                                                "box": (tx1, ty1, tx2, ty2)
                                            }
                                        
                                        # Vẽ box xanh lá cho container
                                        cv2.rectangle(display_frame, 
                                                    (tx1, ty1), 
                                                    (tx2, ty2),
                                                    (0, 255, 0), 3)
                                        
                                        # Vẽ background đen cho text
                                        text_size = cv2.getTextSize(f"{text} ({ocr_conf:.2f})", 
                                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                                  0.8, 2)[0]
                                        cv2.rectangle(display_frame,
                                                    (tx1, ty1 - 35),
                                                    (tx1 + text_size[0], ty1 - 5),
                                                    (0, 0, 0),
                                                    -1)
                                        
                                        # Hiển thị mã container
                                        cv2.putText(display_frame,
                                                  f"{text} ({ocr_conf:.2f})",
                                                  (tx1, ty1 - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.8, (255, 255, 255), 2)
                                        
                                        last_ocr_time = current_time

                # Xử lý kết thúc chuyển động và gửi dữ liệu
                if has_movement:
                    movement_detected = True
                elif movement_detected and not has_movement:
                    if best_ocr["confidence"] > 0:
                        if not is_similar(best_ocr["text"]):
                            container_cache.append({
                                "text": best_ocr["text"], 
                                "ts": current_time
                            })
                            
                            # Lưu ảnh container
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            container_filename = f"{camera_name}_{best_ocr['text']}_{timestamp}.jpg"
                            container_path = os.path.join(SAVE_DIR, container_filename)
                            
                            # Cắt và lưu ảnh container
                            tx1, ty1, tx2, ty2 = best_ocr["box"]
                            container_img = frame[ty1:ty2, tx1:tx2]
                            cv2.imwrite(container_path, container_img)

                            # Cập nhật thống kê
                            detection_stats.update_detection(
                                camera_name, 
                                True, 
                                best_ocr["confidence"]
                            )

                            # Thêm đường dẫn ảnh vào data
                            data = {
                                "container_code": best_ocr["text"],
                                "confidence": round(best_ocr["confidence"] * 100, 2),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "image_path": container_path
                            }
                            upload_callback([data], camera_name)
                    
                    best_ocr = {"text": "", "confidence": 0}
                    movement_detected = False

                # Thêm timestamp vào frame
                cv2.putText(display_frame,
                           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2)

                # Chỉ hiển thị trên terminal nếu mode là "both"
                if display_mode == "both":
                    cv2.imshow(f"{camera_name} OCR", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Chỉ đưa frame vào queue để hiển thị trên web
                    if frame_queue is not None:
                        try:
                            if frame_queue.full():
                                frame_queue.get_nowait()
                            frame_queue.put_nowait(display_frame)
                        except:
                            pass

            except Exception as e:
                logger.error(f"[{camera_name}] Lỗi xử lý frame: {str(e)}")
                detection_stats.update_detection(camera_name, False)
                error_count += 1

    except Exception as e:
        logger.error(f"[{camera_name}] Lỗi: {str(e)}")
        detection_stats.update_camera_status(camera_name, "error", str(e))
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

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
