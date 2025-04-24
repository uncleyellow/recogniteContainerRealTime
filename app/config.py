import os
from typing import Dict
import cv2

# Cấu hình camera
RTSP_URLS: Dict[str, str] = {
    "CONG-VAO-DA-1": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/202/",
    "CONG-VAO-DA-2": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/102/",
    "CONG-RA-DA-1": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/302/",
    "CONG-RA-DA-2": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/402/"
}

# Cấu hình OCR và Detection
MOTION_THRESHOLD = 400
CONFIDENCE_THRESHOLD = 0.35
OCR_CONFIDENCE_THRESHOLD = 0.65
OCR_COOLDOWN = 0.5

# Cấu hình Google Sheets
SHEET_NAME = "ContainerTracking"
CREDENTIALS_FILE = "credentials.json"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Cấu hình hệ thống
LOG_FILE = os.path.join('logs', 'container_ocr.log')
FRAME_QUEUE_SIZE = 10
PORT = 8000
HOST = "0.0.0.0"

# Thư mục lưu trữ
STORAGE_DIR = "storage"
MODEL_DIR = "models"
LOG_DIR = "logs"

# Đảm bảo các thư mục tồn tại
for directory in [STORAGE_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist

# Thêm vào config.py
PERFORMANCE_CONFIG = {
    "frame_skip": 2,  # Bỏ qua frames để giảm tải CPU
    "max_resolution": (1920, 1080),  # Giới hạn độ phân giải
    "compression_quality": 80,  # Chất lượng nén JPEG
    "batch_processing": True,  # Xử lý nhiều frames cùng lúc
}

# Cải thiện xử lý frame trong detect.py
def process_frame(frame):
    if frame.shape[1] > PERFORMANCE_CONFIG["max_resolution"][0]:
        scale = PERFORMANCE_CONFIG["max_resolution"][0] / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame
