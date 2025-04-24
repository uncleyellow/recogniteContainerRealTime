import os
from datetime import datetime, timedelta
from app.config import SAVE_DIR

class StorageManager:
    def __init__(self):
        self.cleanup_threshold = 0.9  # 90% disk usage
        
    def cleanup_old_images(self, days=30):
        """Xóa ảnh cũ hơn X ngày"""
        cutoff = datetime.now() - timedelta(days=days)
        for image in os.listdir(SAVE_DIR):
            if self.get_image_date(image) < cutoff:
                os.remove(os.path.join(SAVE_DIR, image))
                
    def compress_images(self):
        """Nén ảnh để tiết kiệm không gian"""
        for image in os.listdir(SAVE_DIR):
            self.compress_image(os.path.join(SAVE_DIR, image))
