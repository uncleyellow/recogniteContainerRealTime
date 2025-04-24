import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import logging
import time
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

class SheetManager:
    def __init__(self):
        self.client = None
        self.sheet = None
        self.upload_cache = []
        self.retry_queue = []
        self.connect()

    def connect(self):
        """Kết nối với Google Sheets"""
        try:
            credentials = Credentials.from_service_account_file(
                'credentials.json',
                scopes=SCOPES
            )
            self.client = gspread.authorize(credentials)
            # Mở spreadsheet bằng URL hoặc key
            spreadsheet = self.client.open("ContainerTracking")  # Thay tên thực tế của sheet
            self.sheet = spreadsheet.sheet1  # hoặc chọn sheet cụ thể
            
            # Đảm bảo headers tồn tại
            if not self.sheet.row_values(1):
                self.sheet.append_row([
                    "container_code",
                    "Text nhận diện được",
                    "Timestamp",
                    "confidence",
                    "camera",
                    "direction"
                ])
            
            logger.info("Kết nối Google Sheets thành công")
            return True
        except Exception as e:
            logger.error(f"Lỗi kết nối Google Sheets: {str(e)}")
            return False

    def get_data_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Lấy dữ liệu trong khoảng thời gian"""
        try:
            if not self.sheet:
                if not self.connect():
                    return []

            # Lấy tất cả dữ liệu
            all_data = self.sheet.get_all_records()
            
            # Chuyển đổi chuỗi ngày thành đối tượng datetime để so sánh
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            
            # Lọc dữ liệu theo ngày
            filtered_data = []
            for record in all_data:
                try:
                    timestamp = record.get('Timestamp', '')
                    if not timestamp:
                        continue
                    
                    record_date = datetime.strptime(timestamp.split()[0], '%Y-%m-%d').date()
                    
                    if start <= record_date <= end:
                        filtered_data.append({
                            'container_code': record.get('container_code', ''),
                            'Text nhận diện được': record.get('Text nhận diện được', ''),
                            'Timestamp': record.get('Timestamp', ''),
                            'confidence': float(record.get('confidence', 0)),
                            'camera': record.get('camera', ''),
                            'direction': record.get('direction', '')
                        })
                except Exception as e:
                    logger.error(f"Lỗi xử lý bản ghi: {str(e)}")
                    continue
            
            # Sắp xếp theo thời gian giảm dần
            filtered_data.sort(key=lambda x: x['Timestamp'], reverse=True)
            return filtered_data
            
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu: {str(e)}")
            return []

    def export_daily_data(self, date: str) -> pd.DataFrame:
        """Xuất dữ liệu của một ngày cụ thể"""
        try:
            data = self.get_data_by_date_range(date, date)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Lỗi xuất dữ liệu: {str(e)}")
            return pd.DataFrame()

    def upload_data(self, data_list: List[Dict[str, Any]], camera_name: str = "Unknown"):
        """Upload dữ liệu lên Google Sheets"""
        if not data_list:
            return

        for data in data_list:
            try:
                # Xác định hướng dựa trên tên camera
                direction = "IN" if "VAO" in camera_name.upper() else "OUT"
                
                row = [
                    data["container_code"],
                    data["container_code"],  # Text nhận diện được
                    data["timestamp"],
                    str(data["confidence"]),
                    camera_name,
                    direction
                ]

                if not self.sheet:
                    if not self.connect():
                        self.retry_queue.append((row, datetime.now()))
                        continue

                self.sheet.append_row(row)
                logger.info(f"[{camera_name}] Upload thành công: {data['container_code']}")
                
            except Exception as e:
                logger.error(f"Lỗi upload dữ liệu: {str(e)}")
                self.retry_queue.append((row, datetime.now()))

# Singleton instance
_sheet_manager = None

def get_sheet_manager() -> SheetManager:
    global _sheet_manager
    if _sheet_manager is None:
        _sheet_manager = SheetManager()
    return _sheet_manager

def upload_to_sheet(data_list: List[Dict[str, Any]], camera_name: str = "Unknown"):
    """Helper function để upload dữ liệu"""
    sheet_manager = get_sheet_manager()
    if sheet_manager:
        sheet_manager.upload_data(data_list, camera_name)
