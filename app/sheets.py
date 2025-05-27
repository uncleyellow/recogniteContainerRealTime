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
        """Connect to Google Sheets"""
        try:
            credentials = Credentials.from_service_account_file(
                'credentials.json',
                scopes=SCOPES
            )
            self.client = gspread.authorize(credentials)
            spreadsheet = self.client.open("ContainerTracking")
            self.sheet = spreadsheet.sheet1
            
            # Ensure headers exist
            if not self.sheet.row_values(1):
                self.sheet.append_row([
                    "code",
                    "detection_type",
                    "Text nhận diện được",
                    "Timestamp",
                    "confidence",
                    "camera",
                    "direction",
                    "image_path"
                ])
            
            logger.info("Connected to Google Sheets successfully")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Google Sheets: {str(e)}")
            return False

    def get_data_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get data within date range"""
        try:
            if not self.sheet:
                if not self.connect():
                    return []

            all_data = self.sheet.get_all_records()
            
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            
            filtered_data = []
            for record in all_data:
                try:
                    timestamp = record.get('Timestamp', '')
                    if not timestamp:
                        continue
                    
                    record_date = datetime.strptime(timestamp.split()[0], '%Y-%m-%d').date()
                    
                    if start <= record_date <= end:
                        filtered_data.append({
                            'code': record.get('code', ''),
                            'detection_type': record.get('detection_type', ''),
                            'Text nhận diện được': record.get('Text nhận diện được', ''),
                            'Timestamp': record.get('Timestamp', ''),
                            'confidence': float(record.get('confidence', 0)),
                            'camera': record.get('camera', ''),
                            'direction': record.get('direction', ''),
                            'image_path': record.get('image_path', '')
                        })
                except Exception as e:
                    logger.error(f"Error processing record: {str(e)}")
                    continue
            
            filtered_data.sort(key=lambda x: x['Timestamp'], reverse=True)
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error getting data: {str(e)}")
            return []

    def export_daily_data(self, date: str) -> pd.DataFrame:
        """Export data for a specific date"""
        try:
            data = self.get_data_by_date_range(date, date)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return pd.DataFrame()

    def upload_data(self, data_list: List[Dict[str, Any]], camera_name: str = "Unknown"):
        """Upload data to Google Sheets"""
        if not data_list:
            return

        for data in data_list:
            try:
                # Determine direction based on camera name
                direction = "IN" if "ENTRANCE" in camera_name.upper() else "OUT"
                
                row = [
                    data["code"],
                    data.get("detection_type", "container"),  # Default to container if not specified
                    data["code"],  # Text nhận diện được
                    data["timestamp"],
                    str(data["confidence"]),
                    camera_name,
                    direction,
                    data.get("image_path", "")  # Add image path
                ]

                if not self.sheet:
                    if not self.connect():
                        self.retry_queue.append((row, datetime.now()))
                        continue

                self.sheet.append_row(row)
                logger.info(f"[{camera_name}] Upload successful: {data['code']}")
                
            except Exception as e:
                logger.error(f"Error uploading data: {str(e)}")
                self.retry_queue.append((row, datetime.now()))

# Singleton instance
_sheet_manager = None

def get_sheet_manager() -> SheetManager:
    """Get the singleton sheet manager instance"""
    global _sheet_manager
    if _sheet_manager is None:
        _sheet_manager = SheetManager()
    return _sheet_manager

def upload_to_sheet(data_list: List[Dict[str, Any]], camera_name: str = "Unknown"):
    """Helper function to upload data"""
    sheet_manager = get_sheet_manager()
    if sheet_manager:
        sheet_manager.upload_data(data_list, camera_name)
