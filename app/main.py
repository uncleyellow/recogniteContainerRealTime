from fastapi import FastAPI
from app.detect import detect_and_display
from app.sheets import upload_to_sheet

app = FastAPI()

RTSP_URL = "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/202/"

@app.get("/")
def home():
    return {"status": "Container OCR FastAPI đang chạy"}

@app.get("/start")
def start_camera_ocr():
    detect_and_display(RTSP_URL, upload_to_sheet)
    return {"message": "Camera processing started"}
