from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread, Event
from typing import Dict, Any
from app.detect import detect_and_display, detection_stats
from app.sheets import upload_to_sheet, get_sheet_manager
import uvicorn
import logging
from datetime import datetime
import cv2
import queue
import numpy as np
from logging.handlers import RotatingFileHandler
from app.config import LOG_FILE
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import psutil

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Container OCR System",
    description="Hệ thống nhận diện container tự động",
    version="1.0.0"
)

# Cho phép CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Quản lý trạng thái camera và frame queues
camera_threads: Dict[str, Thread] = {}
camera_states: Dict[str, bool] = {}
stop_events: Dict[str, Event] = {}
frame_queues: Dict[str, queue.Queue] = {}

# Cấu hình RTSP
RTSP_URLS = {
    "CONG-VAO-DA-1": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/202/",
#    "CONG-VAO-DA-2": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/201/",
#    "CAM2": "rtsp://user:pass@ip2/Streaming/Channels/2/",
#    "CAM3": "rtsp://user:pass@ip3/Streaming/Channels/3/",
#    "CAM4": "rtsp://user:pass@ip4/Streaming/Channels/4/"
}

# Add utility functions
START_TIME = datetime.now()

def get_uptime():
    return str(datetime.now() - START_TIME)

def get_total_containers():
    return detection_stats.stats["total_detections"]

def get_containers_per_camera():
    return detection_stats.stats["camera_status"]

def get_recent_detections(limit=10):
    return detection_stats.stats.get("last_hour_stats", {})

def generate_frames(camera_name: str):
    """Generator để stream frames"""
    if not camera_states.get(camera_name, False):
        logger.error(f"Camera {camera_name} chưa được khởi động")
        return
        
    while camera_states.get(camera_name, False):
        try:
            if camera_name not in frame_queues:
                logger.error(f"Không tìm thấy queue cho camera {camera_name}")
                break
                
            frame = frame_queues[camera_name].get(timeout=1.0)
            if frame is None:
                continue
                
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if buffer is None:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Lỗi streaming camera {camera_name}: {str(e)}")
            break

@app.get("/video_feed/{camera_name}")
async def video_feed(camera_name: str):
    """Stream video từ camera"""
    try:
        if camera_name not in RTSP_URLS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_name} không tồn tại")
        
        if not camera_states.get(camera_name, False):
            # Tự động khởi động camera nếu chưa chạy
            await start_camera(camera_name)
        
        return StreamingResponse(
            generate_frames(camera_name),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except Exception as e:
        logger.error(f"Lỗi stream camera {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/view")
async def view(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/")
async def home():
    """Kiểm tra trạng thái API"""
    return {
        "status": "Container OCR API is running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": {
            name: "Running" if camera_states.get(name, False) else "Stopped"
            for name in RTSP_URLS
        }
    }

@app.get("/start/{camera_name}")
async def start_camera(camera_name: str):
    """Khởi động một camera cụ thể"""
    if camera_name not in RTSP_URLS:
        raise HTTPException(status_code=404, detail=f"Camera {camera_name} không tồn tại")
    
    if camera_states.get(camera_name, False):
        return {"message": f"Camera {camera_name} đã đang chạy"}
    
    try:
        stop_events[camera_name] = Event()
        thread = Thread(
            target=camera_thread_wrapper,
            args=(camera_name, RTSP_URLS[camera_name], stop_events[camera_name]),
            daemon=True
        )
        thread.start()
        
        camera_threads[camera_name] = thread
        camera_states[camera_name] = True
        
        logger.info(f"Đã khởi động camera {camera_name}")
        return {"message": f"Đã khởi động camera {camera_name}"}
    
    except Exception as e:
        logger.error(f"Lỗi khi khởi động camera {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stop/{camera_name}")
async def stop_camera(camera_name: str):
    """Dừng một camera cụ thể"""
    if camera_name not in RTSP_URLS:
        raise HTTPException(status_code=404, detail=f"Camera {camera_name} không tồn tại")
    
    if not camera_states.get(camera_name, False):
        return {"message": f"Camera {camera_name} đã dừng"}
    
    try:
        stop_events[camera_name].set()
        camera_states[camera_name] = False
        logger.info(f"Đã dừng camera {camera_name}")
        return {"message": f"Đã dừng camera {camera_name}"}
    
    except Exception as e:
        logger.error(f"Lỗi khi dừng camera {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/start")
async def start_all():
    """Khởi động tất cả camera"""
    responses = []
    for camera_name in RTSP_URLS:
        try:
            response = await start_camera(camera_name)
            responses.append({camera_name: response["message"]})
        except Exception as e:
            responses.append({camera_name: f"Lỗi: {str(e)}"})
    
    return {"results": responses}

@app.get("/stop")
async def stop_all():
    """Dừng tất cả camera"""
    responses = []
    for camera_name in RTSP_URLS:
        try:
            response = await stop_camera(camera_name)
            responses.append({camera_name: response["message"]})
        except Exception as e:
            responses.append({camera_name: f"Lỗi: {str(e)}"})
    
    return {"results": responses}

@app.get("/status")
async def get_status():
    """Lấy trạng thái chi tiết của hệ thống"""
    return {
        "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": {
            name: {
                "status": "Running" if camera_states.get(name, False) else "Stopped",
                "url": RTSP_URLS[name],
                "thread_alive": camera_threads.get(name, None) is not None 
                               and camera_threads[name].is_alive() 
                               if name in camera_threads else False
            }
            for name in RTSP_URLS
        }
    }

@app.get("/stats")
async def get_stats():
    """Lấy thống kê hệ thống"""
    stats = {
        "system_uptime": get_uptime(),
        "total_containers_detected": get_total_containers(),
        "containers_per_camera": get_containers_per_camera(),
        "recent_detections": get_recent_detections(limit=10)
    }
    return stats

@app.get("/export/{date}")
async def export_data(date: str):
    """API endpoint để xuất dữ liệu theo ngày"""
    try:
        sheet_manager = get_sheet_manager()
        if not sheet_manager:
            raise HTTPException(status_code=500, detail="Sheet manager chưa được khởi tạo")
            
        data = sheet_manager.export_daily_data(date)
        if data.empty:
            raise HTTPException(status_code=404, detail="Không có dữ liệu cho ngày này")
            
        return StreamingResponse(
            iter([data.to_csv(index=False)]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=container_data_{date}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/sheet-data")
async def get_sheet_data(start_date: str = None, end_date: str = None):
    """API endpoint để lấy dữ liệu từ Google Sheets"""
    try:
        sheet_manager = get_sheet_manager()
        if not sheet_manager:
            return JSONResponse(
                status_code=500,
                content={"message": "Sheet manager chưa được khởi tạo"}
            )
        
        # Validate dates
        if start_date and end_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Định dạng ngày không hợp lệ. Sử dụng YYYY-MM-DD"}
                )
        
        # Nếu không có ngày, sử dụng ngày hiện tại
        if not start_date or not end_date:
            today = datetime.now().strftime('%Y-%m-%d')
            start_date = today
            end_date = today
        
        data = sheet_manager.get_data_by_date_range(start_date, end_date)
        
        return JSONResponse(
            content={
                "data": data,
                "total": len(data),
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu sheet: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Lỗi server: {str(e)}",
                "data": [],
                "total": 0
            }
        )

def camera_thread_wrapper(camera_name: str, rtsp_url: str, stop_event: Event):
    """Wrapper để xử lý camera thread với khả năng dừng"""
    try:
        # Khởi tạo queue cho camera này
        frame_queues[camera_name] = queue.Queue(maxsize=10)
        detect_and_display(rtsp_url, upload_to_sheet, camera_name, stop_event, frame_queues[camera_name])
    except Exception as e:
        logger.error(f"Lỗi trong camera thread {camera_name}: {str(e)}")
        camera_states[camera_name] = False
    finally:
        camera_states[camera_name] = False
        if camera_name in frame_queues:
            del frame_queues[camera_name]

router = APIRouter()

class RuntimeParams(BaseModel):
    MOTION_THRESHOLD: int = None
    CONFIDENCE_THRESHOLD: float = None
    OCR_CONFIDENCE_THRESHOLD: float = None
    OCR_COOLDOWN: float = None

@router.get("/api/stats")
async def get_detection_stats():
    """Lấy thống kê nhận dạng"""
    return detection_stats.stats

@router.get("/api/params")
async def get_runtime_params():
    """Lấy các thông số hiện tại"""
    return detection_stats.get_runtime_params()

@router.post("/api/params")
async def update_runtime_params(params: RuntimeParams):
    """Cập nhật các thông số trong runtime"""
    global MOTION_THRESHOLD, CONFIDENCE_THRESHOLD, OCR_CONFIDENCE_THRESHOLD, OCR_COOLDOWN
    
    update_dict = {}
    if params.MOTION_THRESHOLD is not None:
        MOTION_THRESHOLD = params.MOTION_THRESHOLD
        update_dict["MOTION_THRESHOLD"] = params.MOTION_THRESHOLD
    if params.CONFIDENCE_THRESHOLD is not None:
        CONFIDENCE_THRESHOLD = params.CONFIDENCE_THRESHOLD
        update_dict["CONFIDENCE_THRESHOLD"] = params.CONFIDENCE_THRESHOLD
    if params.OCR_CONFIDENCE_THRESHOLD is not None:
        OCR_CONFIDENCE_THRESHOLD = params.OCR_CONFIDENCE_THRESHOLD
        update_dict["OCR_CONFIDENCE_THRESHOLD"] = params.OCR_CONFIDENCE_THRESHOLD
    if params.OCR_COOLDOWN is not None:
        OCR_COOLDOWN = params.OCR_COOLDOWN
        update_dict["OCR_COOLDOWN"] = params.OCR_COOLDOWN

    return detection_stats.update_runtime_params(update_dict)

@router.get("/api/camera-status")
async def get_camera_status():
    """Lấy trạng thái của các camera"""
    return detection_stats.stats["camera_status"]

@app.get("/api/system/health")
async def system_health():
    return {
        "status": "healthy",
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_usage": psutil.cpu_percent(),
        "uptime": get_uptime(),
        "storage": {
            "total": get_storage_info()["total"],
            "used": get_storage_info()["used"]
        }
    }

@app.get("/api/analytics/daily")
async def daily_analytics(date: str = None):
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    return {
        "total_containers": get_daily_containers(date),
        "accuracy_rate": get_daily_accuracy(date),
        "peak_hours": get_peak_hours(date),
        "camera_performance": get_camera_stats(date)
    }

def get_storage_info():
    disk = psutil.disk_usage('/')
    return {
        "total": disk.total / (1024 * 1024 * 1024),  # Convert to GB
        "used": disk.used / (1024 * 1024 * 1024),    # Convert to GB
        "free": disk.free / (1024 * 1024 * 1024)     # Convert to GB
    }

def get_daily_containers(date: str):
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    return len(data)

def get_daily_accuracy(date: str):
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    if not data:
        return 0
    return sum(record['confidence'] for record in data) / len(data)

def get_peak_hours(date: str):
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    hours = {}
    for record in data:
        hour = datetime.strptime(record['Timestamp'], '%Y-%m-%d %H:%M:%S').hour
        hours[hour] = hours.get(hour, 0) + 1
    return hours

def get_camera_stats(date: str):
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    stats = {}
    for record in data:
        camera = record['camera']
        stats[camera] = stats.get(camera, 0) + 1
    return stats

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# Add after router = APIRouter()
app.include_router(router)