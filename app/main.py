from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread, Event
from typing import Dict, Any
from app.detect import detect_and_display, detection_stats, is_valid_plate, is_valid_container_code, preprocess_image
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
from app.sensor import get_sensor_manager
from ultralytics import YOLO
import easyocr
import os

# Configure logging
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

# Load models
model = YOLO("models/yolov8n.pt")
reader = easyocr.Reader(['en', 'vi'])  # Support both English and Vietnamese

# Detection parameters
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CONFIDENCE_THRESHOLD = 0.35
OCR_CONFIDENCE_THRESHOLD = 0.65
OCR_COOLDOWN = 0.5
EMAIL_ALERT_THRESHOLD = 0.45

# Create save directory if not exists
SAVE_DIR = "storage/containers"
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI(
    title="Container & License Plate OCR System",
    description="Hệ thống nhận diện container và biển số xe tự động",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Camera and frame queue management
camera_threads: Dict[str, Thread] = {}
camera_states: Dict[str, bool] = {}
stop_events: Dict[str, Event] = {}
frame_queues: Dict[str, queue.Queue] = {}

# RTSP configuration
RTSP_URLS = {
    "ENTRANCE": {
        "url": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/301/",
        "type": "plate"  # License plate detection
    },
    "EXIT": {
        "url": "rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/201/",
        "type": "container"  # Container code detection
    }
}

# Utility functions
START_TIME = datetime.now()

def get_uptime():
    return str(datetime.now() - START_TIME)

def get_total_detections():
    return {
        "total": detection_stats.stats["total_detections"],
        "containers": detection_stats.stats.get("container_detections", 0),
        "plates": detection_stats.stats.get("plate_detections", 0)
    }

def get_detections_per_camera():
    return detection_stats.stats["camera_status"]

def get_recent_detections(limit=10):
    return detection_stats.stats.get("last_hour_stats", {})

def generate_frames(camera_name: str):
    """Frame generator for streaming"""
    if not camera_states.get(camera_name, False):
        logger.error(f"Camera {camera_name} is not running")
        return
        
    while camera_states.get(camera_name, False):
        try:
            if camera_name not in frame_queues:
                logger.error(f"No queue found for camera {camera_name}")
                break
                
            frame = frame_queues[camera_name].get(timeout=1.0)
            if frame is None:
                continue
                
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if buffer is None:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Streaming error for camera {camera_name}: {str(e)}")
            break

@app.get("/video_feed/{camera_name}")
async def video_feed(camera_name: str):
    """Stream video from camera"""
    try:
        if camera_name not in RTSP_URLS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_name} not found")
        
        if not camera_states.get(camera_name, False):
            await start_camera(camera_name)
        
        return StreamingResponse(
            generate_frames(camera_name),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except Exception as e:
        logger.error(f"Stream error for camera {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/view")
async def view(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cameras": RTSP_URLS
    })

@app.get("/")
async def home():
    """API status check"""
    return {
        "status": "Container & License Plate OCR API is running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": {
            name: {
                "status": "Running" if camera_states.get(name, False) else "Stopped",
                "type": config["type"]
            }
            for name, config in RTSP_URLS.items()
        }
    }

@app.get("/start/{camera_name}")
async def start_camera(camera_name: str):
    """Start a specific camera"""
    if camera_name not in RTSP_URLS:
        raise HTTPException(status_code=404, detail=f"Camera {camera_name} not found")
    
    if camera_states.get(camera_name, False):
        return {"message": f"Camera {camera_name} is already running"}
    
    try:
        stop_events[camera_name] = Event()
        thread = Thread(
            target=camera_thread_wrapper,
            args=(camera_name, RTSP_URLS[camera_name]["url"], stop_events[camera_name]),
            kwargs={"detection_type": RTSP_URLS[camera_name]["type"]},
            daemon=True
        )
        thread.start()
        
        camera_threads[camera_name] = thread
        camera_states[camera_name] = True
        
        logger.info(f"Started camera {camera_name}")
        return {"message": f"Started camera {camera_name}"}
    
    except Exception as e:
        logger.error(f"Error starting camera {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stop/{camera_name}")
async def stop_camera(camera_name: str):
    """Stop a specific camera"""
    if camera_name not in RTSP_URLS:
        raise HTTPException(status_code=404, detail=f"Camera {camera_name} not found")
    
    if not camera_states.get(camera_name, False):
        return {"message": f"Camera {camera_name} is already stopped"}
    
    try:
        stop_events[camera_name].set()
        camera_states[camera_name] = False
        logger.info(f"Stopped camera {camera_name}")
        return {"message": f"Stopped camera {camera_name}"}
    
    except Exception as e:
        logger.error(f"Error stopping camera {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/start")
async def start_all():
    """Start all cameras"""
    responses = []
    for camera_name in RTSP_URLS:
        try:
            response = await start_camera(camera_name)
            responses.append({camera_name: response["message"]})
        except Exception as e:
            responses.append({camera_name: f"Error: {str(e)}"})
    
    return {"results": responses}

@app.get("/stop")
async def stop_all():
    """Stop all cameras"""
    responses = []
    for camera_name in RTSP_URLS:
        try:
            response = await stop_camera(camera_name)
            responses.append({camera_name: response["message"]})
        except Exception as e:
            responses.append({camera_name: f"Error: {str(e)}"})
    
    return {"results": responses}

@app.get("/status")
async def get_status():
    """Get detailed system status"""
    return {
        "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": {
            name: {
                "status": "Running" if camera_states.get(name, False) else "Stopped",
                "type": config["type"],
                "url": config["url"],
                "thread_alive": camera_threads.get(name, None) is not None 
                               and camera_threads[name].is_alive() 
                               if name in camera_threads else False
            }
            for name, config in RTSP_URLS.items()
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "system_uptime": get_uptime(),
        "total_detections": get_total_detections(),
        "detections_per_camera": get_detections_per_camera(),
        "recent_detections": get_recent_detections(limit=10)
    }
    return stats

@app.get("/export/{date}")
async def export_data(date: str):
    """Export data for a specific date"""
    try:
        sheet_manager = get_sheet_manager()
        if not sheet_manager:
            raise HTTPException(status_code=500, detail="Sheet manager not initialized")
            
        data = sheet_manager.export_daily_data(date)
        if data.empty:
            raise HTTPException(status_code=404, detail="No data for this date")
            
        return StreamingResponse(
            iter([data.to_csv(index=False)]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=detection_data_{date}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/sheet-data")
async def get_sheet_data(start_date: str = None, end_date: str = None):
    """Get data from Google Sheets"""
    try:
        sheet_manager = get_sheet_manager()
        if not sheet_manager:
            return JSONResponse(
                status_code=500,
                content={"message": "Sheet manager not initialized"}
            )
        
        # Validate dates
        if start_date and end_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Invalid date format. Use YYYY-MM-DD"}
                )
        
        # Use current date if not specified
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
        logger.error(f"Error getting sheet data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Server error: {str(e)}",
                "data": [],
                "total": 0
            }
        )

def camera_thread_wrapper(camera_name: str, rtsp_url: str, stop_event: Event, detection_type: str = "container"):
    """Camera thread wrapper with detection type support"""
    try:
        frame_queues[camera_name] = queue.Queue(maxsize=10)
        detect_and_display(
            rtsp_url, 
            upload_to_sheet, 
            camera_name, 
            stop_event, 
            frame_queues[camera_name],
            detection_type=detection_type
        )
    except Exception as e:
        logger.error(f"Error in camera thread {camera_name}: {str(e)}")
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
    EMAIL_ALERT_THRESHOLD: float = None

@router.get("/api/stats")
async def get_detection_stats():
    """Get detection statistics"""
    return detection_stats.stats

@router.get("/api/params")
async def get_runtime_params():
    """Get current parameters"""
    return detection_stats.get_runtime_params()

@router.post("/api/params")
async def update_runtime_params(params: RuntimeParams):
    """Update runtime parameters"""
    update_dict = {}
    for param, value in params.dict().items():
        if value is not None:
            update_dict[param] = value

    return detection_stats.update_runtime_params(update_dict)

@router.get("/api/camera-status")
async def get_camera_status():
    """Get camera status"""
    return detection_stats.stats["camera_status"]

@app.get("/api/system/health")
async def system_health():
    """Get system health status"""
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
    """Get daily analytics"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    return {
        "total_detections": get_daily_detections(date),
        "accuracy_rate": get_daily_accuracy(date),
        "peak_hours": get_peak_hours(date),
        "camera_performance": get_camera_stats(date)
    }

def get_storage_info():
    """Get storage information"""
    disk = psutil.disk_usage('/')
    return {
        "total": disk.total / (1024 * 1024 * 1024),  # Convert to GB
        "used": disk.used / (1024 * 1024 * 1024),    # Convert to GB
        "free": disk.free / (1024 * 1024 * 1024)     # Convert to GB
    }

def get_daily_detections(date: str):
    """Get daily detection counts"""
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    return {
        "total": len(data),
        "containers": len([d for d in data if d.get("detection_type") == "container"]),
        "plates": len([d for d in data if d.get("detection_type") == "plate"])
    }

def get_daily_accuracy(date: str):
    """Get daily accuracy rate"""
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    if not data:
        return 0
    return sum(record['confidence'] for record in data) / len(data)

def get_peak_hours(date: str):
    """Get peak detection hours"""
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    hours = {}
    for record in data:
        hour = datetime.strptime(record['Timestamp'], '%Y-%m-%d %H:%M:%S').hour
        hours[hour] = hours.get(hour, 0) + 1
    return hours

def get_camera_stats(date: str):
    """Get camera performance statistics"""
    sheet_manager = get_sheet_manager()
    data = sheet_manager.get_data_by_date_range(date, date)
    stats = {}
    for record in data:
        camera = record['camera']
        if camera not in stats:
            stats[camera] = {
                "total": 0,
                "containers": 0,
                "plates": 0,
                "avg_confidence": 0
            }
        stats[camera]["total"] += 1
        if record.get("detection_type") == "container":
            stats[camera]["containers"] += 1
        else:
            stats[camera]["plates"] += 1
        stats[camera]["avg_confidence"] = (
            (stats[camera]["avg_confidence"] * (stats[camera]["total"] - 1) + 
             record["confidence"]) / stats[camera]["total"]
        )
    return stats

@app.get("/detection_results/{camera_name}")
async def get_detection_results(camera_name: str):
    """Get latest detection results for a camera"""
    try:
        if camera_name not in RTSP_URLS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_name} not found")
            
        # Get the latest detection from detection_stats
        latest_detection = detection_stats.get_latest_detection(camera_name)
        
        if latest_detection:
            return {
                "hasNewDetection": True,
                "code": latest_detection.get("code", ""),
                "confidence": latest_detection.get("confidence", 0),
                "timestamp": latest_detection.get("timestamp", ""),
                "detection_type": latest_detection.get("detection_type", "")
            }
        else:
            return {
                "hasNewDetection": False
            }
            
    except Exception as e:
        logger.error(f"Error getting detection results for {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capture/{camera_name}")
async def capture_image(camera_name: str):
    """Capture and process image from camera, return all OCR results with green boxes and overlay text"""
    try:
        if camera_name not in RTSP_URLS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_name} not found")
        if camera_name not in frame_queues:
            raise HTTPException(status_code=400, detail="Camera not running")
        try:
            frame = frame_queues[camera_name].get(timeout=1.0)
        except queue.Empty:
            raise HTTPException(status_code=400, detail="No frame available")
        detection_type = RTSP_URLS[camera_name]["type"]
        results = model(frame)[0]
        ocr_results_list = []
        # Process each detection
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id in VEHICLE_CLASSES and conf > CONFIDENCE_THRESHOLD:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                if detection_type == "plate":
                    plate_height = (by2 - by1) // 4
                    plate_y1 = by1 + plate_height
                    plate_y2 = by1 + plate_height * 2
                    region = frame[plate_y1:plate_y2, bx1:bx2]
                else:
                    container_height = (by2 - by1) // 2
                    container_y1 = max(0, by1 - container_height)
                    region = frame[container_y1:by1, bx1:bx2]
                if region.size == 0:
                    continue
                processed = preprocess_image(region, detection_type)
                ocr_result = reader.readtext(processed)
                for result in ocr_result:
                    text, ocr_conf = result[1].strip().upper(), result[2]
                    ocr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Vẽ box xanh và overlay text cho mọi kết quả OCR
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                    text_overlay = f"{text} ({ocr_conf:.2f})"
                    text_size = cv2.getTextSize(text_overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (bx1, by1 - 35), (bx1 + text_size[0], by1 - 5), (0, 0, 0), -1)
                    cv2.putText(frame, text_overlay, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    ocr_results_list.append({
                        "code": text,
                        "confidence": round(ocr_conf * 100, 2),
                        "timestamp": ocr_time,
                        "detection_type": detection_type
                    })
        # Save detection image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_name}_capture_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        # Trả về tất cả kết quả OCR
        return {
            "ocr_results": ocr_results_list,
            "image_path": filepath,
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error capturing image from {camera_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Add router
app.include_router(router)