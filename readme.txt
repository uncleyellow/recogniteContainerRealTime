📦 Container OCR Streaming – Realtime Detection App
Ứng dụng Python sử dụng YOLOv8, EasyOCR, OpenCV, và FastAPI để:

Nhận diện chuyển động từ camera RTSP Hikvision

Khoanh hình đỏ cho vùng có chuyển động

Dùng YOLOv8 phát hiện vùng chứa text → dùng EasyOCR nhận diện text

Khoanh hình xanh vùng có mã container và overlay text

Gửi dữ liệu nhận diện lên Google Sheets

🧩 Yêu cầu hệ thống
Python 3.9+

Máy tính có GPU (tuỳ chọn) để tăng tốc YOLOv8 (khuyến khích)

Kết nối mạng và có quyền truy cập camera RTSP Hikvision

🛠️ Cài đặt thư viện
bash
pip install ultralytics easyocr opencv-python-headless fastapi uvicorn gspread oauth2client

🔐 Google Sheets API
Truy cập: https://console.cloud.google.com/

Tạo dự án mới → Kích hoạt Google Sheets API và Google Drive API

Tạo Service Account, tạo key JSON và tải về → lưu thành credentials.json

Chia sẻ Google Sheet với email trong file credentials.json

📁 Cấu trúc thư mục
app/
├── main.py              # FastAPI app khởi động camera + nhận diện
├── detect.py            # Nhận diện chuyển động + OCR + bounding box
├── sheets.py            # Kết nối và gửi dữ liệu lên Google Sheets
├── models/
│   └── yolov8_container.pt  # Model YOLOv8 tùy chỉnh
credentials.json         # File xác thực Google API
README.md
✏️ Cấu hình RTSP và Google Sheet
📸 main.py
RTSP_URL = "rtsp:"
📄 sheets.py
spreadsheet = client.open("ContainerTracking")  # Tên Google Sheet
sheet = spreadsheet.sheet1                      # Sheet đầu tiên
🚀 Chạy ứng dụng
bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
Sau đó mở trình duyệt:

bash
http://localhost:8000/start/CONG-VAO-DA-1
Ứng dụng sẽ:

Mở camera trực tiếp từ Hikvision

Vẽ vùng chuyển động → khung đỏ

Vùng có mã container → khung xanh và overlay mã

Gửi mã OCR lên Google Sheet nếu không trùng lặp trong 2 phút gần nhất

Để thoát ứng dụng nhấn q

📊 Cột dữ liệu trong Google Sheet

Mã Container	Text nhận diện được	Thời gian gửi lên	Độ tin cậy (%)
TCNU1234567	        TCNU1234567	     2025-04-23 10:55	    93.75


rtsp://MrKhanhKHDT:ratraco@118@14.232.166.207:1554/Streaming/Channels/202/