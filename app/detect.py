import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import easyocr
from collections import deque

model = YOLO("models/yolov8n.pt")
reader = easyocr.Reader(['en'])

# Lưu cache các mã container đã nhận
container_cache = deque(maxlen=100)  # chỉ giữ khoảng 100 mục mới nhất

def is_duplicate(text):
    # Kiểm tra trùng lặp dựa trên text + thời gian
    for item in container_cache:
        if item["text"] == text and time.time() - item["ts"] < 120:  # 2 phút điều chỉnh thời gian nhận diện ở đây
            return True
    return False

def detect_and_display(rtsp_url, upload_callback):
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame1 = cap.read()
    time.sleep(1)
    ret, frame2 = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được từ camera.")
            break

        # ========== PHÁT HIỆN CHUYỂN ĐỘNG ==========
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame1 = frame2
        ret, frame2 = cap.read()

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Vùng chuyển động

        # ========== YOLO + OCR ==========
        results = model(frame)[0]
        output_data = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cropped = frame[y1:y2, x1:x2]
            ocr_result = reader.readtext(cropped)

            if ocr_result:
                text, ocr_conf = ocr_result[0][1], ocr_result[0][2]
                if not is_duplicate(text):
                    # Nếu không trùng lặp, lưu lại cache và gửi lên
                    container_cache.append({"text": text, "ts": time.time()})
                    data = {
                        "container_code": text,
                        "confidence": round(ocr_conf * 100, 2),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    output_data.append(data)

                # Vẽ box xanh với text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

        if output_data:
            upload_callback(output_data)

        cv2.imshow("Container OCR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
