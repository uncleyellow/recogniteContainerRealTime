import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Đặt tên file Google Sheet và tên sheet
spreadsheet = client.open("ContainerTracking")
sheet = spreadsheet.sheet1

def upload_to_sheet(data):
    for item in data:
        sheet.append_row([
            item["container_code"],
            item["container_code"],  # hoặc item["full_text"] nếu có nhiều dòng OCR
            item["timestamp"],
            item["confidence"]
        ])
