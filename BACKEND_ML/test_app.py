from ultralytics import YOLO
import cv2
import numpy as np
import requests
import serial
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Initialize Firebase
cred = credentials.Certificate('services.json')  # Replace with your Firebase service account key
firebase_admin.initialize_app(cred)
db = firestore.client()
coordinates_collection = db.collection('Coordinates')

# MJPEG stream URL
url = "http://192.168.1.54/800x600.mjpeg"

# Serial port configuration for GPS data
serial_port = serial.Serial('/dev/cu.usbserial-0001', baudrate=115200, timeout=1)  # Replace with your serial port
gps_data = ""

# Open the stream using requests
response = requests.get(url, stream=True)

if response.status_code != 200:
    print(f"Error: Unable to access stream. HTTP {response.status_code}")
    exit()

bytes_data = bytes()
count = 0

def send_to_firebase(latitude, longitude):
    """Send latitude and longitude data to Firebase."""
    unique_id = str(uuid.uuid4())
    try:
        coordinates_collection.document(unique_id).set({
            'location': f"{latitude},{longitude}"
        })
        print(f"Location sent to Firebase: {latitude}, {longitude}")
    except Exception as e:
        print(f"Error sending to Firebase: {e}")

while True:
    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            count += 1
            if count % 3 != 0:  # Skip frames to reduce processing
                continue

            # Resize the frame
            frame = cv2.resize(frame, (1020, 500))
            h, w, _ = frame.shape
            results = model.predict(frame)

            for r in results:
                boxes = r.boxes  # Boxes object for bbox outputs
                masks = r.masks  # Masks object for segment masks outputs

            pothole_detected = False

            if masks is not None:
                masks = masks.data.cpu()
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        d = int(box.cls)
                        c = class_names[d]
                        if c == "road-pothole":  # Replace with your pothole class name
                            pothole_detected = True
                            x, y, x1, y1 = cv2.boundingRect(contour)
                            cv2.polylines(frame, [contour], True, color=(0, 0, 255), thickness=2)
                            cv2.putText(frame, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Read GPS data from serial
            if pothole_detected:
                gps_line = serial_port.readline().decode('utf-8', errors='ignore').strip()
                if "$GPGGA" in gps_line:
                    parts = gps_line.split(",")
                    if len(parts) > 5 and parts[2] and parts[4]:
                        lat = float(parts[2]) / 100.0
                        lon = float(parts[4]) / 100.0
                        send_to_firebase(lat, lon)

            # Display the frame
            cv2.imshow('Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
