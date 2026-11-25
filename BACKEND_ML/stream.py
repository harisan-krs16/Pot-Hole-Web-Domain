import requests
import cv2
import numpy as np

url = "http://192.168.132.239/640x480.mjpeg"
response = requests.get(url, stream=True)

if response.status_code == 200:
    bytes_data = bytes()
    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
else:
    print(f"Error: Unable to access stream. HTTP {response.status_code}")

cv2.destroyAllWindows()
