from ultralytics import YOLO
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import random

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Initialize Firebase
cred = credentials.Certificate('services.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
coordinates_collection = db.collection('Coordinates')

document_name = input("Enter the document name: ")

def get_next_counter(document_name):
    print("\n=== Initializing Counter ===")
    doc_ref = coordinates_collection.document(f"{document_name}_1")
    doc = doc_ref.get()
    
    if doc.exists:
        # If the first document exists, find the last document
        counter = 1
        while True:
            doc_ref = coordinates_collection.document(f"{document_name}_{counter+1}")
            doc = doc_ref.get()
            if not doc.exists:
                break
            counter += 1
        print(f"Existing counter value: {counter}")
        return counter + 1
    else:
        print("No existing counter, starting from 1")
        return 1

def send_to_firebase(document_name, lat, lon, counter):
    doc_name = f"{document_name}_{counter}"
    print(f"\nAttempting Firebase update for {doc_name}")
    
    try:
        doc_ref = coordinates_collection.document(doc_name)
        doc_ref.set({
            'location': f"{lat},{lon}",
            'name': doc_name
        }, merge=True)
        print(f"Successfully wrote {doc_name}")
    except Exception as e:
        print(f"Firebase Error: {str(e)}")

def update_last_suffix(document_name, suffix):
    doc_ref = coordinates_collection.document(document_name)
    try:
        doc_ref.set({'last_suffix': str(suffix)}, merge=True)
        print(f"Updated last_suffix to {suffix} for {document_name}")
    except Exception as e:
        print(f"Error updating last_suffix: {e}")

# Create a document with the entered name and initialize last_suffix
doc_ref = coordinates_collection.document(document_name)
try:
    doc_ref.set({'last_suffix': 0}, merge=True)
    print(f"Document {document_name} created with last_suffix initialized.")
except Exception as e:
    print(f"Error creating document: {e}")

# Webcam setup
cap = cv2.VideoCapture(1)
counter = get_next_counter(document_name)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (1020, 500))
    h, w, _ = frame.shape
    results = model.predict(frame, verbose=False)

    pothole_detected = False
    
    for r in results:
        if r.masks is not None:
            masks = r.masks.data.cpu()
            for seg, box in zip(masks.numpy(), r.boxes):
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]
                conf = box.conf[0].item()
                
                print(f"\nDetection: {class_name} ({conf:.2f})")

                if class_name.lower() == "roadpothole" and conf > 0.5:
                    pothole_detected = True
                    print("!!! VALID POTHOLE DETECTED !!!")
                    
                    # Generate random coordinates
                    lat = round(random.uniform(8.0, 12.0), 6)
                    lon = round(random.uniform(76.0, 78.0), 6)
                    
                    send_to_firebase(document_name, lat, lon, counter)
                    update_last_suffix(document_name, counter)
                    counter += 1

                    # Resize the mask to match the frame size
                    seg = cv2.resize(seg, (w, h))
                    # Find contours of the mask
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Draw contours and label on the frame
                    for contour in contours:
                        x, y, x1, y1 = cv2.boundingRect(contour)
                        cv2.polylines(frame, [contour], True, color=(0, 0, 255), thickness=2)
                        cv2.putText(frame, f"{class_name} {conf:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()