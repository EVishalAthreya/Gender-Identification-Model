import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Load the trained gender classification model
gender_model = tf.keras.models.load_model('gender_classification_model.h5')

# Load the YOLOv8 model for object detection
yolo_model = YOLO('yolov8n.pt')  # Use YOLOv8n (nano) version, it'll download the weights automatically

# Define a function to preprocess the frame for gender classification (resize and normalize)
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to the gender model's input size
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully!")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break

    # YOLOv8 object detection
    results = yolo_model(frame)

    # Initialize counters for male and female
    male_count = 0
    female_count = 0

    # Loop through detected objects and filter only humans (class 0)
    for result in results[0].boxes:
        class_id = int(result.cls[0])  # Get class ID of the detected object
        if class_id == 0:  # Class 0 is 'person' in YOLO
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0].numpy())  # Extract bounding box coordinates

            # Crop the detected human region
            face_crop = frame[y1:y2, x1:x2]

            # Preprocess the cropped human region for gender prediction
            processed_face = preprocess_frame(face_crop)

            # Gender classification
            gender_prediction = gender_model.predict(processed_face)
            gender = 'Male' if gender_prediction >= 0.5 else 'Female'

            # Count males and females
            if gender == 'Male':
                male_count += 1
            else:
                female_count += 1

            # Draw bounding box and label the human with gender
            label = f"{gender}"
            color = (255, 0, 0) if gender == 'Male' else (0, 0, 255)  # Blue for Male, Red for Female
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display counts of male and female on the screen
    cv2.putText(frame, f'Males: {male_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Females: {female_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow('Gender Detection and Counting', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
