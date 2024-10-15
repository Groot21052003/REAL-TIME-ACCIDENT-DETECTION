import os
import warnings
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show error messages
tf.get_logger().setLevel('ERROR')  # Set the logger to error level

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the YOLOv5 model (pre-trained)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the CNN model
cnn_model = load_model('final_saved_model/my_model.keras', compile=False)

# Function to enhance and resize images
def enhance_and_resize(image, target_size=(256, 256)):
    # Enhance sharpness and contrast
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Sharpen the image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast
    # Resize the image to the target size
    image = image.resize(target_size)
    return image

# Function to classify images using the CNN model
def classify_car_image(image):
    image = np.array(image) / 255.0
    image = tf.image.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)

    # Get predictions from the CNN model
    prediction = cnn_model.predict(image, verbose=0)  # Set verbose=0 to suppress output

    return prediction[0][0]  # Return the probability score

# Function to extract, enhance, classify, and output car images from a video
def extract_and_classify_cars(video_path):
    cap = cv2.VideoCapture(video_path)
    accident_detected = False
    frame_count = 0

    accident_probabilities = []  # Store probabilities for consecutive frames
    accident_frame_count = 0  # Counter for frames classified as accidents
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = yolo_model(pil_frame)
        detections = results.pandas().xyxy[0]
        cars = detections[detections['name'] == 'car']

        for i, car in cars.iterrows():
            x1, y1, x2, y2 = int(car['xmin']), int(car['ymin']), int(car['xmax']), int(car['ymax'])
            car_image = pil_frame.crop((x1, y1, x2, y2))  # Crop the car image
            enhanced_car_image = enhance_and_resize(car_image)  # Enhance and resize the car image

            accident_probability = classify_car_image(enhanced_car_image)  # Classify the car image
            accident_probabilities.append(accident_probability)  # Collect the probabilities
            
            # If a frame's accident probability is greater than 0.5, consider it an "accident" frame
            if accident_probability > 0.5:
                accident_frame_count += 1

        total_frames += 1

    cap.release()

    # Calculate the percentage of accident frames
    if total_frames > 0:
        accident_frame_percentage = accident_frame_count / total_frames

        # Adjust threshold based on the proportion of accident frames
        if accident_frame_percentage > 0.5:  # Adjust this percentage threshold based on performance
            accident_detected = True

    if accident_detected:
        print("Accident detected in the video.")
    else:
        print("No accidents detected in the video.")

# Example usage
video_path = 'accident_2.mp4'  # Path to the input video file
extract_and_classify_cars(video_path)
