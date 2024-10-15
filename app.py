from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
import torch
from twilio.rest import Client  # Import Twilio Client

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load CNN model
cnn_model = tf.keras.models.load_model('final_saved_model/my_model.keras', compile=False)

# Function to enhance and resize images
def enhance_and_resize(image, target_size=(256, 256)):
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Sharpen the image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast
    image = image.resize(target_size)
    return image

# Function to classify images using the CNN model
def classify_car_image(image):
    image = np.array(image) / 255.0
    image = tf.image.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)
    prediction = cnn_model.predict(image, verbose=0)
    return prediction[0][0]

# Function to extract and classify cars from video
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

    # Return result as a string
    if accident_detected:
        return "Accident detected in the video."
    else:
        return "No accidents detected in the video."

# Function to send an SMS using Twilio
def send_sms(result):
    account_sid = '' # Add your Twilio Account SID here  
    auth_token = ''  # Add your Twilio Auth Token here
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=f"Accident Detection Result: {result}",  # Message to be sent
        from_='',  # Your Twilio number
        to=''  # Receiver's phone number
    )
    print(f"Message sent with SID: {message.sid}")

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle video upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(url_for('home'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        # Ensure uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)

        # Call your detection function and get the result
        result = extract_and_classify_cars(video_path)

        # Send the result via SMS using Twilio
        send_sms(result)

        # Pass the result to result.html to display
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
