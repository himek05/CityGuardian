from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import cv2
import time
import os

app = Flask(__name__)
model = YOLO(r"C:\Users\himek\Desktop\proje\runs\detect\train\weights\best.pt")  # Ensure this file exists or download a YOLOv8 model
print(model.names)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    if request.method == 'POST':
        image = request.files['image']
        input_path = 'static/input.jpg'
        image.save(input_path)

        # Run YOLO model
        results = model(input_path)

        # Get detected image
        detected_img = results[0].plot()

        # Save output with timestamp to prevent caching
        timestamp = str(int(time.time()))
        output_filename = f'result_{timestamp}.jpg'
        output_path = os.path.join('static', output_filename)
        cv2.imwrite(output_path, detected_img)

        result_image = output_filename  # just filename

    return render_template('index.html', result_image=result_image)

if __name__ == '__main__':
    app.run(debug=True)
