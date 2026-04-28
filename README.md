🚀 Real-Time Object Detection with YOLOv8 (FastAPI)
📌 Project Title & Description

This project implements a real-time object detection system using a pretrained YOLOv8 model served through a FastAPI backend.

The system allows users to upload images and videos via API endpoints, perform object detection, and receive annotated outputs. A separate webcam script enables live detection by sending frames to the API.

The YOLOv8 model is pretrained on the COCO dataset (80 object classes) and works out of the box.

🌐 API Endpoints
🔹 1. GET /

Description: Health check endpoint

Input: None

Output Example:

{
  "status": "ok",
  "model": "yolov8n"
}
🔹 2. GET /classes

Description: Returns all COCO class names

Input: None

Output Example:

{
  "classes": ["person", "bicycle", "car", "dog", "..."]
}
🔹 3. POST /detect/image

Description: Detect objects in an uploaded image

Input:

file (required): Image file (JPEG/PNG)
confidence (optional): float (default = 0.4)
classes (optional): comma-separated class names

Output:

Annotated image (JPEG)
Response header containing detection counts:
X-Detections: {'person': 2, 'car': 1}
🔹 4. POST /detect/video

Description: Detect objects in an uploaded video

Input:

file (required): Video file (MP4)
confidence (optional)
classes (optional)

Output:

Annotated video (MP4 stream)
⚙️ Installation & Running the Server
1. Clone the Repository
git clone <your-github-repo-link>
cd yolo-detection
2. Install Dependencies
pip install -r requirements.txt
3. Start the FastAPI Server
fastapi dev main.py
4. Access Swagger UI

Open your browser and go to:

http://localhost:8000/docs
🧪 Testing the API

Run the test script:

python test_api.py

Expected terminal output:

Detections: {'person': 1}
📸 Screenshots (Required)
🔹 1. Swagger UI (/docs)

👉 Insert screenshot showing all API endpoints in Swagger UI

🔹 2. test_api.py Output (Terminal)

👉 Insert screenshot showing successful API test results

🔹 3. Webcam Detection Running

👉 Insert screenshot of live detection using webcam (if tested locally)

🎥 Webcam Detection

Run the webcam script:

python webcam.py
Captures live video from webcam
Sends frames to API
Displays annotated output

Press 'q' to exit.

📂 Project Structure
yolo-detection/
│── main.py          # FastAPI backend
│── webcam.py        # Webcam detection script
│── test_api.py      # API testing script
│── requirements.txt # Dependencies
│── README.md        # Documentation