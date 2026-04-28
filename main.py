import cv2
import numpy as np
import tempfile
import os
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 Object Detection API")

model = YOLO("yolov8n.pt")

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange'
]

def get_class_ids(classes: list[str]) -> Optional[list[int]]:
    if not classes:
        return None
    return [COCO_CLASSES.index(c) for c in classes if c in COCO_CLASSES]

def run_detection(frame_bgr, conf, class_ids):
    results = model(frame_bgr, conf=conf, classes=class_ids, verbose=False)

    annotated = results[0].plot()
    boxes = results[0].boxes

    counts = {}
    if boxes is not None:
        for cls_id in boxes.cls.tolist():
            name = COCO_CLASSES[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1

    return annotated, counts


@app.get("/")
def health():
    return {"status": "ok", "model": "yolov8n"}

@app.get("/classes")
def list_classes():
    return {"classes": COCO_CLASSES}


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.4),
    classes: str = Query("")
):
    selected = [c.strip() for c in classes.split(",") if c.strip()]
    class_ids = get_class_ids(selected)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    annotated, counts = run_detection(frame, confidence, class_ids)

    _, buffer = cv2.imencode(".jpg", annotated)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"X-Detections": str(counts)}
    )


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Query(0.4),
    classes: str = Query("")
):
    selected = [c.strip() for c in classes.split(",") if c.strip()]
    class_ids = get_class_ids(selected)

    # Save file safely (important fix)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(await file.read())
    tfile.close()  # FIX (was flush before)

    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 25  # FIX fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        os.unlink(tfile.name)
        return JSONResponse(status_code=400, content={"error": "Invalid video file"})

    out_path = tfile.name.replace(".mp4", "_out.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        os.unlink(tfile.name)
        return JSONResponse(status_code=500, content={"error": "VideoWriter failed"})

    all_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, counts = run_detection(frame, confidence, class_ids)
        out.write(annotated)

        for name, count in counts.items():
            all_counts[name] = all_counts.get(name, 0) + count

    cap.release()
    out.release()
    os.unlink(tfile.name)

    def video_stream():
        with open(out_path, "rb") as f:
            yield from f
        os.unlink(out_path)

    return StreamingResponse(
        video_stream(),
        media_type="video/mp4",
        headers={"X-Detections": str(all_counts)}
    )