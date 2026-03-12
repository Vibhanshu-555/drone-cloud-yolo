from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

model = YOLO("best.pt")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img_bytes = await file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img, conf=0.25)

    output = []

    for r in results:
        for b in r.boxes:
            output.append({
                "cls": int(b.cls),
                "conf": float(b.conf)
            })

    return {"detections": output}