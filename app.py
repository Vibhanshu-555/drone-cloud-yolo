from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

model = YOLO("best.pt")


# ----------------------
# Upload page
# ----------------------

@app.get("/", response_class=HTMLResponse)
def main():

    return """
    <html>
    <body>
    <h2>Drone YOLO Detection</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Detect">
    </form>

    </body>
    </html>
    """


# ----------------------
# Prediction API
# ----------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img_bytes = await file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img, conf=0.25)

    detections = []

    for r in results:
        for b in r.boxes:
            detections.append({
                "class": int(b.cls),
                "conf": float(b.conf)
            })

    return {"detections": detections}
