from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, os
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATE_PATH = "template/template.png"

def detect_face(img: Image.Image):
    cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(cv_img, 1.1, 4)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])

def cartoonize(img: Image.Image):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for _ in range(6):
        cv_img = cv2.bilateralFilter(cv_img, 9, 75, 75)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)

    color = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    cartoon = cv2.bitwise_and(color, edges)
    return Image.fromarray(cartoon)

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    original_bytes = await file.read()
    img = Image.open(io.BytesIO(original_bytes))

    face = detect_face(img)
    if face:
        x, y, w, h = face
        crop = img.crop((x, y, x+w, y+h))
    else:
        w, h = img.size
        crop = img.crop((w//4, h//4, 3*w//4, 3*h//4))

    cartoon = cartoonize(crop)

    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    cartoon = cartoon.resize((200, 200)).convert("RGBA")

    final = template.copy()
    final.paste(cartoon, (150, 80), cartoon)

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
