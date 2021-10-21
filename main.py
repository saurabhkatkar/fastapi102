from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

#domain where this   api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.

MODEL = tf.keras.models.load_model('tomato')


@app.get("/")
def home():
    return {"message":"Hello TutLinks.com"}


@app.get("/ping")
async def ping():
    return "Project Group 08 ,we are alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    

    CLASS_NAMES = ["Early Blight", "Late Blight","Septoria leaf Spot","Healthy"]

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
