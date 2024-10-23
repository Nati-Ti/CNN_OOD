from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # MODEL = tf.keras.models.load_model("../model/model_100epo_10val_4convLay_92acc10Epo_88scoreAcc", compile=False)
# MODEL = tf.keras.models.load_model("../model/model\plant_disease_prediction_model_38_Class.h5", compile=False)

# CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy','Unknown']

# Load model
MODEL = tf.keras.models.load_model("../model/model_with_ood.h5", compile=False)

DISEASE_CLASSES = ['healthy', 'diseased1', 'diseased2'] 
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Unknown']


@app.get("/")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    img_array = np.expand_dims(image, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  

    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())
    
    predictions = MODEL.predict(image)

    disease_pred = predictions[0]
    ood_pred = predictions[1]
    
    predicted_class_index = np.argmax(disease_pred)
    predicted_class = DISEASE_CLASSES[predicted_class_index]
    disease_confidence = float(disease_pred[0][predicted_class_index])

    ood_confidence = float(ood_pred[0][0])
    is_potato = ood_confidence > 0.5  
    
    ood_label = "Potato" if is_potato else "Not Potato"

    return {
        'disease_classification': {
            'class': predicted_class,
            'confidence': disease_confidence
        },
        'ood_detection': {
            'is_potato': ood_label,
            'confidence': ood_confidence
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)