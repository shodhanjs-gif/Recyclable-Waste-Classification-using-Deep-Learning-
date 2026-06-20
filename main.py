from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from train import train_model

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = 'waste_model.keras'
CUSTOM_CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# Load Model
model = None
is_custom_model = False

def load_prediction_model():
    global model, is_custom_model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            is_custom_model = True
            print(f"Loaded custom model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            model = None
    
    if model is None:
        try:
            model = MobileNetV2(weights='imagenet')
            is_custom_model = False
            print("Loaded base MobileNetV2 model")
        except Exception as e:
            print(f"Error loading base model: {e}")
            model = None

# Initial load
load_prediction_model()

def prepare_image(image_data):
    try:
        img = Image.open(io.BytesIO(image_data)).resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error preparing image: {e}")
        return None

@app.get("/")
def read_root():
    return {"message": "Waste Detection API is running", "custom_model": is_custom_model}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        contents = await file.read()
        processed_image = prepare_image(contents)
        
        if processed_image is None:
             return {"error": "Failed to process image"}

        predictions = model.predict(processed_image)
        
        results = []
        if is_custom_model:
            # Custom model predictions
            for i, score in enumerate(predictions[0]):
                results.append({"label": CUSTOM_CLASSES[i], "confidence": float(score)})
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            results = results[:3] # Top 3
        else:
            # ImageNet predictions
            decoded_predictions = decode_predictions(predictions, top=3)[0]
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                results.append({"label": label, "confidence": float(score)})
            
        return {"predictions": results, "is_custom_model": is_custom_model}

    except Exception as e:
        return {"error": str(e)}

@app.post("/reload-model")
def reload_model_endpoint():
    load_prediction_model()
    return {"message": "Model reload attempted", "custom_model": is_custom_model}

@app.post("/collect")
async def collect_image(label: str, file: UploadFile = File(...)):
    if label not in CUSTOM_CLASSES:
        return {"error": "Invalid label"}
    
    try:
        # Create directory if it doesn't exist (safety check)
        label_dir = os.path.join('dataset', label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Generate filename
        import time
        filename = f"{int(time.time() * 1000)}.jpg"
        filepath = os.path.join(label_dir, filename)
        
        # Save file
        contents = await file.read()
        image_data = Image.open(io.BytesIO(contents)).convert('RGB')
        image_data.save(filepath, "JPEG")
        
        return {"message": f"Image saved to {label} category", "filename": filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/upload")
async def upload_images(label: str, files: list[UploadFile] = File(...)):
    if label not in CUSTOM_CLASSES:
        return {"error": "Invalid label"}
    
    try:
        label_dir = os.path.join('dataset', label)
        os.makedirs(label_dir, exist_ok=True)
        
        count = 0
        import time
        for file in files:
            # Generate unique filename
            filename = f"{int(time.time() * 1000)}_{count}.jpg"
            filepath = os.path.join(label_dir, filename)
            
            # Save file
            contents = await file.read()
            image_data = Image.open(io.BytesIO(contents)).convert('RGB')
            image_data.save(filepath, "JPEG")
            count += 1
            
        print(f"Uploaded {count} images to {label}")
        return {"message": f"Successfully uploaded {count} images to {label} category"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_training():
    try:
        print("Starting training in background...")
        # Reload module to ensure latest code if changed, though not strictly necessary for simple cases
        import train
        import importlib
        importlib.reload(train)
        from train import train_model
        
        train_model()
        print("Training complete. Reloading model...")
        load_prediction_model()
    except Exception as e:
        print(f"Training failed: {e}")

@app.post("/train")
async def train_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training)
    return {"message": "Training started in background. Check console for progress."}
