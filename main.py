from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2  # Assuming you are using OpenCV for image processing
from keras.models import load_model  # or appropriate import for your model

app = FastAPI()

# Load your gesture recognition model
gesture_model = load_model('gesture_model.h5')

# Define your object detection function
def detect_objects(image):
    # Your object detection logic here
    # Return detected objects in a suitable format
    return detected_objects

# Define your gesture recognition function
def recognize_gesture(image):
    # Preprocess the image as required for your model
    img = cv2.resize(image, (your_input_size))  # Adjust as necessary
    img = img / 255.0  # Normalization if required
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = gesture_model.predict(img)
    gesture = np.argmax(prediction)
    confidence = np.max(prediction)

    return gesture, confidence

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    detected_objects = detect_objects(image)

    # Perform gesture recognition
    gesture, confidence = recognize_gesture(image)

    # Prepare the response
    result = {
        "detected_objects": detected_objects,
        "gesture": gesture,
        "confidence": confidence
    }

    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
