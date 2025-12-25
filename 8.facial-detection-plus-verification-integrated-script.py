import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from faceverificationmodel import FaceRecognitionCNN
from real_image_face_verification_script import transform

# 1. INITIALIZE FACE DETECTOR (Using OpenCV)
# This downloads a pre-trained face-finding model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_path):
    """Finds a face in an image and crops it."""
    # Load image using OpenCV
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print(f"⚠️ No face detected in {image_path}")
        return None
    
    # Take the first face found (x, y, width, height)
    (x, y, w, h) = faces[0]
    
    # Crop the face with a small margin
    face_crop = img[y:y+h, x:x+w]
    
    # Convert back to RGB for PIL/PyTorch
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)

def get_embedding_from_large_photo(image_path, model):
    """Finds face, crops it, and runs it through our CNN."""
    cropped_face = detect_and_crop_face(image_path)
    
    if cropped_face is None:
        return None
    
    # Use the transformation pipeline from our previous step
    img_tensor = transform(cropped_face).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        return model(img_tensor)

# --- INTEGRATED VERIFICATION ---
def verify_complex_images(path1, path2, model):
    emb1 = get_embedding_from_large_photo(path1, model)
    emb2 = get_embedding_from_large_photo(path2, model)
    
    if emb1 is not None and emb2 is not None:
        distance = torch.dist(emb1, emb2).item()
        print(f"Distance between detected faces: {distance:.4f}")
        return distance < 0.6
    return False

# Example Usage:
image_a = "data/123456.jpg"
image_b = "data/hebba1.jpg"

model = FaceRecognitionCNN()
model.eval() # Set to evaluation mode for inference

verify_complex_images(image_a, image_b, model)