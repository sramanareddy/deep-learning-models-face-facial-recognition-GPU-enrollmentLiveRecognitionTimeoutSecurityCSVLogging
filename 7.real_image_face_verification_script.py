from faceverificationmodel import FaceRecognitionCNN
#from face_detector import detect_and_crop_face
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# 1. THE PREPROCESSING PIPELINE
# This ensures any image from any source is converted to the format the CNN expects
transform = transforms.Compose([
    transforms.Resize((64, 64)),      # Match the input size of our CNN
    transforms.ToTensor(),            # Convert pixels (0-255) to Tensors (0-1)
    transforms.Normalize(             # Standardize the data
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def load_and_embed(image_path, model):
    """Loads an image from a path and returns its 128-D embedding."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image at {image_path}")
        
    img = Image.open(image_path).convert('RGB') # Ensure it's 3-channel RGB
    img_tensor = transform(img).unsqueeze(0)    # Add 'batch' dimension: [1, 3, 64, 64]
    
    model.eval()
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding

def verify_files(path1, path2, model, threshold=0.6):
    """Takes two file paths and determines if they are the same person."""
    print(f"Comparing: {os.path.basename(path1)} vs {os.path.basename(path2)}")
    
    emb1 = load_and_embed(path1, model)
    emb2 = load_and_embed(path2, model)
    
    distance = torch.dist(emb1, emb2).item()
    
    print(f"📏 Euclidean Distance: {distance:.4f}")
    if distance < threshold:
        print("✅ RESULT: MATCH (Access Granted)")
    else:
        print("❌ RESULT: NO MATCH (Access Denied)")
    return distance

# --- EXECUTION ---
# Assume you have a model instance from our previous CNN class
# model = FaceRecognitionCNN() 

# Example Paths (Replace these with your actual local file paths)
image_a = "data/123456.jpg"
image_b = "data/hebba1.jpg"

model = FaceRecognitionCNN()
model.eval() # Set to evaluation mode for inference
try:
    verify_files(image_a, image_b, model)
except Exception as e:
    print(f"Error: {e}")
    print("Please ensure the image paths are correct and you have an initialized 'model'.")    