"""
To make this script accurate for real-world faces, we will use 'Transfer Learning'. 
Instead of using our randomly initialized CNN, 
    we will load a model that has already been trained on millions of faces.

The industry standard for this is the InceptionResnetV1 (part of the FaceNet architecture). 
    We will use the facenet-pytorch library, which is the most reliable way 
    to get high-accuracy face embeddings in Python.

1. The Pre-trained Pipeline Architecture
    In this setup, we replace our custom CNN with a "Backbone" that has already learned to recognize eyes, nose shapes, and bone structures.

2. The Updated "High-Accuracy" Script
    First, you'll need to install the library: pip install facenet-pytorch
    # pip install --upgrade pip setuptools wheel
    # pip install -r requirements.txt

"""

# pip install --upgrade pip setuptools wheel
# pip install -r requirements.txt

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import cv2
import os

class AdvancedFacePipeline:
    def __init__(self):
        # MTCNN is a much more accurate face detector than Haar Cascades
        self.detector = MTCNN(keep_all=False, device='cpu')
        
        # Load the Pre-trained 'FaceNet' model (Trained on VGGFace2 dataset)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        
    def get_embedding(self, frame):
        """Detects face and generates a high-accuracy embedding."""
        # Convert OpenCV BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # MTCNN detects and crops the face automatically
        face_tensor = self.detector(img_pil)
        
        if face_tensor is not None:
            # Generate the 512-D high-accuracy embedding
            with torch.no_grad():
                embedding = self.model(face_tensor.unsqueeze(0))
            return embedding
        return None

    def enroll(self, user_name, image_path):
        img = cv2.imread(image_path)
        emb = self.get_embedding(img)
        if emb is not None:
            torch.save(emb, f"{user_name}.pth")
            print(f"✅ Enrollment successful for {user_name}")

    def verify_live(self, user_name):
        ref_emb = torch.load(f"{user_name}.pth")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            curr_emb = self.get_embedding(frame)
            
            if curr_emb is not None:
                # Use Cosine Similarity for FaceNet (Higher is more similar)
                # 1.0 is a perfect match, < 0.7 is usually a mismatch
                similarity = torch.cosine_similarity(ref_emb, curr_emb).item()
                
                is_match = similarity > 0.7
                color = (0, 255, 0) if is_match else (0, 0, 255)
                label = f"User: {user_name} | Confidence: {similarity:.2f}"
                
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Pro Facial Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

# --- RUNNING IT ---
if __name__ == "__main__":
    pipeline = AdvancedFacePipeline()
    # 1. First time: 
    pipeline.enroll("Anushka", "data/1234.jpg")
    # 2. Then: 
    pipeline.verify_live("Anushka")

"""
Why this works so well
The MTCNN detector doesn't just find a "box"—it finds 5 key landmarks: the left eye, right eye, nose, and the corners of the mouth. It then "warps" the face so it's perfectly level before the model looks at it.

Summary of what you've built:
 * Backbone: InceptionResnetV1 (Pre-trained on millions of faces).
 * Detection: MTCNN (Landmark-based alignment).
 * Verification: Cosine Similarity (The industry standard for vector comparison).

Would you like me to explain how to optimize this script to run on a GPU (CUDA) if you have one, which would make the frame rate much smoother?
"""