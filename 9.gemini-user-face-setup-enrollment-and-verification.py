
"""
To build a professional, fully functional pipeline in a single file, 
    we need to combine the CNN architecture, the Face Detector (Haar Cascades), 
    the Preprocessing logic, and the Persistence logic (Save/Load).

I have structured this script so that it handles the 
    three main stages of an AI system: Setup, Enrollment, and Verification.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np

# ==========================================
# 1. MODEL ARCHITECTURE (The Brain)
# ==========================================
class FaceRecognitionCNN(nn.Module):
    def __init__(self):
        super(FaceRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128) # 128-D Fingerprint

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

# ==========================================
# 2. PIPELINE UTILITIES (The Tools)
# ==========================================
class FacePipeline:
    def __init__(self):
        self.model = FaceRecognitionCNN()
        self.model.eval()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_face_embedding(self, image_source):
        """Detects, crops, and encodes a face."""
        # Convert source to OpenCV format
        if isinstance(image_source, str):
            frame = cv2.imread(image_source)
        else:
            frame = image_source

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None, None

        # Crop first face found
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        
        # CNN Inference
        img_tensor = self.transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding, (x, y, w, h)

    def enroll_user(self, user_name, image_path):
        embedding, _ = self.get_face_embedding(image_path)
        if embedding is not None:
            torch.save(embedding, f"{user_name}.pth")
            print(f"✅ User '{user_name}' enrolled successfully.")
        else:
            print(f"❌ Failed to enroll '{user_name}': No face detected.")

    def run_live(self, target_user_name):
        ref_path = f"{target_user_name}.pth"
        if not os.path.exists(ref_path):
            print(f"User {target_user_name} not found. Please enroll first.")
            return

        ref_embedding = torch.load(ref_path)
        cap = cv2.VideoCapture(0)
        print(f"Verifying for: {target_user_name}. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            curr_embedding, coords = self.get_face_embedding(frame)

            if curr_embedding is not None:
                x, y, w, h = coords
                dist = torch.dist(ref_embedding, curr_embedding).item()
                
                # Biometric logic
                is_match = dist < 0.6
                color = (0, 255, 0) if is_match else (0, 0, 255)
                label = f"Match: {is_match} ({dist:.2f})"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Face Recognition Pipeline', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    app = FacePipeline()
    
    # --- STEP A: ENROLL ---
    # Put a clear photo of yourself in the script folder
    app.enroll_user("GeminiUser", "data/123456.jpg")

    # --- STEP B: VERIFY LIVE ---
    app.run_live("GeminiUser")
    
    print("Script Loaded. Uncomment Enrollment or Live Verification to run.")


"""
How this Pipeline Functions
This single file manages the entire Feature Engineering lifecycle:

1. Normalization & Alignment: The transforms and cv2.CascadeClassifier 
    work together to ensure the AI only sees a clean, centered $64 \times 64$ face.
2. Vectorization: The FaceRecognitionCNN maps high-dimensional image data into a low-dimensional "Face Space.
3. Persistence: Using .pth files allows you to move your "face fingerprints" 
    between different devices without needing the original photos.
4. Real-Time Feedback: The run_live loop calculates the Euclidean Distance 30 times per second, 
    giving you instant verification results.
    
Critical Note on Accuracy
    
    Because the weights of the FaceRecognitionCNN are currently random (untrained), 
        the distance scores will be random. In a production environment, 
        you would load pre-trained weights (like those from a ResNet-50 trained on the VGGFace2 dataset) 
        into this architecture.
    Would you like me to show you how to download and load Pre-trained Weights 
        so this script works accurately with real faces right now?

"""