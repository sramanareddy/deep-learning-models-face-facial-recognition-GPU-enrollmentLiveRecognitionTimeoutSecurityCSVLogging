"""
To make this script production-ready, we will add GPU acceleration (CUDA) and a timeout mechanism.

The timeout logic is essential for security and power saving: if the system cannot verify a face within a specific timeframe (e.g., 10 seconds), it will automatically shut down the camera and exit.

High-Accuracy Script with GPU & Timeout

"""
# python -m pip install --upgrade pip setuptools wheel
# pip install pyinstaller
# pip install --upgrade pyinstaller
# pip install --upgrade pyinstaller-hooks-contrib
# pyinstaller --noconfirm --onedir --windowed --collect-all torch --collect-all facenet_pytorch --add-data ".venv/Lib/site-packages/facenet_pytorch;facenet_pytorch" 11.high-accuracy-script-with-GPU-and-timeout.py
# pyinstaller --noconfirm --onedir --windowed --add-data ".venv/Lib/site-packages/facenet_pytorch;facenet_pytorch"  11.high-accuracy-script-with-GPU-and-timeout.py
# pyinstaller --noconfirm --onedir --windowed --collect-all torch --collect-all facenet_pytorch --exclude-module tensorboard 11.high-accuracy-script-with-GPU-and-timeout.py

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import cv2
import time
import sys

class SecureFacePipeline:
    def __init__(self, timeout_seconds=10):
        # 1. GPU ACCELERATION: Automatically detect if a GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Running on device: {self.device}")

        # Initialize MTCNN for face detection
        self.detector = MTCNN(keep_all=False, device=self.device)
        
        # Load Pre-trained model and move it to GPU/CPU
        self.model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        
        self.timeout_limit = timeout_seconds

    def get_embedding(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Detect and return face tensor on the correct device
        face_tensor = self.detector(img_pil)
        
        if face_tensor is not None:
            # Move tensor to device and add batch dimension
            face_tensor = face_tensor.to(self.device).unsqueeze(0)
            with torch.no_grad():
                return self.model(face_tensor)
        return None

    def enroll(self, user_name, image_path):
        img = cv2.imread(image_path)
        emb = self.get_embedding(img)
        if emb is not None:
            torch.save(emb, f"{user_name}.pth")
            print(f"✅ Enrollment successful for {user_name}")

    def verify_live(self, user_name):
        # Load reference embedding to the device
        try:
            ref_emb = torch.load(f"{user_name}.pth").to(self.device)
        except FileNotFoundError:
            print(f"❌ Error: {user_name}.pth not found. Please enroll first.")
            return

        cap = cv2.VideoCapture(0)
        start_time = time.time() # Start the timer
        
        print(f"🔍 Verifying... You have {self.timeout_limit} seconds.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            elapsed_time = time.time() - start_time
            remaining_time = max(0, self.timeout_limit - elapsed_time)

            curr_emb = self.get_embedding(frame)
            
            # --- TIMEOUT LOGIC ---
            if elapsed_time > self.timeout_limit:
                print("\n⏰ TIMEOUT: Face not recognized within the time limit.")
                break

            if curr_emb is not None:
                similarity = torch.cosine_similarity(ref_emb, curr_emb).item()
                is_match = similarity > 0.7
                
                if is_match:
                    print(f"🔓 ACCESS GRANTED: Welcome {user_name}!")
                    # You could trigger a door lock or login here
                    break 
                
                # Visual Feedback
                color = (0, 0, 255) # Red for no match
                label = f"Verifying... {remaining_time:.1f}s left"
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            else:
                cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('Secure Face Verification', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

# --- EXECUTION ---
if __name__ == "__main__":
    pipeline = SecureFacePipeline(timeout_seconds=10) # 10-second timeout
    
    # Example usage:
    #pipeline.enroll("RAMANA", "data/hebba1.jpg")
    pipeline.enroll("Admin", "data/1000028140.jpg")
    pipeline.verify_live("Admin")

"""
Key Technical Improvements
1. Device Agnostic Logic
The script now uses .to(self.device). If you have an NVIDIA GPU with CUDA installed, the model will run significantly faster (higher FPS). If not, it gracefully falls back to the CPU.

2. Timeout & Security
By tracking start_time, the system prevents "unauthorized lingering." In a healthcare or finance setting, you don't want the camera to stay active indefinitely if no authorized person is present.

3. MTCNN vs. Haar Cascades
Notice that we replaced cv2.CascadeClassifier with MTCNN. MTCNN uses a 3-stage neural network process (P-Net, R-Net, and O-Net) to identify faces with far higher precision in varying light conditions.

Final Checklist for Performance
Lighting: Ensure your face is well-lit.

GPU Drivers: If using a GPU, ensure CUDA and cuDNN are installed correctly.

Environment: Run pip install facenet-pytorch opencv-python torch torchvision.

Would you like me to help you package this entire project into a .exe (for Windows) or a standalone application so it can be installed on other computers without needing Python installed?

"""