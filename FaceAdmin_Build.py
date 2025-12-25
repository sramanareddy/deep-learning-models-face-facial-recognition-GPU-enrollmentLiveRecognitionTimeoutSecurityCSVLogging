
"""
This final version integrates everything: GPU support, Multi-user Enrollment, Live Recognition, Timeout Security, and CSV Logging into a single "Admin Control" interface.

The Complete Facial Recognition Admin System

"""

# python -m pip install --upgrade pip setuptools wheel
# pip install pyinstaller
# pip install --upgrade pyinstaller
# pip install --upgrade pyinstaller-hooks-contrib
# pyinstaller --noconfirm --onedir --windowed --collect-all torch --collect-all facenet_pytorch --add-data ".venv/Lib/site-packages/facenet_pytorch;facenet_pytorch" FaceAdmin_Build.py
# pyinstaller --noconfirm --onedir --windowed --add-data ".venv/Lib/site-packages/facenet_pytorch;facenet_pytorch" FaceAdmin_Build.py
# pyinstaller --noconfirm --onedir --windowed --collect-all torch --collect-all facenet_pytorch --exclude-module tensorboard FaceAdmin_Build.py


import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import cv2
import os
import time
import csv
from datetime import datetime

class ProfessionalFaceSystem:
    def __init__(self, db_path="face_db", log_file="access_log.csv", timeout=15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_path = db_path
        self.log_file = log_file
        self.timeout_limit = timeout
        
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)
        self.init_log()

        # Load High-Accuracy Models
        self.detector = MTCNN(keep_all=False, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        
        self.user_db = self.load_all_users()
        self.last_logged = {} 

    def init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "User", "Confidence", "Status"])

    def load_all_users(self):
        db = {}
        for file in os.listdir(self.db_path):
            if file.endswith(".pth"):
                name = file.replace(".pth", "")
                db[name] = torch.load(os.path.join(self.db_path, file), map_location=self.device)
        print(f"--- Database Loaded: {len(db)} users found ---")
        return db

    def get_embedding(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_tensor = self.detector(Image.fromarray(img_rgb))
        if face_tensor is not None:
            face_tensor = face_tensor.to(self.device).unsqueeze(0)
            with torch.no_grad():
                return self.model(face_tensor)
        return None

    def enroll_via_camera(self):
        name = input("Enter the name of the person to enroll: ").strip()
        if not name: return
        
        cap = cv2.VideoCapture(0)
        print("Look at the camera. Capturing in 3 seconds...")
        time.sleep(3)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            emb = self.get_embedding(frame)
            if emb is not None:
                torch.save(emb, os.path.join(self.db_path, f"{name}.pth"))
                self.user_db[name] = emb
                print(f"✅ User '{name}' successfully enrolled.")
            else:
                print("❌ No face detected. Try again.")

    def log_entry(self, name, score):
        now = time.time()
        if name not in self.last_logged or (now - self.last_logged[name] > 30):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name, f"{score:.2f}", "Verified"])
            self.last_logged[name] = now
            print(f"📝 Access Logged: {name}")

    def run_live_verification(self):
        if not self.user_db:
            print("❌ Database is empty. Please enroll someone first.")
            return

        cap = cv2.VideoCapture(0)
        start_time = time.time()
        print(f"🔍 System Active. Timeout in {self.timeout_limit}s. Press 'q' to stop.")

        while True:
            ret, frame = cap.read()
            if not ret or (time.time() - start_time > self.timeout_limit):
                print("⏰ Session ended (Timeout or Camera Error).")
                break

            curr_emb = self.get_embedding(frame)
            if curr_emb is not None:
                best_name, best_score = "Unknown", 0.0
                for name, ref_emb in self.user_db.items():
                    sim = torch.cosine_similarity(ref_emb, curr_emb).item()
                    if sim > best_score:
                        best_score, best_name = sim, name

                is_match = best_score > 0.75
                name_to_show = best_name if is_match else "Unknown"
                color = (0, 255, 0) if is_match else (0, 0, 255)
                
                if is_match: self.log_entry(best_name, best_score)
                
                cv2.putText(frame, f"{name_to_show} ({best_score:.2f})", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Security Console', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

# ==========================================
# ADMIN CONTROL MENU
# ==========================================
if __name__ == "__main__":
    sys_admin = ProfessionalFaceSystem(timeout=30)
    
    while True:
        print("\n--- AI FACIAL RECOGNITION ADMIN ---")
        print("1. Enroll New User (Webcam)")
        print("2. Start Live Verification & Logging")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ")
        
        if choice == '1':
            sys_admin.enroll_via_camera()
        elif choice == '2':
            sys_admin.run_live_verification()
        elif choice == '3':
            print("Shutting down...")
            break
        else:
            print("Invalid selection.")

"""
Key Features of this Integrated Blueprint
Memory Management: The user_db dictionary ensures that even with hundreds of users, the "matching" step happens in milliseconds using vector math.

Thresholding: I have set the threshold to 0.75. This is a "High Security" setting common in Finance—it's better to occasionally fail to recognize a valid user (False Reject) than to accidentally let an intruder in (False Accept).

Dynamic UI: The "Admin Menu" allows you to manage the lifecycle of your system without restarting the script.

Important Distribution Note
When you use PyInstaller to turn this into an .exe, ensure you include the face_db folder and the access_log.csv in your distribution folder. Without the face_db folder, the script won't have a place to save or load the users.

Would you like me to show you how to add Encrypted Data Storage, so that even if someone finds the .pth files in your face_db folder, they cannot read the biometric data?

"""