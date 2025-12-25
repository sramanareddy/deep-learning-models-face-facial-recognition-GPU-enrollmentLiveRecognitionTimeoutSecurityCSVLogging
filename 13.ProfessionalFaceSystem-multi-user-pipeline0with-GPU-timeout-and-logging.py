"""
Adding a logging system transforms your script from a simple demo into a Functional Attendance or Security System. We will use Python's built-in csv module and datetime to record entries.

To prevent the log from being spammed with thousands of entries (since the camera runs at ~30 frames per second), we will implement a Cool-down Logic: the system will only log a specific user once every 30 seconds (or any duration you choose).

Multi-User Pipeline with GPU, Timeout, and Logging

"""

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
        
        # Ensure folders and log files exist
        if not os.path.exists(self.db_path): os.makedirs(self.db_path)
        self.init_log()

        # AI Components
        self.detector = MTCNN(keep_all=False, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        
        # Memory
        self.user_db = self.load_all_users()
        self.last_logged = {} # To prevent duplicate logs: {name: last_time}

    def init_log(self):
        """Creates the CSV header if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "User", "Confidence", "Status"])

    def load_all_users(self):
        db = {}
        for file in os.listdir(self.db_path):
            if file.endswith(".pth"):
                name = file.replace(".pth", "")
                db[name] = torch.load(os.path.join(self.db_path, file)).to(self.device)
        return db

    def log_entry(self, name, score):
        """Records a match to the CSV file with cooldown logic."""
        current_time = time.time()
        # Only log the same person once every 30 seconds
        if name not in self.last_logged or (current_time - self.last_logged[name] > 30):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name, f"{score:.2f}", "Verified"])
            self.last_logged[name] = current_time
            print(f"📝 Logged entry for {name}")

    def run_live(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            elapsed = time.time() - start_time
            if elapsed > self.timeout_limit:
                print("⏰ Session Timeout.")
                break

            # Process Frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_tensor = self.detector(Image.fromarray(img_rgb))
            
            if face_tensor is not None:
                face_tensor = face_tensor.to(self.device).unsqueeze(0)
                with torch.no_grad():
                    curr_emb = self.model(face_tensor)
                
                # Match against DB
                best_name, best_score = "Unknown", 0.0
                for name, ref_emb in self.user_db.items():
                    sim = torch.cosine_similarity(ref_emb, curr_emb).item()
                    if sim > best_score:
                        best_score, best_name = sim, name

                if best_score > 0.7:
                    self.log_entry(best_name, best_score)
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.putText(frame, f"{best_name} ({best_score:.2f})", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Secure Logging System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = ProfessionalFaceSystem()
    system.run_live()

"""
Why this Logic is Essential for Finance/Healthcare
In high-security environments, the "Audit Trail" (the CSV log) is just as important as the recognition itself.

Transparency: You can see exactly what time a person accessed the system.

Compliance: Many financial regulations require a record of who accessed sensitive data.

Efficiency: The Cooldown Logic ensures your log file doesn't grow to gigabytes in size by recording the same person 30 times a second.

The Packaging Note
When you use PyInstaller to build this into an .exe, remember that your code now expects a folder called face_db and a file called access_log.csv.

 * The .exe will look for these in the same folder where it is running.

 * If you share the .exe with someone else, you should send them the whole folder so they have the face_db directory ready to store their fingerprints.

Would you like me to add an "Unauthorized Alert" feature? If an unknown person is detected for more than 3 seconds, it could play a warning sound or save a screenshot of the intruder's face to an alerts/ folder.

"""