"""
To handle multiple users, we need to transition from a "single-file" logic to a Database-style logic. Instead of hardcoding one user, the system will scan a folder of .pth files (the fingerprints) and identify whoever is in front of the camera.

1. The Multi-User Recognition Architecture
In this setup, we load all enrolled user embeddings into a Dictionary {Name: Embedding}. During live verification, the AI compares the current face against every person in the dictionary and picks the best match.

2. The Multi-User "High-Accuracy" Script

"""

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import cv2
import os
import time

class MultiUserFaceSystem:
    def __init__(self, db_path="face_db"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_path = db_path
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        # High-accuracy components
        self.detector = MTCNN(keep_all=False, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        
        # Load all users into memory for fast matching
        self.user_db = self.load_all_users()

    def load_all_users(self):
        """Loads all .pth files into a dictionary."""
        db = {}
        for file in os.listdir(self.db_path):
            if file.endswith(".pth"):
                name = file.replace(".pth", "")
                db[name] = torch.load(os.path.join(self.db_path, file)).to(self.device)
        print(f"Loaded {len(db)} users from database.")
        return db

    def get_embedding(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_tensor = self.detector(Image.fromarray(img_rgb))
        if face_tensor is not None:
            face_tensor = face_tensor.to(self.device).unsqueeze(0)
            with torch.no_grad():
                return self.model(face_tensor)
        return None

    def enroll_new_user(self, user_name, image_path):
        """Add a new person to the system."""
        img = cv2.imread(image_path)
        emb = self.get_embedding(img)
        if emb is not None:
            torch.save(emb, os.path.join(self.db_path, f"{user_name}.pth"))
            self.user_db[user_name] = emb # Update memory
            print(f"✅ User '{user_name}' enrolled.")
        else:
            print(f"❌ Could not find face for {user_name}.")

    def identify_face(self, current_emb, threshold=0.7):
        """Compares current face against all enrolled users."""
        best_match = "Unknown"
        max_sim = -1.0

        for name, ref_emb in self.user_db.items():
            similarity = torch.cosine_similarity(ref_emb, current_emb).item()
            if similarity > max_sim:
                max_sim = similarity
                best_match = name

        if max_sim > threshold:
            return best_match, max_sim
        return "Unknown", max_sim

    def run_live(self):
        cap = cv2.VideoCapture(0)
        print("Starting Multi-User Recognition... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            curr_emb = self.get_embedding(frame)
            
            if curr_emb is not None:
                name, score = self.identify_face(curr_emb)
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({score:.2f})"
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Multi-User AI System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

# --- HOW TO USE ---
if __name__ == "__main__":
    system = MultiUserFaceSystem()
    
    # 1. Enroll multiple people
    # system.enroll_new_user("RAMANA", "data/1000028140.jpg")
    # system.enroll_new_user("Bob", "bob_selfie.jpg")
    
    # 2. Start identifying everyone
    system.run_live()

"""
Why this is better for your POC:
Efficiency: It loads the "Database" into RAM (the user_db dictionary) only once. It doesn't read from the disk while the camera is running, ensuring high FPS.

Scalability: You can add 100 users, and the script will simply loop through the 100 vectors to find the best match.

Unknown Detection: If someone not in your database appears, the system correctly labels them as "Unknown" because their similarity score won't hit the 0.7 threshold.

Summary of Improvements
Folder-based DB: All user data is kept in a dedicated face_db folder.

Cosine Similarity Search: The identify_face function acts as a search engine for faces.

Real-Time Labeling: It dynamically draws the name of the recognized person on the video feed.

Would you like me to add a "Logging System" that saves a CSV file every time a user is recognized, including the exact timestamp? (Useful for attendance or security logs).
"""