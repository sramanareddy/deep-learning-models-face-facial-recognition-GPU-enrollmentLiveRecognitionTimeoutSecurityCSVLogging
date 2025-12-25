import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceRecognitionCNN(nn.Module):
    def __init__(self):
        super(FaceRecognitionCNN, self).__init__()
        
        # Layer 1: Detects basic edges and textures
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: Detects complex shapes like eyes/nose
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: High-level facial geometry
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # The "Embedding" Layer: Turns the face into a unique vector
        # Assuming input image is resized to 64x64
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128) # Final 128-D Fingerprint

    def forward(self, x):
        # Convolutional Block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for Dense layers
        x = x.view(-1, 128 * 8 * 8)
        
        # Generating the Embedding
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # L2 Normalization (Crucial for comparing two different faces)
        return F.normalize(x, p=2, dim=1)

# --- TESTING THE MODEL ---
model = FaceRecognitionCNN()
model.eval() # Set to evaluation mode for inference

# Simulate a 64x64 RGB face image input
dummy_face = torch.randn(1, 3, 64, 64)

with torch.no_grad():
    face_fingerprint = model(dummy_face)

print(f"Face Embedding generated! Size: {face_fingerprint.shape}")
print("This vector can now be compared against a database of known faces.")