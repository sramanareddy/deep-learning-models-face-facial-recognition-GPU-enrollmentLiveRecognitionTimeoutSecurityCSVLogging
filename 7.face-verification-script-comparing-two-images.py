import torch
import torch.nn.functional as F

def verify_faces(embedding1, embedding2, threshold=0.6):
    """
    Compares two face embeddings using Euclidean Distance.
    """
    # Calculate the distance between the two vectors
    distance = torch.dist(embedding1, embedding2)
    
    is_match = distance.item() < threshold
    
    print(f"📏 Distance: {distance.item():.4f}")
    if is_match:
        print("✅ Identity Verified: It's the same person!")
    else:
        print("❌ Identity Mismatch: These are different people.")
    
    return is_match

# --- SIMULATION ---
# 1. Generate a 'Base' fingerprint for User A
user_a_ref = torch.randn(1, 128)
user_a_ref = F.normalize(user_a_ref, p=2, dim=1)

# 2. Simulate a new login attempt by User A (slightly different vector due to lighting/angle)
user_a_login = user_a_ref + torch.randn(1, 128) * 0.05 
user_a_login = F.normalize(user_a_login, p=2, dim=1)

# 3. Simulate an intruder (totally different vector)
intruder_login = torch.randn(1, 128)
intruder_login = F.normalize(intruder_login, p=2, dim=1)

print("--- Attempt 1: Real User ---")
verify_faces(user_a_ref, user_a_login)

print("\n--- Attempt 2: Intruder ---")
verify_faces(user_a_ref, intruder_login)

# 4. Simulate an intruder (totally same vector)
intruder_login = user_a_ref + torch.randn(1, 128) * 0.05
intruder_login = F.normalize(intruder_login, p=2, dim=1)

print("\n--- Attempt 3: Intruder ---")
verify_faces(user_a_ref, intruder_login)