from facenet_pytorch import MTCNN
from PIL import Image

# Initialize face detector
mtcnn = MTCNN(
    image_size=64,     # Match CNN input
    margin=10,
    select_largest=True,
    post_process=True
)

def detect_and_crop_face(image_path):
    """
    Detects face and returns a cropped PIL image.
    Raises error if no face is found.
    """
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        raise ValueError("❌ No face detected in image")

    return face
