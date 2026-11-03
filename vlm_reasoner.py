from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2

print("Loading VLM...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("VLM loaded successfully.")


def analyze_scene(frame):

    # OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    image = Image.fromarray(frame_rgb)

    # Prepare image for BLIP
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    # Generate description
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

    # Convert output to text
    description = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return description