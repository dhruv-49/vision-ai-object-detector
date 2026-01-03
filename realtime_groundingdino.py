import cv2
import torch
from torchvision import transforms
import numpy as np

from groundingdino.util.inference import load_model, predict, annotate

# ---------------- CONFIG ----------------
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

TEXT_PROMPT = "person,dog,mobile phone"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cpu"  # CPU only

FRAME_WIDTH = 320   # smaller frame for faster CPU
FRAME_HEIGHT = 240
FRAME_SKIP = 3      # run detection every 3 frames
# ----------------------------------------

# Transform: numpy HWC [0,255] -> tensor CHW [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
])

def main():
    print("Loading GroundingDINO model...")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    print("Press 'q' to quit.")
    frame_count = 0
    annotated_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize for CPU speed
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = transform(frame_rgb).to(DEVICE)

        # Run detection every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            with torch.no_grad():
                boxes, scores, labels = predict(
                    model=model,
                    image=image_tensor,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=DEVICE
                )
            annotated_frame = annotate(
                image_source=frame_rgb,
                boxes=boxes,
                logits=scores,
                phrases=labels
            )
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Show last annotated frame if skipping frames
        display_frame = annotated_frame if annotated_frame is not None else frame
        cv2.imshow("GroundingDINO Realtime (CPU Optimized)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
