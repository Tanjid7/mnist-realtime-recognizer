import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# ── Model (matches YOUR saved weights EXACTLY) ────────────────────────────────
# conv_layers.0  → Conv2d(1,  32)   index 0
# conv_layers.1  → BN(32)           index 1
# conv_layers.4  → Conv2d(32, 64)   index 4  (ReLU=2, MaxPool=3 skipped saving)
# conv_layers.5  → BN(64)           index 5
# conv_layers.8  → Conv2d(64,128)   index 8
# conv_layers.9  → BN(128)          index 9
# fc_layers.1    → Linear(1152,256)
# fc_layers.4    → Linear(256,10)

class DigitCNN_BN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 0  weight:(32,1,3,3)
            nn.BatchNorm2d(32),                            # 1  weight:(32,)
            nn.ReLU(inplace=True),                         # 2
            nn.MaxPool2d(2, 2),                            # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 4  weight:(64,32,3,3)
            nn.BatchNorm2d(64),                            # 5  weight:(64,)
            nn.ReLU(inplace=True),                         # 6
            nn.MaxPool2d(2, 2),                            # 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8  weight:(128,64,3,3)
            nn.BatchNorm2d(128),                           # 9  weight:(128,)
            nn.ReLU(inplace=True),                         # 10
            nn.MaxPool2d(2, 2),                            # 11
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),                        # 0
            nn.Linear(1152, 256),                # 1  weight:(256,1152)
            nn.ReLU(inplace=True),               # 2
            nn.Dropout(0.5),                     # 3
            nn.Linear(256, num_classes),         # 4  weight:(10,256)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(weights_path="mnist_bn.pth", device="cpu"):
    model = DigitCNN_BN(num_classes=10)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"[OK] Model loaded from '{weights_path}' on {device}")
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

def preprocess_roi(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized_20 = cv2.resize(thresh, (20, 20), interpolation=cv2.INTER_AREA)
    resized = cv2.copyMakeBorder(resized_20, top=4, bottom=4, left=4, right=4,
                                 borderType=cv2.BORDER_CONSTANT, value=0)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0)
    return tensor, resized


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, tensor, device="cpu"):
    tensor = tensor.to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    pred = probs.argmax(dim=1).item()
    conf = probs[0, pred].item() * 100.0
    return pred, conf


# ── Smoother ──────────────────────────────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, window=10):
        self.buffer = deque(maxlen=window)

    def update(self, pred):
        self.buffer.append(pred)
        return max(set(self.buffer), key=self.buffer.count)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(weights_path="mnist_bn.pth"):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = load_model(weights_path, device)
    smoother = PredictionSmoother(window=8)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    roi_size = 250
    last_pred, last_conf = None, 0.0
    show_thumb = None

    print("Running — Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        cx, cy  = w // 2, h // 2
        x1, y1  = cx - roi_size // 2, cy - roi_size // 2
        x2, y2  = cx + roi_size // 2, cy + roi_size // 2

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            tensor, show_thumb = preprocess_roi(roi)
            raw_pred, conf     = predict(model, tensor, device)
            last_pred          = smoother.update(raw_pred)
            last_conf          = conf

        # Green ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, "Place digit here", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

        # Prediction display
        if last_pred is not None:
            color = (0, 220, 0) if last_conf >= 80 else (0, 80, 220)
            cv2.putText(frame, str(last_pred), (x2 + 20, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 6)
            cv2.putText(frame, f"{last_conf:.1f}%", (x2 + 20, cy + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 28x28 thumbnail (top-left)
        if show_thumb is not None:
            thumb_bgr     = cv2.cvtColor(show_thumb, cv2.COLOR_GRAY2BGR)
            thumb_resized = cv2.resize(thumb_bgr, (84, 84), interpolation=cv2.INTER_NEAREST)
            frame[10:94, 10:94] = thumb_resized
            cv2.rectangle(frame, (10, 10), (93, 93), (200, 200, 200), 1)
            cv2.putText(frame, "Model sees", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.putText(frame, "Q: quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Digit Recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run("mnist_bn.pth")