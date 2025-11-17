import numpy as np

class PersonCropper:
    def __init__(self, pad: float = 0.25, model_name: str = "yolov8n.pt"):
        self.pad = pad
        self.model = None
        self.model_name = model_name

    def _ensure_model(self):
        if self.model is None:
            from ultralytics import YOLO  # import here to avoid hard dependency at import time
            self.model = YOLO(self.model_name)

    def crop(self, frame_bgr):
        """
        Returns a torso+arms crop around the most confident 'person'.
        Falls back to full frame if none found.
        """
        H, W = frame_bgr.shape[:2]
        try:
            self._ensure_model()
            r = self.model(frame_bgr, verbose=False)[0]
            persons = [b for b in r.boxes if int(b.cls) == 0]
            if not persons:
                return frame_bgr
            b = max(persons, key=lambda x: float(x.conf))
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - self.pad * w)); x2 = min(W, int(x2 + self.pad * w))
            y1 = max(0, int(y1 - self.pad * h)); y2 = min(H, int(y2 + self.pad * h))
            return frame_bgr[y1:y2, x1:x2]
        except Exception:
            # if YOLO not installed or any error, just return original
            return frame_bgr
