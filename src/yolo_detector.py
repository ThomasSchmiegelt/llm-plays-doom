"""YOLO als Detektor - dieselbe Schnittstelle wie LocateAnythingDetector.

Zwei Betriebsarten, beide lokal:

  open vocabulary   YOLO-World. Nimmt dieselben Textklassen wie
                    LocateAnything ("brown insect monster", ...) und braucht
                    kein Training.
  destilliert       ein normales YOLO, trainiert auf Bildern, die
                    LocateAnything beschriftet hat (src/distill.py). Die
                    Klassen stehen dann fest im Modell.

Die Nachbearbeitung - Plausibilitaetsfilter, Unbekanntes, Dedup - ist
dieselbe wie bei LocateAnything, damit die Messungen vergleichbar bleiben.
"""
import time

import numpy as np

from detector import (BROAD_CLASSES, DEFAULT_CLASSES, PROTECTED_CLASSES,
                      Detection, _dedup, _to_pixels, mark_unknown)

WORLD_WEIGHTS = "yolov8x-worldv2.pt"


class YoloDetector:
    def __init__(self, weights: str = WORLD_WEIGHTS, conf: float = 0.2,
                 imgsz: int = 640, device: str = "cuda"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.open_vocab = "world" in weights.lower()
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._classes: list[str] | None = None
        self.last_latency = 0.0
        self.last_raw = ""

    @property
    def known_classes(self) -> list[str]:
        """Destilliert: was das Modell gelernt hat. Open vocabulary: alles."""
        return list(self.model.names.values())

    def detect(self, frame_rgb: np.ndarray, classes: list[str] | None = None,
               protected: list[str] | None = None,
               broad: list[str] | None = None) -> list[Detection]:
        classes = classes or DEFAULT_CLASSES
        protected = PROTECTED_CLASSES if protected is None else protected
        broad = BROAD_CLASSES if broad is None else broad
        if self.open_vocab and classes != self._classes:
            # Text-Embeddings neu berechnen - nur, wenn sich der Prompt aendert.
            self.model.set_classes(list(classes))
            self._classes = list(classes)

        height, width = frame_rgb.shape[:2]
        start = time.perf_counter()
        # Ultralytics erwartet BGR wie OpenCV.
        result = self.model.predict(np.ascontiguousarray(frame_rgb[:, :, ::-1]), conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        self.last_latency = time.perf_counter() - start

        protected_set = {c.strip().lower() for c in protected}
        broad_set = {c.strip().lower() for c in broad}
        wanted = {c.strip().lower() for c in classes}
        names = result.names
        detections = []
        for xyxy, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            label = names[int(cls)]
            low = label.lower()
            if low not in wanted:
                continue   # destilliertes Modell kennt mehr, als gefragt ist
            kind = ("protected" if low in protected_set else
                    "broad" if low in broad_set else "hostile")
            # Auf [0, 1000] normiert, damit dieselben Filter greifen wie bei
            # LocateAnything.
            nums = [xyxy[0] / width * 1000, xyxy[1] / height * 1000,
                    xyxy[2] / width * 1000, xyxy[3] / height * 1000]
            det = _to_pixels(nums, width, height, label, kind)
            if det is not None:
                detections.append(det)
        self.last_raw = f"{len(detections)} Boxen"
        return _dedup(mark_unknown(detections))
