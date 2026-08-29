"""LocateAnything-3B als Open-Vocabulary-Detektor.

Das Modell bekommt ein Bild und eine Textbeschreibung der gesuchten Klassen
und antwortet mit Boxen im Format <box><x1><y1><x2><y2></box>, wobei die
Koordinaten auf [0, 1000] normiert sind.
"""
import re
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
from PIL import Image

MODEL_ID = "nvidia/LocateAnything-3B"

# Vom Modell vorgegebenes Promptformat fuer Open-Vocabulary-Detection.
_PROMPT = "Locate all the instances that matches the following description: {classes}"

# Die gesuchten Klassen zerfallen in zwei Gruppen: worauf geschossen werden
# darf und was geschuetzt ist. Beide gehen in denselben Prompt - die Trennung
# passiert allein ueber das Label, das das Modell zurueckgibt.
#
# Zwei gemessene Eigenheiten des Modells stecken in dieser Liste
# (Zahlen und Vergleich in src/eval_detector.py bzw. im README):
#
# 1. Beschreibende Formulierungen schlagen semantische Kategorien deutlich.
#    "brown insect monster" trifft den Freedoom-Imp verlaesslicher als "imp",
#    und vor allem haelt das Modell die Marines dann seltener faelschlich
#    fuer Monster - 6 % statt 56 %.
# 2. Mindestens drei Klassen angeben. Bei ein oder zwei Klassen geraet das
#    Modell auf Doom-Frames in eine Wiederholungsschleife und gibt dieselbe
#    Box bis zum Token-Limit aus - Faktor 6 bei der Latenz.
HOSTILE_CLASSES = [
    "brown insect monster",     # Imp
    "pink fleshy worm creature",  # Demon
    "orange flying skull",      # Lost Soul
    "blue shelled monster",     # Cacodemon
]
PROTECTED_CLASSES = [
    "grey armored human soldier",  # Marine - darf nicht beschossen werden
]

# Absichtlich weite Klasse. Sie faengt auch das ein, wofuer der Agent noch
# keine Beschreibung hat. Eine Box dieser Klasse, die keine spezifische Box
# ueberlappt, ist ein unbekanntes Wesen - der einzige ehrliche Weg, denn die
# Engine-Labels duerfen dafuer nicht herangezogen werden.
BROAD_CLASSES = [
    "any creature or figure",
]

DEFAULT_CLASSES = HOSTILE_CLASSES + PROTECTED_CLASSES + BROAD_CLASSES

# Antwortformat: <ref>klasse</ref><box><x1><y1><x2><y2></box>...
# Fehlt die Klasse im Bild, antwortet das Modell mit <box>None</box>.
_SEG_RE = re.compile(r"<ref>(.*?)</ref>(.*?)(?=<ref>|$)", re.S)
_BOX_RE = re.compile(r"<box>(.*?)</box>", re.S)
_NUM_RE = re.compile(r"-?\d+")

# Boxen, die sich staerker als das ueberlappen, gelten als dasselbe Objekt.
_DEDUP_IOU = 0.45

# Menschliche Sprites sind in Doom immer deutlich hoeher als breit. Flache
# Streifen entlang der Horizontlinie, die das Modell gelegentlich als Mensch
# meldet, wuerden sonst den Schusskorridor dauerhaft blockieren.
_PROTECTED_MIN_ASPECT = 0.9

# Fuer Unbekanntes muss der Filter lockerer sein: darunter sind auch Faesser
# und Saeulen, die nicht dem Hochformat einer Figur folgen. Nur ausgesprochen
# flache Streifen und Winzlinge fliegen raus, sonst blockiert Rauschen den
# Schusskorridor dauerhaft.
_UNKNOWN_MIN_ASPECT = 0.35
_UNKNOWN_MIN_AREA_FRAC = 0.0012

# Reihenfolge beim Zusammenlegen: Im Zweifel gewinnt die vorsichtigere
# Einordnung. Geschuetzt schlaegt unbekannt, unbekannt schlaegt feindlich.
_KIND_PRIORITY = {"protected": 3, "unknown": 2, "hostile": 1, "broad": 0}


@dataclass
class Detection:
    """Box in Pixelkoordinaten des Eingabebildes."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    kind: str = "hostile"   # hostile | protected | unknown | broad

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def protected(self) -> bool:
        return self.kind == "protected"

    @property
    def unknown(self) -> bool:
        return self.kind == "unknown"


class LocateAnythingDetector:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, max_new_tokens: int = 96):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        self.last_latency = 0.0
        self.last_raw = ""

    def detect(self, frame_rgb: np.ndarray, classes: list[str] | None = None,
               protected: list[str] | None = None,
               broad: list[str] | None = None) -> list[Detection]:
        """Erkennt die gegebenen Klassen in einem RGB-Frame (H, W, 3).

        `broad` benennt die weiten Klassen. Was nur von diesen gefunden wird,
        gilt als unbekannt.
        """
        classes = classes or DEFAULT_CLASSES
        protected = PROTECTED_CLASSES if protected is None else protected
        broad = BROAD_CLASSES if broad is None else broad
        image = Image.fromarray(frame_rgb).convert("RGB")
        height, width = frame_rgb.shape[:2]

        # Die Klassen werden vom Modell mit </c> getrennt erwartet.
        query = _PROMPT.format(classes="</c>".join(classes))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            raw = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                generation_mode="hybrid",
                use_cache=True,   # vom Modell zwingend gefordert
            )
        self.last_latency = time.perf_counter() - start
        self.last_raw = raw
        return parse_boxes(raw, width, height, classes, protected, broad)


def parse_boxes(raw: str, width: int, height: int, classes: list[str],
                protected: list[str] | None = None,
                broad: list[str] | None = None) -> list[Detection]:
    """Wandelt die Modellantwort in Pixel-Boxen um."""
    protected_set = {c.strip().lower() for c in (protected or [])}
    broad_set = {c.strip().lower() for c in (broad or [])}
    detections: list[Detection] = []
    segments = _SEG_RE.findall(raw)
    if not segments:
        # Kein <ref>-Tag: alle Boxen der ersten Klasse zuordnen.
        segments = [(classes[0] if classes else "object", raw)]

    for label, body in segments:
        label = label.strip() or "object"
        for match in _BOX_RE.finditer(body):
            nums = [int(n) for n in _NUM_RE.findall(match.group(1))]
            if len(nums) < 4:
                continue  # "<box>None</box>" oder Punktausgabe
            low = label.lower()
            kind = ("protected" if low in protected_set else
                    "broad" if low in broad_set else "hostile")
            det = _to_pixels(nums[:4], width, height, label, kind)
            if det is not None:
                detections.append(det)

    return _dedup(mark_unknown(detections))


def mark_unknown(detections: list[Detection],
                 iou_threshold: float = 0.4) -> list[Detection]:
    """Trennt Unbekanntes von Bekanntem.

    Die weite Klasse findet auch alles, wofuer es schon eine spezifische
    Beschreibung gibt. Ueberlappt eine weite Box eine spezifische, ist sie nur
    eine zweite Sicht auf dasselbe und faellt weg. Bleibt sie allein, steht
    dort etwas, wofuer der Agent noch kein Wort hat.
    """
    specific = [d for d in detections if d.kind in ("hostile", "protected")]
    out = list(specific)
    for det in detections:
        if det.kind != "broad":
            continue
        if any(_iou(det, other) >= iou_threshold for other in specific):
            continue
        out.append(replace(det, kind="unknown"))
    return out


def _to_pixels(nums, width, height, label, kind="hostile") -> Detection | None:
    """Rechnet [0, 1000]-Koordinaten in Pixel um und verwirft Unsinn."""
    x1, y1, x2, y2 = (int(round(n / 1000 * s)) for n, s in
                      zip(nums, (width, height, width, height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    width_px, height_px = x2 - x1, y2 - y1
    if width_px < 2 or height_px < 2:
        return None
    if kind == "protected" and height_px < _PROTECTED_MIN_ASPECT * width_px:
        return None
    if kind == "broad":
        if height_px < _UNKNOWN_MIN_ASPECT * width_px:
            return None
        if width_px * height_px < _UNKNOWN_MIN_AREA_FRAC * width * height:
            return None
    return Detection(x1, y1, x2, y2, label, kind)


def _dedup(detections: list[Detection]) -> list[Detection]:
    """Das Modell gibt dasselbe Objekt oft mehrfach aus - groesste Box gewinnt.

    Bei Ueberschneidung gewinnt die vorsichtigere Einordnung - geschuetzt vor
    unbekannt vor feindlich. Im Zweifel lieber nicht schiessen.
    """
    kept: list[Detection] = []
    for det in sorted(detections,
                      key=lambda d: (_KIND_PRIORITY.get(d.kind, 0), d.area),
                      reverse=True):
        if all(_iou(det, other) < _DEDUP_IOU for other in kept):
            kept.append(det)
    return kept


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
