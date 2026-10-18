"""LocateAnything-3B als Open-Vocabulary-Detektor.

Das Modell bekommt ein Bild und eine Textbeschreibung der gesuchten Klassen
und antwortet mit Boxen im Format <box><x1><y1><x2><y2></box>, wobei die
Koordinaten auf [0, 1000] normiert sind.
"""
import re
import sys
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
                 dtype: torch.dtype = torch.bfloat16, max_new_tokens: int = 96,
                 decoding: str = "guided"):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        assert decoding in DECODING_MODES, decoding
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.decoding = decoding
        self.box_threshold = GUIDED_BOX_THRESHOLD
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
        if self.decoding == "guided":
            raw = guided_generate(self.model, self.tokenizer, inputs, classes,
                                  box_threshold=self.box_threshold)
            self.last_latency = time.perf_counter() - start
            self.last_raw = raw
            return parse_boxes(raw, width, height, classes, protected, broad)
        with torch.no_grad():
            raw = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                generation_mode=self.decoding,
                use_cache=True,   # vom Modell zwingend gefordert
            )
        self.last_latency = time.perf_counter() - start
        self.last_raw = raw
        return parse_boxes(raw, width, height, classes, protected, broad)


# guided  - eigene Schleife, siehe guided_generate (Default)
# hybrid  - Modellcode: Mehrtoken-Vorhersage, bei Fehlern Einzeltoken
# slow    - Modellcode: nur Einzeltoken, sauber, aber langsam
DECODING_MODES = ("guided", "hybrid", "slow")

# Mehr Boxen je Klasse kommen in der Arena nicht vor. Die Grenze faengt
# Schleifen ab, die keine exakte Wiederholung sind.
_GUIDED_MAX_BOXES = 10

# Ab dieser Wahrscheinlichkeit fuer <box> laesst der Schiedsrichter eine
# weitere Box zu. Niedriger findet mehr Monster, laesst aber auch mehr
# Phantome durch; gemessen in src/eval_detector.py --box-threshold.
GUIDED_BOX_THRESHOLD = 0.05


@torch.no_grad()
def guided_generate(model, tokenizer, inputs, classes: list[str],
                    n_future: int = 6, max_boxes: int = _GUIDED_MAX_BOXES,
                    box_threshold: float = GUIDED_BOX_THRESHOLD) -> str:
    """Dekodiert die Boxen mit vorgegebener Struktur.

    Der Hybridmodus des Modells hat auf Doom-Frames zwei Schwaechen, die beide
    auf dasselbe hinauslaufen: nach den echten Boxen wiederholt er eine Box
    endlos, bis das Token-Limit greift. Das kostet Zeit, und - schlimmer -
    die Klassen hinter der Schleife kommen gar nicht mehr dran. Steht die
    geschuetzte Klasse dort, wird nach Menschen nie gefragt.

    Das Modell arbeitet die Klassen stets in Prompt-Reihenfolge ab. Deshalb
    setzt diese Schleife die <ref>Klasse</ref>-Tokens selbst ein - sie
    laufen im naechsten Forward-Schritt einfach mit, kosten also nichts -
    und wertet eine wiederholte Box als Ende der Klasse. So kommt jede Klasse
    garantiert an die Reihe.

    Die Bausteine (Sampling, Musterpruefung) stammen aus dem Modellcode;
    nachgebaut ist nur die Steuerung aus LocateAnything.generate().
    """
    mod = sys.modules[type(model).__module__]
    tid = model.token_ids
    device = inputs["input_ids"].device

    def encode(text):
        return torch.tensor([tokenizer.encode(text, add_special_tokens=False)],
                            dtype=torch.long, device=device)

    refs = [encode(f"<ref>{c}</ref>") for c in classes]
    masks = torch.full((1, n_future - 1), tid["default_mask_token_id"],
                       dtype=torch.long, device=device)
    positions = torch.arange(0, inputs["input_ids"].shape[1] + 1024,
                             device=device).unsqueeze(0)

    grid = inputs.get("image_grid_hws")
    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid).to(device, dtype=torch.int32)
    vit = model.extract_feature(inputs["pixel_values"].to(model.language_model.dtype), grid)
    vit = model.mlp1(torch.cat(vit, dim=0))

    prompt_len = inputs["input_ids"].shape[1]
    generated = torch.cat([inputs["input_ids"], refs[0]], dim=1)
    next_class, seen, ar_mode, past = 1, set(), False, None

    def append(tokens):
        nonlocal generated
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=device).view(1, -1)
        generated = torch.cat([generated, tokens], dim=1)

    for step in range(16 * len(classes)):
        start = past[0][0].size(2) if past is not None else 0
        if ar_mode:
            seq = generated
            pos = positions[:, start:seq.size(1)]
        else:
            # Letztes Token doppelt plus Masken: so verlangt es die
            # Mehrtoken-Vorhersage des Modells.
            seq = torch.cat([generated, generated[:, -1:], masks], dim=1)
            pos = positions[:, start:seq.size(1)].clone()
            pos[0, -n_future:] -= 1
        prep = model.language_model.prepare_inputs_for_generation(
            seq, past, None, inputs_embeds=None, use_cache=True, position_ids=pos)
        if step == 0:
            prep.update(visual_features=vit,
                        image_token_index=model.config.image_token_index)
        out = model.language_model(**prep)
        past = tuple((k[:, :, :generated.size(1)], v[:, :, :generated.size(1)])
                     for k, v in out.past_key_values)

        if ar_mode:
            # Einzeltoken nur, um eine missratene Box zu Ende zu schreiben.
            _, _, x0, _ = mod.sample_tokens(out.logits[:, -1:], generated, tid)
            token = x0[0, 0].item()
            if (tid["coord_start_token_id"] <= token <= tid["coord_end_token_id"]
                    or token == tid["none_token_id"]):
                append([token])
                continue
            ar_mode = False
            if token == tid["box_end_token_id"]:
                append([token])
                continue
            kind = "class_done"
        else:
            # Schiedsrichter: An der Position des letzten echten Tokens steht
            # die gewoehnliche Einzeltoken-Vorhersage - im selben
            # Forward-Schritt, also umsonst. Sie entscheidet, ob ueberhaupt
            # noch eine Box folgt. Die Mehrtoken-Vorhersage allein meldet
            # sonst Phantomboxen entlang der Horizontlinie, wo der langsame
            # Modus laengst aufgehoert haette.
            p_box = out.logits[0, -n_future - 1].float().softmax(-1)[tid["box_start_token_id"]]
            if p_box.item() < box_threshold:
                kind = "class_done"
            else:
                _, _, x0, box_avg = mod.sample_tokens(out.logits[:, -n_future:], generated,
                                                      tid, keep_k=5)
                new = x0[0] if (box_avg[0] == 0).all() else box_avg[0]
                pattern = mod.handle_pattern(new, tid, "hybrid")
                kind, tokens = pattern["type"], pattern["tokens"]
            if kind == "coord_box":
                key = tuple(tokens.tolist() if torch.is_tensor(tokens) else tokens)
                if key not in seen and len(seen) < max_boxes:
                    seen.add(key)
                    append(tokens)
                    continue
                kind = "class_done"   # Wiederholung: die Klasse ist erschoepft
            elif kind == "point_box":
                append(tokens)
                continue
            elif kind == "error_box":
                append(tokens)
                ar_mode = True
                continue
            elif kind == "empty_box":
                append(tokens)        # <box>None</box>: Klasse nicht im Bild

        # Klasse abgeschlossen (leer, wiederholt, neues <ref> oder Ende):
        # die naechste Klasse wird vorgegeben, statt dem Modell zu folgen.
        if next_class >= len(classes):
            break
        append(refs[next_class])
        next_class += 1
        seen = set()

    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=False) + "<|im_end|>"


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
