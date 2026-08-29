"""Verfolgung erkannter Objekte ueber mehrere Frames.

Zwei Dinge macht das noetig. Erstens uebersieht der Detektor ein Objekt
gelegentlich fuer einen einzelnen Frame - eine Entscheidung darf davon nicht
abhaengen. Zweitens braucht die Einordnung eines unbekannten Wesens mehrere
Beobachtungen, und die muessen demselben Wesen zugeordnet werden.

Weil der Agent zwischen den Frames dreht, wandern die Objekte im Bild. Der
Drehwinkel ist bekannt, also laesst sich die Verschiebung exakt herausrechnen:
eine Drehung nach rechts schiebt alles nach links.
"""
import itertools
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

# Dieselbe Ueberlappungsrechnung, die der Detektor zum Zusammenlegen nutzt -
# keine zweite Umsetzung danebenstellen.
from detector import Detection, _iou as iou

# Kantenlaenge der Miniatur, auf die der Boxinhalt normiert wird.
THUMB = 32
# Die Erkennungsbox wackelt von Frame zu Frame um einige Pixel. Ungedaempft
# uebertoent dieser Versatz die eigentliche Animation: ein stillstehendes Fass
# kam so auf einen hoeheren Wert als ein Monster. Deshalb wird der Vergleich
# ueber kleine Verschiebungen minimiert - was sich durch blosses Verrutschen
# erklaeren laesst, zaehlt nicht als Bewegung.
THUMB_SHIFT = 3


def shift_box(det: Detection, dx: float) -> Detection:
    """Verschiebt eine Box waagerecht - die Eigenbewegungs-Korrektur."""
    return replace(det, x1=int(det.x1 + dx), x2=int(det.x2 + dx))


def pixel_shift(turn_deg: float, deg_per_pixel: float) -> float:
    """Bildverschiebung, die aus einer Drehung folgt."""
    return -turn_deg / deg_per_pixel


@dataclass
class Observation:
    """Ein Blick auf ein verfolgtes Objekt."""
    step: int
    area: int
    center_x: float
    health: float


@dataclass
class Track:
    """Ein ueber mehrere Frames verfolgtes Objekt."""
    id: int
    box: Detection
    age: int = 0                      # Frames seit der letzten frischen Erkennung
    seen: int = 1                     # wie oft frisch erkannt
    history: list[Observation] = field(default_factory=list)
    # Normierte Graustufen-Ansichten des Boxinhalts, eine je frischer Sicht.
    thumbs: list[np.ndarray] = field(default_factory=list)
    votes: dict[str, int] = field(default_factory=dict)
    verdict: str | None = None        # dangerous | harmless, sobald entschieden
    verdict_phrase: str | None = None  # die Beschreibung, die gewonnen hat
    verdict_source: str | None = None  # "Aussehen" (vorlaeufig) oder "Verhalten"

    @property
    def fresh(self) -> bool:
        return self.age == 0

    @property
    def probes(self) -> int:
        return sum(self.votes.values())

    def approach_rate(self) -> float:
        """Relatives Flaechenwachstum ueber die Beobachtungen.

        Waechst die Box, kommt das Objekt naeher. Der Wert ist nur Evidenz
        fuer die Anzeige - die Einordnung trifft der Detektor.
        """
        if len(self.history) < 2:
            return 0.0
        first, last = self.history[0].area, self.history[-1].area
        return (last - first) / first if first > 0 else 0.0

    def motion_score(self) -> float:
        """Wie stark aendert sich der Bildinhalt innerhalb der Box?

        Steinsaeulen, Baeume und Faesser stehen still - ihre Ansicht bleibt
        von Frame zu Frame gleich. Kreaturen animieren sich, auch wenn sie
        nicht laufen: sie drehen sich zum Spieler, Gliedmassen bewegen sich.
        Weil die Box dem Objekt folgt und der Inhalt auf eine feste Groesse
        normiert wird, fallen Eigenbewegung und Entfernungsaenderung heraus.

        Der Median statt des Mittels, damit ein einzelner Ausreisser das
        Ergebnis nicht traegt. Rueckgabe -1, wenn es zu wenige Ansichten gibt.
        """
        if len(self.thumbs) < 3:
            return -1.0   # zu wenig Ansichten fuer eine Aussage
        diffs = [_aligned_diff(a, b) for a, b in zip(self.thumbs, self.thumbs[1:])]
        diffs.sort()
        return diffs[len(diffs) // 2]

    def harmed(self) -> bool:
        """Ist die Gesundheit gefallen, waehrend das Objekt sichtbar war?"""
        if len(self.history) < 2:
            return False
        return self.history[-1].health < self.history[0].health


class Tracker:
    """Ordnet frische Erkennungen bestehenden Spuren zu."""

    MATCH_IOU = 0.3

    def __init__(self, deg_per_pixel: float, width: int, ttl: int = 6,
                 max_tracks: int = 12):
        self.deg_per_pixel = deg_per_pixel
        self.width = width
        self.ttl = ttl
        self.max_tracks = max_tracks
        self._tracks: list[Track] = []
        self._ids = itertools.count(1)

    def advance(self, turn_deg: float):
        """Eigenbewegung herausrechnen und abgelaufene Spuren verwerfen."""
        dx = pixel_shift(turn_deg, self.deg_per_pixel)
        alive = []
        for track in self._tracks:
            if track.age + 1 >= self.ttl:
                continue
            box = shift_box(track.box, dx)
            if box.x2 < 0 or box.x1 > self.width:
                continue  # aus dem Bild gedreht
            track.box = box
            track.age += 1
            alive.append(track)
        self._tracks = alive

    def update(self, detections: list[Detection], step: int = 0, health: float = 100.0,
               frame_rgb=None):
        """Weist jede Erkennung der am besten passenden Spur zu.

        Eine Erkennung, zu der keine Spur passt, eroeffnet eine neue. Ist
        `frame_rgb` gesetzt, wird zusaetzlich eine normierte Ansicht des
        Boxinhalts abgelegt - Grundlage fuer motion_score().
        """
        taken: set[int] = set()
        for det in detections:
            best, best_iou = None, self.MATCH_IOU
            for track in self._tracks:
                if id(track) in taken:
                    continue
                value = iou(det, track.box)
                if value > best_iou:
                    best, best_iou = track, value
            if best is None:
                if len(self._tracks) >= self.max_tracks:
                    continue
                best = Track(id=next(self._ids), box=det)
                self._tracks.append(best)
            else:
                best.box = det
                best.age = 0
                best.seen += 1
            taken.add(id(best))
            best.history.append(
                Observation(step=step, area=det.area, center_x=det.cx, health=health))
            if frame_rgb is not None:
                thumb = _thumbnail(frame_rgb, det)
                if thumb is not None:
                    best.thumbs.append(thumb)

    def tracks(self, kind: str | None = None) -> list[Track]:
        if kind is None:
            return list(self._tracks)
        return [t for t in self._tracks if t.box.kind == kind]

    def boxes(self, kind: str | None = None) -> list[Detection]:
        return [t.box for t in self.tracks(kind)]

    def stale(self) -> list[Detection]:
        """Boxen, die in diesem Frame nicht frisch erkannt wurden."""
        return [t.box for t in self._tracks if t.age > 0]


def _thumbnail(frame_rgb, box: Detection):
    """Boxinhalt als normierte Graustufen-Miniatur."""
    height, width = frame_rgb.shape[:2]
    x1, y1 = max(0, box.x1), max(0, box.y1)
    x2, y2 = min(width, box.x2), min(height, box.y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = frame_rgb[y1:y2, x1:x2]
    grey = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, (THUMB, THUMB), interpolation=cv2.INTER_AREA)


def _aligned_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Kleinster Unterschied zweier Miniaturen ueber kleine Verschiebungen.

    Nur was auch nach bestmoeglicher Ausrichtung uebrig bleibt, ist echte
    Veraenderung im Objekt statt Versatz der Box.
    """
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    k = THUMB_SHIFT
    inner = a[k:-k, k:-k]
    best = None
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            window = b[k + dy:THUMB - k + dy, k + dx:THUMB - k + dx]
            value = float(np.abs(inner - window).mean())
            if best is None or value < best:
                best = value
    return best if best is not None else 0.0
