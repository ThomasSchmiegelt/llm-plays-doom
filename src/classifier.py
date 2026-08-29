"""Einordnung unbekannter Wesen durch LocateAnything selbst.

Das Modell beschriftet nicht, es lokalisiert. Als Klassifikator laesst es sich
trotzdem einsetzen: Man bietet ihm mehrere Beschreibungen an und schaut, auf
welche es eine Box an die Stelle des verfolgten Wesens legt.

Ein einzelnes Bild reicht dafuer nicht - die Antwort schwankt. Deshalb wird
ueber ein Beobachtungsfenster abgestimmt. Dieselbe Idee hat bei den geschuetzten
Menschen die Eigenbeschuesse von 19 auf 0 gebracht; hier soll sie aus einem
verrauschten Einzelurteil eine belastbare Einordnung machen.
"""
import random
from dataclasses import dataclass

import cv2
import numpy as np

from detector import Detection, _iou
from tracking import Track

# Feldfuehrer: konkrete Beschreibungen dessen, was auftauchen kann, jeweils mit
# der Einschaetzung. Beschreibend statt kategorisch - dieselbe Lehre wie bei
# den Hauptklassen ("brown insect monster" schlaegt "imp" deutlich).
DESCRIPTIVE_HYPOTHESES = [
    # An den Freedoom-Sprites ausgerichtet: die Gegner dort wirken mechanisch,
    # nicht daemonisch. "armored humanoid monster with horns" ging deshalb
    # daneben, "silver armored mechanical soldier" trifft besser.
    ("silver armored mechanical soldier", "dangerous"),
    ("man in dark uniform holding a gun", "dangerous"),
    ("skeletal creature with green armor", "dangerous"),
    ("large spider with many legs", "dangerous"),
    ("dark metal barrel", "harmless"),
    ("grey stone pillar", "harmless"),
    ("green striped column", "harmless"),
    ("bare dead tree", "harmless"),
]

# Der Schnitt, den das Modell offenbar am besten trifft: belebt gegen
# unbelebt. In dieser Arena faellt er fast mit gefaehrlich/harmlos zusammen -
# ausser bei den freundlich gesetzten Monstern, dem bekannten Haertefall.
ANIMATE_HYPOTHESES = [
    ("a living creature with limbs", "dangerous"),
    ("an armored figure standing upright", "dangerous"),
    ("a monster with a head and a body", "dangerous"),
    ("a metal barrel", "harmless"),
    ("a stone column", "harmless"),
    ("a tree trunk", "harmless"),
    ("a lamp or torch", "harmless"),
]

# Ohne jede Vorabbeschreibung: kann das Modell die Funktion statt des Aussehens
# greifen? Offen, und genau deshalb gemessen. Vier Begriffe, weil das Modell
# bei ein oder zwei Klassen in eine Wiederholungsschleife geraet.
ABSTRACT_HYPOTHESES = [
    ("a dangerous monster", "dangerous"),
    ("an aggressive creature attacking", "dangerous"),
    ("a harmless object", "harmless"),
    ("an inanimate decoration", "harmless"),
]

# Kombination: die gefaehrlichen Beschreibungen aus dem einen Satz, die
# harmlosen aus dem anderen - gemessen war jeder Satz auf genau einer Seite
# stark. Dazu zwei Ausweichbeschreibungen: ohne sie ordnet das Modell auch
# einer Steinsaeule noch einen "mechanical soldier" zu, weil es gezwungen ist,
# eine der angebotenen Klassen zu waehlen.
COMBINED_HYPOTHESES = [
    ("silver armored mechanical soldier", "dangerous"),
    ("man in dark uniform holding a gun", "dangerous"),
    ("skeletal creature with green armor", "dangerous"),
    ("large spider with many legs", "dangerous"),
    ("a metal barrel", "harmless"),
    ("a stone column", "harmless"),
    ("a tree trunk", "harmless"),
    ("a lamp or torch", "harmless"),
]

DANGEROUS = "dangerous"
HARMLESS = "harmless"


@dataclass
class Verdict:
    kind: str          # dangerous | harmless
    phrase: str        # die Beschreibung, die gewonnen hat
    confidence: float  # Stimmenanteil
    probes: int


class UnknownClassifier:
    """Fragt den Detektor mit Hypothesen ab und zaehlt die Stimmen aus."""

    def __init__(self, detector, hypotheses=None, mode: str = "crop",
                 window: int = 6, majority: float = 0.7, crop_margin: float = 0.25,
                 crop_size: int = 320, duel_size: int = 4):
        self.detector = detector
        self.hypotheses = list(hypotheses or DESCRIPTIVE_HYPOTHESES)
        self.mode = mode              # crop | frame
        self.window = window
        self.majority = majority
        self.crop_margin = crop_margin
        self.crop_size = crop_size
        self.duel_size = duel_size
        self.phrases = [phrase for phrase, _ in self.hypotheses]
        self._verdict_of = {phrase: kind for phrase, kind in self.hypotheses}
        self._pools = {
            DANGEROUS: [p for p, v in self.hypotheses if v == DANGEROUS],
            HARMLESS: [p for p, v in self.hypotheses if v == HARMLESS],
        }
        self.probe_latency = 0.0
        self._plan = self._schedule()

    def _schedule(self) -> list[list[str]]:
        """Fester Spielplan, der jede Paarung genau einmal ansetzt.

        Treten alle Beschreibungen auf einmal an, gewinnen einzelne von ihnen
        unabhaengig vom Bildinhalt - eine Steinsaeule wurde so sechsmal von
        sechs als "silver armored mechanical soldier" gemeldet. Deshalb tritt
        pro Frame nur ein Quartett an, je zur Haelfte aus beiden Toepfen.

        Entscheidend ist, die Paarungen zu *planen* statt zu wuerfeln. Zufaellig
        gezogene Quartette brauchen fuer die vollstaendige Abdeckung aller
        16 Kreuzpaare im Mittel 12,3 Frames, im schlechtesten Fall 44. Der
        Spielplan unten schafft sie in 4 - dem theoretischen Minimum, denn ein
        Quartett deckt hoechstens 2x2 Paare ab. Bei rund 850 ms je Abfrage sind
        das 3,4 statt 10,5 Sekunden bis zur Entscheidung.
        """
        half = max(1, self.duel_size // 2)
        blocks = {}
        for kind in (DANGEROUS, HARMLESS):
            pool = self._pools[kind]
            blocks[kind] = [pool[i:i + half] for i in range(0, len(pool), half)] or [[]]
        plan = []
        for left in blocks[DANGEROUS]:
            for right in blocks[HARMLESS]:
                field = list(left) + list(right)
                if field:
                    plan.append(field)
        return plan or [self.phrases[:self.duel_size]]

    def duel(self, index: int) -> list[str]:
        """Das Teilfeld fuer die Abfrage Nummer `index`.

        Der Spielplan wird durchlaufen und danach wiederholt; die Reihenfolge
        innerhalb eines Quartetts wechselt, damit die Position im Prompt nicht
        selbst zum Vorteil wird.
        """
        plan = self._plan
        field = list(plan[index % len(plan)])
        random.Random(index).shuffle(field)
        return field

    # -- Einzelne Probe --------------------------------------------------
    def probe(self, frame_rgb: np.ndarray, track: Track,
              field: list[str] | None = None) -> str | None:
        """Welche Beschreibung landet in diesem Bild auf dem Wesen?"""
        field = field or self.phrases
        if self.mode == "crop":
            phrase = self._probe_crop(frame_rgb, track, field)
        else:
            phrase = self._probe_frame(frame_rgb, track, field)
        self.probe_latency = self.detector.last_latency
        return phrase

    def _probe_crop(self, frame_rgb, track, field):
        """Auf dem Ausschnitt: weniger Bildinhalt, der ablenken kann."""
        crop = _crop(frame_rgb, track.box, self.crop_margin, self.crop_size)
        if crop is None:
            return None
        found = self.detector.detect(crop, classes=field, protected=[], broad=[])
        if not found:
            return None
        # Das Wesen fuellt den Ausschnitt - die groesste Box gewinnt.
        return max(found, key=lambda d: d.area).label

    def _probe_frame(self, frame_rgb, track, field):
        """Auf dem ganzen Bild: die Box mit der besten Ueberdeckung gewinnt."""
        found = self.detector.detect(frame_rgb, classes=field,
                                     protected=[], broad=[])
        best, best_iou = None, 0.2
        for det in found:
            value = _iou(det, track.box)
            if value > best_iou:
                best, best_iou = det, value
        return best.label if best else None

    # -- Abstimmung ------------------------------------------------------
    def observe(self, frame_rgb: np.ndarray, track: Track) -> Verdict | None:
        """Eine Beobachtung hinzufuegen. Gibt das Urteil zurueck, sobald es steht."""
        phrase = self.probe(frame_rgb, track, self.duel(track.probes))
        if phrase is not None and self._verdict_of.get(phrase) is not None:
            track.votes[phrase] = track.votes.get(phrase, 0) + 1
        return self.decide(track)

    def decide(self, track: Track) -> Verdict | None:
        """Steht schon eine Mehrheit, oder ist das Fenster ausgeschoepft?

        Gewertet wird auf Ebene der Toepfe, nicht der einzelnen Beschreibung:
        weil das Teilfeld von Frame zu Frame wechselt, treten nicht immer
        dieselben Phrasen an. Die Beschreibung fuer den Prompt ist dann die
        haeufigste innerhalb des siegreichen Topfes.
        """
        total = track.probes
        if total == 0:
            return None
        by_pool: dict[str, int] = {}
        for phrase, count in track.votes.items():
            kind = self._verdict_of.get(phrase)
            if kind:
                by_pool[kind] = by_pool.get(kind, 0) + count
        if not by_pool:
            return None
        kind, count = max(by_pool.items(), key=lambda kv: kv[1])
        phrase = max((p for p in track.votes if self._verdict_of.get(p) == kind),
                     key=lambda p: track.votes[p])
        share = count / total
        # Nicht entscheiden, bevor der Spielplan einmal durch ist. Sonst steht
        # nach zwei uebereinstimmenden Proben rechnerisch eine Mehrheit von
        # 100 %, waehrend die Haelfte der Paarungen nie stattgefunden hat -
        # gemessen kostete das 13 Prozentpunkte Trefferquote.
        enough = share >= self.majority and total >= len(self._plan)
        if not enough and total < self.window:
            return None
        verdict = Verdict(kind=kind, phrase=phrase, confidence=share, probes=total)
        track.verdict = verdict.kind
        track.verdict_phrase = verdict.phrase
        return verdict

    def progress(self, track: Track) -> tuple[int, int]:
        """Fortschritt gegen das, was mindestens noetig ist."""
        return (track.probes, max(self.window, len(self._plan)))


def _crop(frame_rgb: np.ndarray, box: Detection, margin: float, target: int):
    """Schneidet das Wesen mit etwas Rand aus und skaliert es hoch."""
    height, width = frame_rgb.shape[:2]
    pad_x = int((box.x2 - box.x1) * margin)
    pad_y = int((box.y2 - box.y1) * margin)
    x1, y1 = max(0, box.x1 - pad_x), max(0, box.y1 - pad_y)
    x2, y2 = min(width, box.x2 + pad_x), min(height, box.y2 + pad_y)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame_rgb[y1:y2, x1:x2]
    # Doom-Sprites sind grob gerastert - beim Hochskalieren die Kanten
    # erhalten statt sie weichzuzeichnen.
    scale = target / max(crop.shape[0], crop.shape[1])
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(crop)
