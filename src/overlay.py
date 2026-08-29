"""Visualisierung: was das Modell sieht, was es erkennt, was der Agent tut."""
from dataclasses import replace

import cv2
import numpy as np

from detector import Detection
from tracking import shift_box

PANEL_W = 340
BG = (18, 18, 22)
FG = (235, 235, 240)
DIM = (140, 140, 150)
GREEN = (90, 220, 120)      # feindlich erkannt
RED = (70, 70, 245)         # aktuelles Ziel
BLUE = (255, 180, 60)       # geschuetzt - Mensch
AMBER = (60, 190, 245)      # Feuer gesperrt
CYAN = (200, 200, 90)       # Engine-Ground-Truth
VIOLET = (230, 120, 190)    # unbekannt - noch kein Urteil

_STATE_COLOR = {"SCANNING": DIM, "AIMING": AMBER, "FIRING": RED,
                "HOLD_FIRE": BLUE, "OBSERVING": VIOLET}
_STATE_TEXT = {
    "SCANNING": "SUCHE",
    "AIMING": "ZIELT",
    "FIRING": "FEUER",
    "HOLD_FIRE": "FEUER GESPERRT",
    "OBSERVING": "BEOBACHTET",
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def render(frame_rgb, detections, decision, stats, latency, classes, bot,
           ground_truth=None, show_gt=False, muzzle_flash=False,
           shift: float = 0.0) -> np.ndarray:
    """Baut das Anzeigebild: Spielbild mit Boxen links, Infopanel rechts (BGR)."""
    detections, decision, remembered = _shifted(
        detections, decision, bot.memory.remembered(), shift)
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    state = getattr(decision, "state", "-")
    target = getattr(decision, "target", None)
    blockers = getattr(decision, "blockers", []) or []

    if state == "HOLD_FIRE":
        _corridor(img, w, h, bot)

    if show_gt and ground_truth:
        for gt in ground_truth:
            _dashed_rect(img, (gt.x1, gt.y1), (gt.x2, gt.y2), CYAN, 1)

    # Erinnerte Menschen zuerst, gestrichelt - sie sind in diesem Frame
    # nicht erkannt worden, gelten aber weiter als geschuetzt.
    for index, det in enumerate(sorted(remembered, key=lambda d: d.area, reverse=True)):
        _dashed_rect(img, (det.x1, det.y1), (det.x2, det.y2), BLUE, 2)
        if index == 0:   # nur die groesste beschriften, sonst Textsalat
            _label(img, "MENSCH (erinnert)", det.x1, det.y1, BLUE)

    # Geschuetzte und Unbekanntes zuletzt zeichnen, damit sie oben liegen.
    for det in sorted(detections, key=lambda d: (d.protected, d.unknown)):
        if det.unknown:
            color, text = VIOLET, "UNBEKANNT"
            thickness = 3 if det in blockers else 2
            if det in blockers and getattr(decision, "progress", None):
                seen, window = decision.progress
                text = f"UNBEKANNT {seen}/{window}"
        elif det.protected:
            color, text = BLUE, "MENSCH - nicht schiessen"
            thickness = 3 if det in blockers else 2
        elif det is target:
            color, text, thickness = RED, det.label, 2
        else:
            color, text, thickness = GREEN, det.label, 1
        cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
        _label(img, text, det.x1, det.y1, color)
        if det.unknown and getattr(decision, "progress", None) and det in blockers:
            _progress(img, det, decision.progress, VIOLET)
        if det is target:
            # Zielmarke: Linie vom Fadenkreuz zur Boxmitte macht das Anvisieren
            # sichtbar, auch wenn das Ziel noch weit aussen steht.
            cv2.line(img, (w // 2, h // 2), (int(det.cx), int(det.cy)), RED, 1, cv2.LINE_AA)
            cv2.circle(img, (int(det.cx), int(det.cy)), 6, RED, 2, cv2.LINE_AA)

    _crosshair(img, w // 2, h // 2, RED if muzzle_flash else FG)
    if muzzle_flash:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), RED, 4)
    elif state == "HOLD_FIRE":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), BLUE, 4)
    elif state == "OBSERVING":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), VIOLET, 4)

    _banner(img, _STATE_TEXT.get(state, state), _STATE_COLOR.get(state, FG))
    panel = _panel(h, detections, decision, stats, latency, classes, bot,
                   show_gt, ground_truth, remembered)
    return np.hstack([img, panel])


def _panel(height, detections, decision, stats, latency, classes, bot,
           show_gt, ground_truth, remembered=()):
    p = np.full((height, PANEL_W, 3), BG, np.uint8)
    y = 30

    def line(text, color=FG, scale=0.44, dy=18, x=16):
        nonlocal y
        cv2.putText(p, text, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)
        y += dy

    def section(title):
        nonlocal y
        y += 7
        cv2.putText(p, title, (16, y), _FONT, 0.4, DIM, 1, cv2.LINE_AA)
        y += 6
        cv2.line(p, (16, y), (PANEL_W - 16, y), (55, 55, 62), 1)
        y += 17

    hostile = [d for d in detections if not d.protected and not d.unknown]
    protected = [d for d in detections if d.protected]
    unknown = [d for d in detections if d.unknown]

    line("LLM PLAYS DOOM", FG, 0.6, 24)
    line("LocateAnything-3B  ->  Ziel & Schuss", DIM, 0.36, 12)

    section("WAHRNEHMUNG")
    line(f"Feinde erkannt:   {len(hostile)}", GREEN)
    count = len(remembered)
    suffix = f"  (+{count} erinnert)" if count else ""
    line(f"Menschen erkannt: {len(protected)}{suffix}", BLUE)
    if unknown or bot.learned:
        line(f"Unbekannt:        {len(unknown)}", VIOLET)
    line(f"Inferenz: {latency * 1000:5.0f} ms  ({_fps(latency)} FPS)", FG)
    if show_gt and ground_truth is not None:
        line(f"Engine-Referenz: {len(ground_truth)} Akteure", CYAN)

    section("ENTSCHEIDUNG")
    state = getattr(decision, "state", "-")
    line(_STATE_TEXT.get(state, state), _STATE_COLOR.get(state, FG), 0.56, 23)
    for chunk in _wrap(getattr(decision, "reason", ""), 36):
        line(chunk, DIM, 0.36, 15)

    section("BILANZ")
    line(f"Schuesse:       {bot.shots}", FG)
    line(f"Feuer gesperrt: {bot.holds}", BLUE)
    if bot.learned:
        gefaehrlich = sum(1 for _, v in bot.learned if v == "dangerous")
        line(f"Gelernt: {len(bot.learned)}  "
             f"({gefaehrlich} gefaehrl., {len(bot.learned) - gefaehrlich} harmlos)", VIOLET)
    line(f"Monster erlegt: {stats.get('monsters_killed', 0)}", GREEN)
    # Eigenbeschuss laut Engine-Wahrheit: gefeuert, waehrend ein Mensch im
    # Schusskorridor stand. Menschen, die von Monstern getoetet werden,
    # zaehlen hier bewusst nicht mit.
    ff = stats.get("friendly_fire", 0)
    line(f"Eigenbeschuss:  {ff}", RED if ff else GREEN)

    section("LAGE")
    _bar(p, 16, y, PANEL_W - 32, 10, stats.get("health", 0) / 100.0, GREEN)
    y += 10
    line(f"Health {stats.get('health', 0):.0f}    Munition {stats.get('ammo', 0):.0f}",
         FG, 0.42, 20, x=18)
    line(f"Monster am Leben:  {stats.get('monsters_alive', 0)}", FG, 0.42)
    line(f"Menschen am Leben: {stats.get('humans_alive', 0)}", FG, 0.42)

    y = height - 34
    cv2.line(p, (16, y - 14), (PANEL_W - 16, y - 14), (55, 55, 62), 1)
    grown = f"  (+{len(bot.learned)} gelernt)" if bot.learned else ""
    line(f"Prompt: {len(classes)} Klassen{grown}", DIM, 0.34, 15)
    line("[q] beenden  [g] Ground Truth  [space] Pause", DIM, 0.34, 15)
    return p


def _progress(img, det, progress, color):
    """Balken unter der Box: wie weit ist die Abstimmung?"""
    seen, window = progress
    width = max(24, det.x2 - det.x1)
    x, y = det.x1, min(img.shape[0] - 6, det.y2 + 4)
    cv2.rectangle(img, (x, y), (x + width, y + 4), (50, 50, 58), -1)
    filled = int(width * min(1.0, seen / max(1, window)))
    cv2.rectangle(img, (x, y), (x + filled, y + 4), color, -1)


def _corridor(img, w, h, bot):
    """Faerbt den Streifen ein, in dem kein Mensch stehen darf."""
    half = bot.fire_tolerance_px + bot.safety_margin_px
    band = img[:, int(w / 2 - half):int(w / 2 + half)]
    band[:] = (band * 0.55 + np.array(BLUE) * 0.45).astype(np.uint8)


def _fps(latency):
    return f"{1.0 / latency:.1f}" if latency > 0 else "-"


def _bar(img, x, y, w, h, frac, color):
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 62), -1)
    if frac > 0:
        cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)


def _crosshair(img, cx, cy, color):
    cv2.line(img, (cx - 12, cy), (cx - 4, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + 4, cy), (cx + 12, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 12), (cx, cy - 4), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + 4), (cx, cy + 12), color, 1, cv2.LINE_AA)


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.38, 1)
    top = max(0, y - th - 6)
    x = min(x, img.shape[1] - tw - 10)
    cv2.rectangle(img, (x, top), (x + tw + 8, top + th + 6), color, -1)
    cv2.putText(img, text, (x + 4, top + th + 1), _FONT, 0.38, (20, 20, 20), 1, cv2.LINE_AA)


def _banner(img, text, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 18), _FONT, 0.5, color, 1, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, thickness, dash=6):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for yy in range(y1, y2, dash * 2):
        cv2.line(img, (x1, yy), (x1, min(yy + dash, y2)), color, thickness)
        cv2.line(img, (x2, yy), (x2, min(yy + dash, y2)), color, thickness)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines[:3]


def _shifted(detections, decision, remembered, shift):
    """Fuehrt alle Boxen um `shift` Pixel mit.

    Zwischen zwei Entscheidungen dreht sich die Kamera weiter. Die Boxen
    stammen aber vom Entscheidungsbild. Weil der gedrehte Winkel bekannt ist,
    laesst sich die Verschiebung exakt nachziehen - dieselbe Rechnung, die
    auch das Schutz-Gedaechtnis benutzt.
    """
    if not shift:
        return detections, decision, remembered
    detections = [shift_box(d, shift) for d in detections]
    remembered = [shift_box(d, shift) for d in remembered]
    target = getattr(decision, "target", None)
    observed = getattr(decision, "observed", None)
    decision = replace(
        decision,
        target=shift_box(target, shift) if target else None,
        blockers=[shift_box(b, shift) for b in getattr(decision, "blockers", [])],
        observed=(replace(observed, box=shift_box(observed.box, shift))
                  if observed is not None else None),
    )
    return detections, decision, remembered
