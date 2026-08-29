"""Hochkant-Ansicht 1080x1920 fuer YouTube Shorts und Instagram Reels.

Andere Anforderungen als die Desktop-Ansicht: alles muss auf einem Telefon
aus einem Meter Entfernung lesbar sein. Deshalb wenige, grosse Elemente
statt eines dichten Infopanels.
"""
from dataclasses import replace

import cv2
import numpy as np

from tracking import shift_box
from overlay import AMBER, BLUE, CYAN, DIM, FG, GREEN, RED, VIOLET

W, H = 1080, 1920
BG = (16, 16, 20)
CARD = (30, 30, 36)

# Shorts und Reels legen ihre eigene Bedienoberflaeche ueber die unteren rund
# 300 px und den rechten Rand. Alles Wesentliche bleibt darueber.
SAFE_BOTTOM = 1620

GAME_TOP = 230
GAME_W, GAME_H = 1080, 810      # 640x480 exakt um 1.6875 skaliert
SCALE = GAME_W / 640

_FONT = cv2.FONT_HERSHEY_DUPLEX
_STATE_TEXT = {
    "SCANNING": "SUCHT ZIEL",
    "AIMING": "ZIELT",
    "FIRING": "FEUER",
    "HOLD_FIRE": "FEUER GESPERRT",
    "OBSERVING": "BEOBACHTET",
}
_STATE_COLOR = {"SCANNING": DIM, "AIMING": AMBER, "FIRING": RED,
                "HOLD_FIRE": BLUE, "OBSERVING": VIOLET}


def render(frame_rgb, detections, decision, stats, latency, bot,
           ground_truth=None, show_gt=False, muzzle_flash=False,
           shift: float = 0.0, heading: str | None = None,
           subheading: str | None = None, heading_tone=None) -> np.ndarray:
    detections, decision, remembered = _shifted(
        detections, decision, bot.memory.remembered(), shift)
    canvas = np.full((H, W, 3), BG, np.uint8)
    state = getattr(decision, "state", "-")
    color = _STATE_COLOR.get(state, FG)
    # Im Urteilsmoment uebernimmt die Einordnung die Farbfuehrung des Bildes.
    verdict = getattr(decision, "verdict", None)
    if verdict:
        color = RED if verdict == "dangerous" else BLUE

    _header(canvas, heading, subheading, heading_tone)
    game = _game_view(frame_rgb, detections, decision, bot, ground_truth,
                      show_gt, muzzle_flash, state, verdict, remembered)
    canvas[GAME_TOP:GAME_TOP + GAME_H] = game

    y = GAME_TOP + GAME_H + 46
    label = (("GEFAEHRLICH" if verdict == "dangerous" else "HARMLOS")
             if verdict else _STATE_TEXT.get(state, state))
    source = getattr(decision, "verdict_source", None)
    if source:
        label = f"{label}  ({source})"
    y = _state_block(canvas, y, label, color, getattr(decision, "reason", ""))
    y = _tiles(canvas, y, stats, bot)
    _legend(canvas, y)
    _footer(canvas, latency)
    return canvas


# -- Bausteine -----------------------------------------------------------
def _header(canvas, heading=None, subheading=None, tone=None):
    _text(canvas, heading or "LLM SPIELT DOOM", 54, 84, tone or FG, 1.9, 3)
    _text(canvas, subheading or "erkennt Monster - verschont Menschen",
          54, 144, DIM, 0.95, 1)
    cv2.line(canvas, (54, 186), (W - 54, 186), (60, 60, 70), 2)


def _game_view(frame_rgb, detections, decision, bot, ground_truth, show_gt,
               muzzle_flash, state, verdict=None, remembered=()):
    # Erst hochskalieren, dann zeichnen - so bleiben Linien und Schrift
    # scharf, waehrend die Doom-Pixel bewusst pixelig bleiben.
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)

    if state == "HOLD_FIRE":
        half = int((bot.fire_tolerance_px + bot.safety_margin_px) * SCALE)
        band = img[:, GAME_W // 2 - half:GAME_W // 2 + half]
        band[:] = (band * 0.6 + np.array(BLUE) * 0.4).astype(np.uint8)

    if show_gt and ground_truth:
        for gt in ground_truth:
            cv2.rectangle(img, _pt(gt.x1, gt.y1), _pt(gt.x2, gt.y2), CYAN, 2)

    target = getattr(decision, "target", None)
    observed = getattr(decision, "observed", None)
    judged = observed.box if (observed is not None and verdict) else None

    # Erinnerte Menschen: alle als gestrichelte Box, aber nur die groesste
    # beschriftet - sonst stapeln sich die Textfahnen unleserlich.
    for index, det in enumerate(sorted(remembered, key=lambda d: d.area, reverse=True)):
        _dashed(img, _pt(det.x1, det.y1), _pt(det.x2, det.y2), BLUE, 4)
        if index == 0:
            _tag(img, "MENSCH (erinnert)", det.x1, det.y1, BLUE)

    for det in sorted(detections, key=lambda d: (d.protected, d.unknown)):
        if det.unknown:
            progress = getattr(decision, "progress", None)
            # Das Urteil gilt nur dem Wesen, das gerade beobachtet wurde -
            # andere Unbekannte im Bild bleiben unbekannt.
            if judged is not None and _same_box(det, judged):
                col, thick = (RED, 6) if verdict == "dangerous" else (BLUE, 6)
                text = "GEFAEHRLICH" if verdict == "dangerous" else "HARMLOS"
            else:
                col, thick = VIOLET, 5
                text = f"UNBEKANNT {progress[0]}/{progress[1]}" if progress else "UNBEKANNT"
        elif det.protected:
            col, text, thick = BLUE, "MENSCH", 5
        elif det is target:
            col, text, thick = RED, _short(det.label), 5
        else:
            col, text, thick = GREEN, _short(det.label), 3
        cv2.rectangle(img, _pt(det.x1, det.y1), _pt(det.x2, det.y2), col, thick)
        _tag(img, text, det.x1, det.y1, col)
        if det is target:
            cv2.line(img, (GAME_W // 2, GAME_H // 2), _pt(det.cx, det.cy), RED, 3, cv2.LINE_AA)

    _crosshair(img, GAME_W // 2, GAME_H // 2, RED if muzzle_flash else FG)
    border = (RED if muzzle_flash else BLUE if state == "HOLD_FIRE"
              else VIOLET if state == "OBSERVING" else (60, 60, 70))
    if verdict:
        border = RED if verdict == "dangerous" else BLUE
    cv2.rectangle(img, (0, 0), (GAME_W - 1, GAME_H - 1), border, 6)
    return img


def _state_block(canvas, y, label, color, reason):
    # Der Zusatz in Klammern nennt die Quelle des Urteils und darf kleiner sein.
    scale = 2.1 if "(" not in label else 1.5
    (tw, th), _ = cv2.getTextSize(label, _FONT, scale, 4)
    cv2.rectangle(canvas, (54, y), (54 + tw + 44, y + th + 40), color, -1)
    cv2.putText(canvas, label, (76, y + th + 18), _FONT, scale, (16, 16, 20), 4, cv2.LINE_AA)
    y += th + 74
    for chunk in _wrap(reason, 46)[:2]:
        _text(canvas, chunk, 54, y, DIM, 0.92, 1)
        y += 44
    return y + 24


def _tiles(canvas, y, stats, bot):
    items = [
        ("MONSTER\nERLEGT", str(int(stats.get("monsters_killed", 0))), GREEN),
        ("MENSCHEN\nVERSCHONT", str(int(stats.get("humans_alive", 0))), BLUE),
        # Sobald der Agent etwas eingeordnet hat, ist das die interessantere
        # Zahl als die Feuersperren.
        (("NEU\nGELERNT", str(len(bot.learned)), VIOLET) if bot.learned
         else ("FEUER\nGESPERRT", str(bot.holds), AMBER)),
    ]
    items = [i if isinstance(i, tuple) and len(i) == 3 else i for i in items]
    gap, tile_w, tile_h = 24, (W - 108 - 48) // 3, 210
    for index, (title, value, color) in enumerate(items):
        x = 54 + index * (tile_w + gap)
        cv2.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), CARD, -1)
        cv2.rectangle(canvas, (x, y), (x + 8, y + tile_h), color, -1)
        _text(canvas, value, x + 30, y + 96, color, 2.4, 4)
        for line_index, part in enumerate(title.split("\n")):
            _text(canvas, part, x + 30, y + 146 + line_index * 36, DIM, 0.72, 1)
    return y + tile_h + 44


def _legend(canvas, y):
    """Eine Zeile statt drei - der Platz nach unten ist knapp."""
    entries = [(GREEN, "Monster"), (RED, "Ziel"), (BLUE, "Mensch"),
               (VIOLET, "unbekannt")]
    x = 54
    for color, text in entries:
        cv2.rectangle(canvas, (x, y - 24), (x + 36, y + 4), color, -1)
        _text(canvas, text, x + 50, y, FG, 0.86, 1)
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.86, 1)
        x += 50 + tw + 56


def _footer(canvas, latency):
    """Steht bewusst unterhalb des sicheren Bereichs - reine Fussnote."""
    y = SAFE_BOTTOM + 90
    fps = f"{1.0 / latency:.1f}" if latency > 0 else "-"
    _text(canvas, "NVIDIA LocateAnything-3B  +  ViZDoom", 54, y, DIM, 0.74, 1)
    _text(canvas, f"{latency * 1000:.0f} ms / Frame  ({fps} FPS)", 54, y + 38, DIM, 0.74, 1)


def title_card(lines, subtitle="") -> np.ndarray:
    """Standbild fuer den Einstieg - die ersten Sekunden entscheiden."""
    canvas = np.full((H, W, 3), BG, np.uint8)
    block = 130 + 100 * (len(lines) - 1)
    y = (SAFE_BOTTOM - block) // 2 + 60
    for index, line in enumerate(lines):
        scale = 2.6 if index == 0 else 1.7
        color = FG if index == 0 else GREEN if index == 1 else BLUE
        _text(canvas, line, 70, y, color, scale, 4 if index == 0 else 3)
        y += 130 if index == 0 else 100
    if subtitle:
        _text(canvas, subtitle, 70, y + 40, DIM, 0.95, 1)
    return canvas


# -- Kleinkram -----------------------------------------------------------
def _same_box(a, b, tolerance: int = 6) -> bool:
    return (abs(a.x1 - b.x1) <= tolerance and abs(a.y1 - b.y1) <= tolerance
            and abs(a.x2 - b.x2) <= tolerance and abs(a.y2 - b.y2) <= tolerance)


def _short(label: str, limit: int = 22) -> str:
    """Lange Prompt-Formulierungen passen nicht neben eine Box."""
    label = label.upper()
    return label if len(label) <= limit else label[:limit - 1] + "."


def _pt(x, y):
    return (int(x * SCALE), int(y * SCALE))


def _text(canvas, text, x, y, color, scale, thickness):
    cv2.putText(canvas, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _tag(img, text, x, y, color):
    scale, thickness = 0.78, 2
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    px, py = _pt(x, y)
    px = max(0, min(px, GAME_W - tw - 16))
    top = max(0, py - th - 18)
    cv2.rectangle(img, (px, top), (px + tw + 16, top + th + 16), color, -1)
    cv2.putText(img, text, (px + 8, top + th + 6), _FONT, scale, (16, 16, 20),
                thickness, cv2.LINE_AA)


def _crosshair(img, cx, cy, color):
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.line(img, (cx + dx * 12, cy + dy * 12), (cx + dx * 34, cy + dy * 34),
                 color, 3, cv2.LINE_AA)


def _dashed(img, p1, p2, color, thickness, dash=18):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


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
    return lines


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
