"""Drei Agenten nebeneinander: wild, streng und gewichtet.

Der Befund, den dieses Video zeigt: Munitionssparsamkeit ist keine Sparmassnahme,
sondern eine Schutzmassnahme. Wer jeden Schuss rechtfertigen muss, feuert nicht
ins Ungewisse - und trifft am Ende sogar besser.

  wild        kein Regelwerk - alles ist Ziel, auch Menschen und Unbekanntes
  streng      Feuer haelt an, bis ein Unbekanntes eingeordnet ist
  gewichtet   dasselbe Regelwerk, aber Beobachten laeuft nebenher, und was
              einmal eingeordnet wurde, bleibt gemerkt

Die dritte Variante ist die Antwort auf einen gemessenen Fehlschlag: streng
ausgelegt kam der Agent bei fuenf Auftritten auf 4 Schuss und 0 Treffer in
160 Schritten - handlungsunfaehig vor lauter Vorsicht.

Alle Laeufe teilen sich Seed und Detektor und laufen im selben Prozess Schritt
fuer Schritt, damit die Bilder exakt gepaart sind.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                              # noqa: E402
from classifier import DESCRIPTIVE_HYPOTHESES, UnknownClassifier    # noqa: E402
from detector import (BROAD_CLASSES, HOSTILE_CLASSES,               # noqa: E402
                      PROTECTED_CLASSES, LocateAnythingDetector)
from env import DoomEnv                                             # noqa: E402
from overlay import BLUE, DIM, FG, GREEN, RED, VIOLET               # noqa: E402
from tracking import pixel_shift, shift_box                         # noqa: E402

W, H = 1080, 1920
BG = (16, 16, 20)
CARD = (30, 30, 36)
VIEW_W, VIEW_H = 1080, 608          # 640x360-Ausschnitt, hochskaliert
CROP_TOP = 60                       # 4:3 auf 16:9 beschneiden
CROP_H = 360
SX, SY = VIEW_W / 640, VIEW_H / CROP_H
TOP_A, TOP_B = 168, 800
FONT = cv2.FONT_HERSHEY_DUPLEX
CLASSES = HOSTILE_CLASSES + PROTECTED_CLASSES


class Side:
    """Ein Lauf mit eigener Umgebung, eigenem Agenten, eigener Bilanz."""

    def __init__(self, detector, mode, seed, monsters, humans, tolerance,
                 window=5):
        self.detector = detector
        self.mode = mode
        guarded = mode != "wild"
        self.env = DoomEnv(scenario="arena_mixed", show_window=False, seed=seed,
                           monsters=monsters, humans=humans, invulnerable=True)
        self.bot = AimBot(self.env.width, self.env.height, self.env.deg_per_pixel(),
                          fire_tolerance_px=tolerance, can_move=self.env.can_move)
        if not guarded:
            # Wichtig: die geschuetzte Klasse bleibt im Prompt. Beide Agenten
            # sehen dasselbe und markieren Menschen blau - nur einer haelt sich
            # daran. Sonst waere nicht vergleichbar, was hier verglichen wird:
            # das Regelwerk, nicht die Wahrnehmung.
            self.bot.respect_protection = False
            self.bot.safety_margin_px = -1e9
            self.bot.memory.ttl = 1
        self.protected = PROTECTED_CLASSES
        # Beobachten kann nur, wer sich auch an das Ergebnis haelt.
        self.classifier = (UnknownClassifier(detector, DESCRIPTIVE_HYPOTHESES,
                                             mode="crop", window=window)
                           if guarded else None)
        if mode == "gewichtet":
            self.bot.observe_interrupts = False   # nebenher statt anhalten
            self.bot.learning = True              # Art merken, nicht neu pruefen
            self.bot.memory.ttl = 8               # laenger festhalten
        else:
            self.bot.learning = False
        self.env.new_episode()
        self.start_ammo = self.env.stats()["ammo"]
        self.start_hits = self.env.stats()["hits"]
        self.start_monsters = self.env.alive_counts()[0]
        self.at_humans = 0
        self.decision = None
        self.detections = []

    def think(self, health_unused=None):
        frame = self.env.frame()
        if frame is None:
            return None
        self.detections = self.bot.filter_visible(self.detector.detect(
            frame, CLASSES + BROAD_CLASSES, self.protected, BROAD_CLASSES))
        self.decision = self.bot.decide(self.detections, self.env.stats()["ammo"],
                                        frame=frame, classifier=self.classifier,
                                        health=self.env.stats()["health"])
        truth = self.env.ground_truth()
        half = self.bot.fire_tolerance_px + max(0.0, self.bot.safety_margin_px)
        left, right = self.env.width / 2 - half, self.env.width / 2 + half
        if self.decision.attack and any(b.kind == "human" and b.x2 >= left and b.x1 <= right
                                        for b in truth):
            self.at_humans += 1
        return frame

    def advance(self, tics):
        return self.env.step_capture(turn_deg=self.decision.turn_deg,
                                     attack=self.decision.attack, tics=tics)

    def tally(self):
        stats = self.env.stats()
        ammo = int(self.start_ammo - stats["ammo"])
        hits = int(stats["hits"] - self.start_hits)
        return dict(ammo=ammo, hits=hits, at_humans=self.at_humans,
                    alive=self.env.alive_counts()[0],
                    rate=(hits / ammo) if ammo else 0.0)

    def close(self):
        self.env.close()


LEGEND = [(GREEN, "Monster erkannt"), (RED, "aktuelles Ziel"),
          (BLUE, "Mensch - nicht schiessen"), (VIOLET, "unbekannt - beobachten")]


def draw_legend(canvas, x, y, scale=0.62, per_row=4, row_gap=34):
    """Ohne Legende sind die Farben nur Dekor.

    Die Spaltenbreite richtet sich nach dem laengsten Eintrag, sonst laeuft
    "unbekannt - beobachten" aus dem Bild.
    """
    width = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for _, t in LEGEND) + 62
    for index, (colour, text) in enumerate(LEGEND):
        cx = x + (index % per_row) * width
        cy = y + (index // per_row) * row_gap
        cv2.rectangle(canvas, (cx, cy - 17), (cx + 24, cy + 1), colour, -1)
        cv2.putText(canvas, text, (cx + 34, cy), FONT, scale, FG, 1, cv2.LINE_AA)


def draw_view(canvas, top, frame_rgb, detections, decision, shift, title, tone):
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)[CROP_TOP:CROP_TOP + CROP_H]
    img = cv2.resize(img, (VIEW_W, VIEW_H), interpolation=cv2.INTER_NEAREST)

    target = getattr(decision, "target", None)
    for det in detections:
        box = shift_box(det, shift)
        col = BLUE if det.protected else (RED if det is target else GREEN)
        p1 = (int(box.x1 * SX), int((box.y1 - CROP_TOP) * SY))
        p2 = (int(box.x2 * SX), int((box.y2 - CROP_TOP) * SY))
        cv2.rectangle(img, p1, p2, col, 4)

    centre = (VIEW_W // 2, VIEW_H // 2)
    firing = getattr(decision, "attack", False)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.line(img, (centre[0] + dx * 10, centre[1] + dy * 10),
                 (centre[0] + dx * 28, centre[1] + dy * 28),
                 RED if firing else FG, 3, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (VIEW_W - 1, VIEW_H - 1), tone, 5)

    cv2.rectangle(img, (0, 0), (VIEW_W, 46), (0, 0, 0), -1)
    cv2.putText(img, title, (16, 33), FONT, 0.95, tone, 2, cv2.LINE_AA)
    canvas[top:top + VIEW_H] = img


def draw_tally(canvas, y, left, right):
    # Trefferquote steht bewusst NICHT hier: der ungesicherte Agent kommt auf
    # 82 %, weil seine Treffer ueberwiegend nahe Marines sind. Die Zahl wuerde
    # ihn besser aussehen lassen, als er ist.
    rows = [("MUNITION VERBRAUCHT", str(left["ammo"]), str(right["ammo"])),
            ("SCHUESSE AUF MENSCHEN", str(left["at_humans"]), str(right["at_humans"])),
            ("MONSTER AM LEBEN", str(left["alive"]), str(right["alive"]))]
    cv2.rectangle(canvas, (54, y), (W - 54, y + 196), CARD, -1)
    for index, (label, a, b) in enumerate(rows):
        row_y = y + 52 + index * 56
        cv2.putText(canvas, label, (78, row_y), FONT, 0.62, DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, a, (660, row_y), FONT, 0.95, RED, 2, cv2.LINE_AA)
        cv2.putText(canvas, b, (880, row_y), FONT, 0.95, GREEN, 2, cv2.LINE_AA)


YT_W, YT_H = 1920, 1080
YT_VW, YT_VH = 620, 465
YT_TOP = 138
YT_X = (18, 650, 1282)
TONES = {"wild": RED, "streng": BLUE, "gewichtet": GREEN}
TITLES = {"wild": "WILD - kein Regelwerk",
          "streng": "STRENG - haelt an zum Beobachten",
          "gewichtet": "GEWICHTET - beobachtet nebenher"}


def draw_youtube(frames, sides, shifts, tallies):
    """Alle Varianten nebeneinander im Breitbild - fuer YouTube."""
    canvas = np.full((YT_H, YT_W, 3), BG, np.uint8)
    cv2.putText(canvas, "DREI REGELWERKE", (24, 58), FONT, 1.3, FG, 3, cv2.LINE_AA)
    cv2.putText(canvas, "derselbe Seed, dieselbe Arena - alle drei sehen dasselbe",
                (26, 96), FONT, 0.6, DIM, 1, cv2.LINE_AA)
    draw_legend(canvas, 660, 54, 0.56, 2)

    for x, frame, side, shift in zip(YT_X, frames, sides, shifts):
        tone = TONES[side.mode]
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (YT_VW, YT_VH), interpolation=cv2.INTER_NEAREST)
        sx, sy = YT_VW / 640, YT_VH / 480
        target = getattr(side.decision, "target", None)
        for det in side.detections:
            box = shift_box(det, shift)
            colour = (VIOLET if det.unknown else BLUE if det.protected
                      else RED if det is target else GREEN)
            cv2.rectangle(img, (int(box.x1 * sx), int(box.y1 * sy)),
                          (int(box.x2 * sx), int(box.y2 * sy)), colour, 2)
        cx, cy = YT_VW // 2, YT_VH // 2
        firing = getattr(side.decision, "attack", False)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.line(img, (cx + dx * 7, cy + dy * 7), (cx + dx * 18, cy + dy * 18),
                     RED if firing else FG, 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (YT_VW - 1, YT_VH - 1), tone, 3)
        cv2.rectangle(img, (0, 0), (YT_VW, 32), (0, 0, 0), -1)
        cv2.putText(img, TITLES[side.mode], (10, 23), FONT, 0.52, tone, 1, cv2.LINE_AA)
        canvas[YT_TOP:YT_TOP + YT_VH, x:x + YT_VW] = img

    y = YT_TOP + YT_VH + 30
    cv2.rectangle(canvas, (18, y), (YT_W - 18, y + 224), CARD, -1)
    rows = [("MUNITION VERBRAUCHT", "ammo"), ("SCHUESSE AUF MENSCHEN", "at_humans"),
            ("EIGENE TREFFER", "hits"), ("MONSTER AM LEBEN", "alive")]
    for index, (label, key) in enumerate(rows):
        ry = y + 46 + index * 46
        cv2.putText(canvas, label, (40, ry), FONT, 0.58, DIM, 1, cv2.LINE_AA)
        for x, side, tally in zip(YT_X, sides, tallies):
            cv2.putText(canvas, str(tally[key]), (x + YT_VW // 2 - 10, ry),
                        FONT, 0.8, TONES[side.mode], 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--monsters", type=int, default=7)
    ap.add_argument("--humans", type=int, default=5)
    ap.add_argument("--tics", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=6.0)
    ap.add_argument("--unknowns", type=int, default=5)
    ap.add_argument("--unknown-every", type=int, default=24)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--out-single", default="out/raw_{name}_v.mp4",
                    help="je ein Hochkant-Video im urspruenglichen Stil")
    ap.add_argument("--out-youtube", default="out/raw_drei_yt.mp4",
                    help="Breitbild, alle nebeneinander")
    args = ap.parse_args()

    import vertical
    detector = LocateAnythingDetector()
    modes = ("wild", "streng", "gewichtet")
    sides = [Side(detector, m, args.seed, args.monsters, args.humans,
                  args.tolerance, args.window) for m in modes]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs("out", exist_ok=True)
    yt = cv2.VideoWriter(args.out_youtube, fourcc, 30.0, (YT_W, YT_H))
    single = {m: cv2.VideoWriter(args.out_single.format(name=m), fourcc, 30.0,
                                 (vertical.W, vertical.H)) for m in modes}

    unknowns_left, next_unknown = args.unknowns, args.unknown_every
    for step in range(args.steps):
        frames = [side.think() for side in sides]
        if any(f is None for f in frames):
            break
        # Alle Seiten teilen Seed und Beutel - dasselbe Wesen an derselben Stelle.
        if unknowns_left > 0 and step >= next_unknown:
            got = [side.env.spawn_unknown() for side in sides]
            next_unknown = step + args.unknown_every
            if got[0]:
                unknowns_left -= 1
                print(f"  Schritt {step}: unbekanntes Wesen ({got[0][0]})", flush=True)

        subs = [side.advance(args.tics) for side in sides]
        tallies = [side.tally() for side in sides]
        length = min((len(x) for x in subs), default=0)
        pairs = range(length) if length else [None]
        for index in pairs:
            if index is None:
                pack = [(f, 0.0) for f in frames]
            else:
                pack = [x[index] for x in subs]
            shifts = [pixel_shift(t, side.env.deg_per_pixel())
                      for (_, t), side in zip(pack, sides)]
            yt.write(draw_youtube([f for f, _ in pack], sides, shifts, tallies))
            for side, (frame, _), shift, tally in zip(sides, pack, shifts, tallies):
                single[side.mode].write(vertical.render(
                    frame, side.detections, side.decision,
                    dict(monsters_killed=side.start_monsters - tally["alive"],
                         humans_alive=side.env.alive_counts()[1],
                         health=100, ammo=tally["ammo"]),
                    detector.last_latency, side.bot, None, False,
                    muzzle_flash=getattr(side.decision, "attack", False),
                    shift=shift, heading=TITLES[side.mode].split(" - ")[0],
                    subheading=TITLES[side.mode].split(" - ")[1],
                    heading_tone=TONES[side.mode]))

        if step % 40 == 0:
            summary = "  ".join(f"{s.mode}: {t['ammo']}/{t['at_humans']}x"
                                for s, t in zip(sides, tallies))
            print(f"  Schritt {step}  (Munition/auf Menschen)  {summary}", flush=True)

    yt.release()
    for w in single.values():
        w.release()
    print()
    for side in sides:
        print(f"  {side.mode:<12} {side.tally()}")
        side.close()
    print(f"\nAufzeichnungen: {args.out_youtube} und "
          f"{', '.join(args.out_single.format(name=m) for m in modes)}")


if __name__ == "__main__":
    main()
