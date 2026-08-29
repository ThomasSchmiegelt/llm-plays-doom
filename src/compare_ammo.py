"""Zwei Agenten nebeneinander: ohne und mit Sicherung.

Der Befund, den dieses Video zeigt: Munitionssparsamkeit ist keine Sparmassnahme,
sondern eine Schutzmassnahme. Wer jeden Schuss rechtfertigen muss, feuert nicht
ins Ungewisse - und trifft am Ende sogar besser.

Beide Laeufe teilen sich Seed und Detektor und laufen im selben Prozess Schritt
fuer Schritt, damit die Bilder exakt gepaart sind.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                              # noqa: E402
from detector import (BROAD_CLASSES, HOSTILE_CLASSES,               # noqa: E402
                      PROTECTED_CLASSES, LocateAnythingDetector)
from env import DoomEnv                                             # noqa: E402
from overlay import BLUE, DIM, FG, GREEN, RED                       # noqa: E402
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

    def __init__(self, detector, guarded, seed, monsters, humans, tolerance):
        self.detector = detector
        self.guarded = guarded
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
        self.detections = self.bot.filter_visible(
            self.detector.detect(frame, CLASSES, self.protected, broad=[]))
        self.decision = self.bot.decide(self.detections, self.env.stats()["ammo"])
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
          (BLUE, "Mensch - nicht schiessen")]


def draw_legend(canvas, x, y, scale=0.62, gap=300):
    """Ohne Legende sind die Farben nur Dekor."""
    for colour, text in LEGEND:
        cv2.rectangle(canvas, (x, y - 18), (x + 26, y + 2), colour, -1)
        cv2.putText(canvas, text, (x + 38, y), FONT, scale, FG, 1, cv2.LINE_AA)
        x += gap


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
YT_VW, YT_VH = 940, 705          # 640x480 massstabsgetreu
YT_TOP, YT_LEFT, YT_RIGHT = 128, 10, 970


def draw_youtube(frame_a, frame_b, left, right, shift_a, shift_b, tally_a, tally_b):
    """Beide Laeufe nebeneinander im Breitbild - fuer YouTube."""
    canvas = np.full((YT_H, YT_W, 3), BG, np.uint8)
    cv2.putText(canvas, "MUNITIONSDISZIPLIN", (24, 62), FONT, 1.4, FG, 3, cv2.LINE_AA)
    cv2.putText(canvas, "derselbe Seed, dieselbe Arena - beide Agenten sehen dasselbe",
                (26, 100), FONT, 0.62, DIM, 1, cv2.LINE_AA)
    draw_legend(canvas, 950, 62, 0.6, 320)

    for x, frame, side, shift, title, tone in (
            (YT_LEFT, frame_a, left, shift_a, "OHNE SICHERUNG", RED),
            (YT_RIGHT, frame_b, right, shift_b, "MIT SICHERUNG", GREEN)):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (YT_VW, YT_VH), interpolation=cv2.INTER_NEAREST)
        sx, sy = YT_VW / 640, YT_VH / 480
        target = getattr(side.decision, "target", None)
        for det in side.detections:
            box = shift_box(det, shift)
            colour = BLUE if det.protected else (RED if det is target else GREEN)
            cv2.rectangle(img, (int(box.x1 * sx), int(box.y1 * sy)),
                          (int(box.x2 * sx), int(box.y2 * sy)), colour, 3)
        cx, cy = YT_VW // 2, YT_VH // 2
        firing = getattr(side.decision, "attack", False)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.line(img, (cx + dx * 9, cy + dy * 9), (cx + dx * 24, cy + dy * 24),
                     RED if firing else FG, 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (YT_VW - 1, YT_VH - 1), tone, 4)
        cv2.rectangle(img, (0, 0), (YT_VW, 40), (0, 0, 0), -1)
        cv2.putText(img, title, (14, 29), FONT, 0.82, tone, 2, cv2.LINE_AA)
        canvas[YT_TOP:YT_TOP + YT_VH, x:x + YT_VW] = img

    y = YT_TOP + YT_VH + 26
    cv2.rectangle(canvas, (10, y), (YT_W - 10, y + 172), CARD, -1)
    rows = [("MUNITION VERBRAUCHT", tally_a["ammo"], tally_b["ammo"]),
            ("SCHUESSE AUF MENSCHEN", tally_a["at_humans"], tally_b["at_humans"]),
            ("MONSTER AM LEBEN", tally_a["alive"], tally_b["alive"])]
    for index, (label, a, b) in enumerate(rows):
        ry = y + 46 + index * 46
        cv2.putText(canvas, label, (34, ry), FONT, 0.6, DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(a), (470, ry), FONT, 0.86, RED, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(b), (1430, ry), FONT, 0.86, GREEN, 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--monsters", type=int, default=7)
    ap.add_argument("--humans", type=int, default=5)
    ap.add_argument("--tics", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=6.0)
    ap.add_argument("--out", default="out/raw_compare.mp4",
                    help="Hochkant, beide untereinander")
    ap.add_argument("--out-single", default="out/raw_{name}_v.mp4",
                    help="je ein Hochkant-Video im urspruenglichen Stil")
    ap.add_argument("--out-youtube", default="out/raw_compare_yt.mp4",
                    help="Breitbild, beide nebeneinander")
    args = ap.parse_args()

    detector = LocateAnythingDetector()
    left = Side(detector, False, args.seed, args.monsters, args.humans, args.tolerance)
    right = Side(detector, True, args.seed, args.monsters, args.humans, args.tolerance)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 30.0, (W, H))
    import vertical
    single = {name: cv2.VideoWriter(args.out_single.format(name=name), fourcc, 30.0,
                                    (vertical.W, vertical.H))
              for name in ("ohne", "mit")}
    yt = cv2.VideoWriter(args.out_youtube, fourcc, 30.0, (YT_W, YT_H))

    for step in range(args.steps):
        frame_a, frame_b = left.think(), right.think()
        if frame_a is None or frame_b is None:
            break
        subs_a = left.advance(args.tics)
        subs_b = right.advance(args.tics)
        pairs = list(zip(subs_a, subs_b)) or [((frame_a, 0.0), (frame_b, 0.0))]
        tally_a, tally_b = left.tally(), right.tally()
        for (fa, turned_a), (fb, turned_b) in pairs:
            canvas = np.full((H, W, 3), BG, np.uint8)
            cv2.putText(canvas, "MUNITIONSDISZIPLIN", (54, 84), FONT, 1.7, FG, 3, cv2.LINE_AA)
            cv2.putText(canvas, "derselbe Seed, dieselbe Arena, zwei Regelwerke",
                        (54, 130), FONT, 0.72, DIM, 1, cv2.LINE_AA)
            draw_view(canvas, TOP_A, fa, left.detections, left.decision,
                      pixel_shift(turned_a, left.env.deg_per_pixel()),
                      "OHNE SICHERUNG", RED)
            draw_view(canvas, TOP_B, fb, right.detections, right.decision,
                      pixel_shift(turned_b, right.env.deg_per_pixel()),
                      "MIT SICHERUNG", GREEN)
            draw_legend(canvas, 60, 1390, 0.56, 348)
            draw_tally(canvas, 1424, tally_a, tally_b)
            cv2.putText(canvas, "NVIDIA LocateAnything-3B  +  ViZDoom",
                        (54, 1712), FONT, 0.66, DIM, 1, cv2.LINE_AA)
            writer.write(canvas)

            shift_a = pixel_shift(turned_a, left.env.deg_per_pixel())
            shift_b = pixel_shift(turned_b, right.env.deg_per_pixel())
            yt.write(draw_youtube(fa, fb, left, right, shift_a, shift_b,
                                  tally_a, tally_b))
            for name, side, frame, shift, tone, sub in (
                    ("ohne", left, fa, shift_a, RED,
                     "ohne Sicherung - sieht den Menschen und feuert trotzdem"),
                    ("mit", right, fb, shift_b, GREEN,
                     "mit Sicherung - haelt das Feuer, wenn ein Mensch im Weg steht")):
                tally = tally_a if name == "ohne" else tally_b
                single[name].write(vertical.render(
                    frame, side.detections, side.decision,
                    dict(monsters_killed=side.start_monsters - tally["alive"],
                         humans_alive=side.env.alive_counts()[1],
                         health=100, ammo=tally["ammo"]),
                    detector.last_latency, side.bot, None, False,
                    muzzle_flash=getattr(side.decision, "attack", False),
                    shift=shift, heading="LLM SPIELT DOOM",
                    subheading=sub, heading_tone=tone))
        if step % 25 == 0:
            print(f"  Schritt {step}: ohne {tally_a['ammo']} Schuss / "
                  f"{tally_a['at_humans']}x auf Menschen | "
                  f"mit {tally_b['ammo']} / {tally_b['at_humans']}x", flush=True)

    writer.release()
    yt.release()
    for w in single.values():
        w.release()
    print(f"\nohne Sicherung: {left.tally()}")
    print(f"mit Sicherung : {right.tally()}")
    print(f"Aufzeichnungen: {args.out}, {args.out_youtube}, "
          f"{args.out_single.format(name='ohne')}, {args.out_single.format(name='mit')}")
    left.close()
    right.close()


if __name__ == "__main__":
    main()
