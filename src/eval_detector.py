"""Misst LocateAnything gegen die Ground Truth der Doom-Engine.

Die Engine kennt die exakten Bildschirmboxen jedes Akteurs und weiss, ob es
ein Monster oder ein Mensch ist. Damit laesst sich zweierlei beziffern:
wie gut das Modell die Sprites ueberhaupt findet, und - fuer diese Demo
entscheidend - wie oft es einen Menschen faelschlich als Monster labelt.
"""
import argparse
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                              # noqa: E402
from detector import (HOSTILE_CLASSES, PROTECTED_CLASSES,           # noqa: E402
                      Detection, LocateAnythingDetector, _iou)
from env import DoomEnv                                             # noqa: E402
from results import record                                          # noqa: E402

IOU_MATCH = 0.35


def collect(scenario, width, height, count, seed, monsters, humans):
    """Sammelt Frames samt Ground Truth, indem der Agent sich langsam dreht."""
    env = DoomEnv(scenario=scenario, width=width, height=height, show_window=False,
                  seed=seed, monsters=monsters, humans=humans, invulnerable=True)
    env.new_episode()
    samples = []
    while len(samples) < count and not env.finished():
        env.step(turn_deg=23, tics=3)
        truth = env.ground_truth()
        if truth:
            samples.append((env.frame().copy(), truth))
    env.close()
    return samples


def evaluate(detector, samples, classes, protected, bot):
    """Ordnet jeder Engine-Box die beste Detektion zu und zaehlt aus."""
    stat = collections.Counter()
    latencies = []
    for frame, truth in samples:
        # broad=[] haelt diese Messung bei Monster gegen Mensch - das
        # Unbekannte hat mit src/eval_unknown.py ein eigenes Skript.
        dets = bot.filter_visible(detector.detect(frame, classes, protected, broad=[]))
        latencies.append(detector.last_latency)
        for box in truth:
            best, best_iou = None, 0.0
            for det in dets:
                value = _iou(det, Detection(box.x1, box.y1, box.x2, box.y2, ""))
                if value > best_iou:
                    best, best_iou = det, value
            if best is None or best_iou < IOU_MATCH:
                stat[f"{box.kind}_miss"] += 1
            else:
                stat[f"{box.kind}_as_{'human' if best.protected else 'monster'}"] += 1
    return stat, latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="arena_mixed")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--monsters", type=int, default=7)
    ap.add_argument("--humans", type=int, default=5)
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--variants", nargs="+", default=[],
                    help="zusaetzliche Prompt-Varianten, Klassen mit | getrennt; "
                         "geschuetzte Klasse mit ! markieren")
    args = ap.parse_args()

    width, height = (int(v) for v in args.resolution.split("x"))
    samples = collect(args.scenario, width, height, args.frames, args.seed,
                      args.monsters, args.humans)
    total = sum(len(t) for _, t in samples)
    print(f"{len(samples)} Frames, {total} Engine-Boxen, Treffer ab IoU >= {IOU_MATCH}\n")

    variants = [(HOSTILE_CLASSES + PROTECTED_CLASSES, PROTECTED_CLASSES)]
    for raw in args.variants:
        classes = [c.strip() for c in raw.split("|")]
        variants.append(([c.lstrip("!") for c in classes],
                         [c[1:] for c in classes if c.startswith("!")]))

    detector = LocateAnythingDetector()
    bot = AimBot(width, height, 90.0 / width)
    print(f"{'Prompt':<46}{'Monster':>9}{'Mensch':>9}{'Mensch->Monster':>17}{'ms':>7}")
    print("-" * 88)
    rows = []
    for classes, protected in variants:
        stat, latencies = evaluate(detector, samples, classes, protected, bot)
        m_total = sum(stat[k] for k in stat if k.startswith("monster_"))
        h_total = sum(stat[k] for k in stat if k.startswith("human_"))
        m_rec = stat["monster_as_monster"] / m_total if m_total else 0.0
        h_rec = stat["human_as_human"] / h_total if h_total else 0.0
        confuse = stat["human_as_monster"] / h_total if h_total else 0.0
        ms = statistics.median(latencies) * 1000
        label = ", ".join(classes)
        rows.append(dict(classes=label, monster_recall=round(m_rec, 3),
                         human_recall=round(h_rec, 3),
                         human_as_monster=round(confuse, 3), ms=round(ms)))
        print(f"{label[:46]:<46}{m_rec:>8.0%}{h_rec:>9.0%}{confuse:>17.0%}{ms:>7.0f}")
    print("\nMonster / Mensch = Anteil korrekt gelabelt. Die letzte Spalte ist die "
          "\ngefaehrliche: ein Mensch, den das Modell fuer ein Monster haelt.")
    path = record("eval_detector", vars(args), rows,
                  notes=f"{len(samples)} Frames, {total} Engine-Boxen, IoU >= {IOU_MATCH}.")
    print(f"Gesichert: {path}")


if __name__ == "__main__":
    main()
