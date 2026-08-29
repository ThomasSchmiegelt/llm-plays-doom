"""Sagt das Verhalten die Wahrheit besser voraus als das Aussehen?

Der visuelle Klassifikator liegt auf allen friedlich gesetzten Akteuren zu
0 % richtig - bei vier optisch identischen Marines, von denen zwei gefaehrlich
sind, ist das Zufallsniveau. Hier wird die Gegenprobe gemessen.

Drei Merkmale werden mitgeschrieben, alle ohne weiteren Modellaufruf:

  Schaden      ist die eigene Gesundheit gefallen, waehrend das Wesen zu sehen
               war? Spaet, aber eindeutig - man weiss es erst, wenn es
               passiert ist.
  Annaeherung  waechst die Box, kommt es also naeher?
  Bewegung     aendert sich der Bildinhalt innerhalb der Box? Erst jetzt
               aussagekraeftig, weil feindliche Akteure durch den Weckschuss
               tatsaechlich aktiv sind.

Der Agent haelt das Wesen im Blick und schiesst nicht - sonst waere nicht zu
trennen, wer wen provoziert hat.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                              # noqa: E402
from detector import (BROAD_CLASSES, HOSTILE_CLASSES,               # noqa: E402
                      PROTECTED_CLASSES, LocateAnythingDetector)
from env import UNKNOWN_ROLES, DoomEnv                              # noqa: E402
from results import record                                          # noqa: E402
from tracking import Tracker                                        # noqa: E402

CLASSES = HOSTILE_CLASSES + PROTECTED_CLASSES + BROAD_CLASSES


def watch(detector, actor, friendly, truth, seed, steps, tics):
    """Ein Wesen aufstellen, beobachten, Verhalten mitschreiben."""
    env = DoomEnv(scenario="arena_mixed", show_window=False, seed=seed,
                  monsters=0, humans=0, invulnerable=False)
    env.new_episode()
    if env.spawn_unknown(actor, friendly, truth, wake=True) is None:
        env.close()
        return None

    bot = AimBot(env.width, env.height, env.deg_per_pixel())
    tracker = Tracker(env.deg_per_pixel(), env.width, ttl=10)
    start_health = env.stats()["health"]
    turn, hurt_at, seen = 0.0, None, 0
    track = None

    for step in range(steps):
        frame = env.frame()
        if frame is None or env.finished():
            break
        found = bot.filter_visible(
            detector.detect(frame, CLASSES, PROTECTED_CLASSES, BROAD_CLASSES))
        health = env.stats()["health"]
        tracker.advance(turn)
        tracker.update([d for d in found if d.unknown], step, health, frame_rgb=frame)
        tracks = tracker.tracks()
        if tracks:
            track = max(tracks, key=lambda t: t.box.area)
            seen += 1
            turn = (track.box.cx - env.width / 2) * env.deg_per_pixel()
        else:
            turn = 22.0
        if hurt_at is None and health < start_health:
            hurt_at = step
        env.step(turn_deg=turn, tics=tics)          # bewusst kein Schuss

    end_health = env.stats()["health"]
    env.close()
    if track is None:
        return None
    return dict(actor=actor, truth=truth,
                damage=int(start_health - end_health),
                hurt_at=hurt_at, seen=seen,
                approach=round(track.approach_rate(), 2),
                motion=round(track.motion_score(), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--tics", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    detector = LocateAnythingDetector()
    print(f"\n{'Rolle':<24}{'Aktor':<17}{'Wahrheit':<11}"
          f"{'Schaden':>8}{'ab Schritt':>12}{'Naeherung':>11}{'Bewegung':>10}")
    print("-" * 94)

    rows, correct, total, delays = [], 0, 0, []
    for actor, friendly, rolle, truth in UNKNOWN_ROLES:
        for run in range(args.repeats):
            result = watch(detector, actor, friendly, truth,
                           args.seed + run, args.steps, args.tics)
            if result is None:
                print(f"{rolle:<24}{actor:<17}{truth:<11}   nicht erfasst", flush=True)
                continue
            # Die Regel, die geprueft wird: Schaden bedeutet gefaehrlich.
            predicted = "dangerous" if result["damage"] > 0 else "harmless"
            hit = predicted == truth
            correct += hit
            total += 1
            if result["hurt_at"] is not None:
                delays.append(result["hurt_at"])
            rows.append(dict(rolle=rolle, **result, predicted=predicted, correct=hit))
            mark = "" if hit else "   <- falsch"
            print(f"{rolle:<24}{actor:<17}{truth:<11}{result['damage']:>8}"
                  f"{str(result['hurt_at']):>12}{result['approach']:>11.2f}"
                  f"{result['motion']:>10.2f}{mark}", flush=True)

    print("-" * 94)
    quote = correct / total if total else 0
    print(f"Regel 'Schaden = gefaehrlich': {correct}/{total} richtig ({quote:.0%})")
    if delays:
        print(f"Erster Schaden im Mittel nach {statistics.mean(delays):.1f} Schritten "
              f"(frueheste {min(delays)}, spaeteste {max(delays)})")
    print("\nZum Vergleich: das Aussehen lag auf den friedlich gesetzten Akteuren "
          "zu 0 % richtig.")

    path = record("eval_behaviour", vars(args), rows,
                  notes=f"Regel 'Schaden bedeutet gefaehrlich' gegen die "
                        f"Engine-Wahrheit: {correct}/{total}. Der Agent schiesst "
                        f"waehrend der Beobachtung nicht.")
    print(f"Gesichert: {path}")


if __name__ == "__main__":
    main()
