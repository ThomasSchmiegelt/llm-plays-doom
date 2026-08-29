"""Misst die Einordnung unbekannter Wesen gegen die Engine-Wahrheit.

Die eine Frage, die dieses Skript beantwortet: Bringt das Beobachtungsfenster
etwas gegenueber einem einzelnen Blick? Der Aufbau ist derselbe wie bei den
anderen Messungen - die Engine kennt die Wahrheit, das Modell wird dagegen
gehalten.

Getrennt ausgewiesen werden die Haertefaelle: ein mit summonfriend gesetztes
Monster sieht pixelgleich aus wie ein feindliches. Rein visuelle Ueberwachung
kann die beiden nicht trennen, und das gehoert ins Ergebnis statt kaschiert.
"""
import argparse
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                              # noqa: E402
from classifier import (ABSTRACT_HYPOTHESES, DESCRIPTIVE_HYPOTHESES,  # noqa: E402
                        UnknownClassifier)
from detector import BROAD_CLASSES, HOSTILE_CLASSES, PROTECTED_CLASSES  # noqa: E402
from env import UNKNOWN_ACTORS, DoomEnv                             # noqa: E402
from results import record                                          # noqa: E402
from tracking import Tracker                                        # noqa: E402

CLASSES = HOSTILE_CLASSES + PROTECTED_CLASSES + BROAD_CLASSES


class Result:
    def __init__(self):
        self.single_hits = 0
        self.single_total = 0
        self.window_hits = 0
        self.window_total = 0
        self.not_detected = 0
        self.steps = []
        self.latency = []
        self.behaviour_agrees = 0
        self.behaviour_total = 0

    def add(self, other):
        for name in ("single_hits", "single_total", "window_hits", "window_total",
                     "not_detected", "behaviour_agrees", "behaviour_total"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.steps += other.steps
        self.latency += other.latency
        return self


def observe_one(detector, hypotheses, mode, window, actor, friendly, truth,
                seed, max_steps=14) -> Result:
    """Ein Wesen aufstellen, hinschauen, einordnen - und mitschreiben."""
    result = Result()
    env = DoomEnv(scenario="arena_mixed", show_window=False, seed=seed,
                  monsters=0, humans=0, invulnerable=True)
    env.new_episode()
    if env.spawn_unknown(actor, friendly, truth) is None:
        env.close()
        return result

    bot = AimBot(env.width, env.height, env.deg_per_pixel())
    classifier = UnknownClassifier(detector, hypotheses, mode=mode, window=window)
    tracker = Tracker(env.deg_per_pixel(), env.width, ttl=8)
    turn = 0.0
    decided = False

    for step in range(max_steps):
        frame = env.frame()
        if frame is None or env.finished():
            break
        found = bot.filter_visible(
            detector.detect(frame, CLASSES, PROTECTED_CLASSES, BROAD_CLASSES))
        tracker.advance(turn)
        tracker.update([d for d in found if d.unknown],
                       step=step, health=env.stats()["health"])
        tracks = tracker.tracks()
        if not tracks:
            turn = 24.0   # nichts gesehen - weitersuchen
            env.step(turn_deg=turn, tics=3)
            continue

        track = max(tracks, key=lambda t: t.box.area)
        phrase = classifier.probe(frame, track)
        result.latency.append(classifier.probe_latency)
        if phrase is not None:
            # Einzelblick: was haette dieser eine Frame allein ergeben?
            single = classifier._verdict_of.get(phrase)
            result.single_total += 1
            result.single_hits += int(single == truth)
            track.votes[phrase] = track.votes.get(phrase, 0) + 1

        verdict = classifier.decide(track)
        if verdict is not None:
            result.window_total += 1
            result.window_hits += int(verdict.kind == truth)
            result.steps.append(step + 1)
            # Verhaltenskanal - nur zum Vergleich, nicht zur Entscheidung.
            behaviour = "dangerous" if (track.harmed() or track.approach_rate() > 0.5) \
                else "harmless"
            result.behaviour_total += 1
            result.behaviour_agrees += int(behaviour == verdict.kind)
            decided = True
            break

        # Auf das Wesen zudrehen, damit der Ausschnitt etwas taugt.
        turn = (track.box.cx - env.width / 2) * env.deg_per_pixel()
        env.step(turn_deg=turn, tics=3)

    if not decided and result.single_total == 0:
        result.not_detected += 1
    env.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2, help="Laeufe je Aktor")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--hypotheses", nargs="+", default=["descriptive"],
                    choices=["descriptive", "abstract"])
    ap.add_argument("--probe-mode", nargs="+", default=["crop"],
                    choices=["crop", "frame"])
    ap.add_argument("--actors", nargs="+", default=[],
                    help="nur diese Aktoren messen")
    args = ap.parse_args()

    from detector import LocateAnythingDetector
    detector = LocateAnythingDetector()

    actors = [r for r in UNKNOWN_ACTORS if not args.actors or r[0] in args.actors]
    sets = {"descriptive": DESCRIPTIVE_HYPOTHESES, "abstract": ABSTRACT_HYPOTHESES}

    for name in args.hypotheses:
        for mode in args.probe_mode:
            print(f"\n=== Hypothesen: {name} ({len(sets[name])})  Abfrage: {mode}  "
                  f"Fenster: {args.window} ===")
            print(f"{'Aktor':<20}{'Wahrheit':<11}{'Einzelbild':>12}{'Fenster':>10}"
                  f"{'Schritte':>10}{'ms':>7}")
            print("-" * 72)
            total, hard, rows = Result(), Result(), []
            for actor, friendly, truth in actors:
                per = Result()
                for k in range(args.repeats):
                    per.add(observe_one(detector, sets[name], mode, args.window,
                                        actor, friendly, truth, args.seed + k))
                label = actor + ("*" if friendly else "")
                single = _pct(per.single_hits, per.single_total)
                win = _pct(per.window_hits, per.window_total)
                steps = f"{statistics.mean(per.steps):.1f}" if per.steps else "-"
                ms = f"{statistics.median(per.latency) * 1000:.0f}" if per.latency else "-"
                rows.append(dict(actor=label, truth=truth, hypotheses=name, mode=mode,
                                 single=single, window=win, steps=steps, ms=ms,
                                 hard_case=bool(friendly)))
                print(f"{label:<20}{truth:<11}{single:>12}{win:>10}{steps:>10}{ms:>7}",
                      flush=True)
                (hard if friendly else total).add(per)

            print("-" * 72)
            print(f"{'GESAMT (normal)':<31}{_pct(total.single_hits, total.single_total):>12}"
                  f"{_pct(total.window_hits, total.window_total):>10}"
                  f"{_mean(total.steps):>10}{_med(total.latency):>7}")
            if hard.window_total:
                print(f"{'HAERTEFAELLE (summonfriend)':<31}"
                      f"{_pct(hard.single_hits, hard.single_total):>12}"
                      f"{_pct(hard.window_hits, hard.window_total):>10}"
                      f"{_mean(hard.steps):>10}{_med(hard.latency):>7}")
            if total.behaviour_total:
                agree = _pct(total.behaviour_agrees, total.behaviour_total)
                print(f"\nVerhalten stimmt mit dem Bild-Urteil ueberein: {agree}")
            if total.not_detected:
                print(f"Gar nicht als unbekannt erkannt: {total.not_detected}")
            path = record("eval_unknown",
                          dict(vars(args), hypothesis_set=name, probe_mode=mode),
                          rows, notes="Einzelbild gegen Beobachtungsfenster, "
                                      "Haertefaelle (summonfriend) getrennt gefuehrt.")
            print(f"Gesichert: {path}")

    print("\nEinzelbild = Anteil der einzelnen Proben, die richtig lagen.")
    print("Fenster    = Anteil der Faelle, in denen die Abstimmung richtig lag.")


def _pct(hits, total):
    return f"{hits / total:.0%}" if total else "-"


def _mean(values):
    return f"{statistics.mean(values):.1f}" if values else "-"


def _med(values):
    return f"{statistics.median(values) * 1000:.0f}" if values else "-"


if __name__ == "__main__":
    main()
