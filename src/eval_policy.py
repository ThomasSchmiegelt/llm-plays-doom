"""Misst, was die Sicherungen in der Schusslogik tatsaechlich bringen.

Verglichen werden drei Auspraegungen auf derselben Startaufstellung:
  ohne        - schiesst auf alles, was erkannt wird (auch auf Menschen)
  korridor    - schiesst nur auf feindliche Labels, sperrt das Feuer bei
                einem Menschen im Schusskorridor
  korridor+gedaechtnis - zusaetzlich bleiben erkannte Menschen einige Frames
                lang gueltig, auch wenn der Detektor sie kurz uebersieht

Gezaehlt wird der Eigenbeschuss nach Engine-Wahrheit: ein abgegebener Schuss,
waehrend laut Engine ein Mensch im Schusskorridor stand.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                              # noqa: E402
from detector import (HOSTILE_CLASSES, PROTECTED_CLASSES,           # noqa: E402
                      LocateAnythingDetector)
from env import DoomEnv                                             # noqa: E402
from results import record                                          # noqa: E402


def run(detector, mode, steps, seed, monsters, humans, tics,
        tolerance=14.0, burst=0):
    env = DoomEnv(scenario="arena_mixed", show_window=False, seed=seed,
                  monsters=monsters, humans=humans, invulnerable=True)
    bot = AimBot(env.width, env.height, env.deg_per_pixel(), can_move=env.can_move,
                 fire_tolerance_px=tolerance)
    if mode == "ohne":
        # Alles gilt als Ziel, keine Sperre.
        bot.safety_margin_px = -1e9
        protected_classes = []
    else:
        protected_classes = PROTECTED_CLASSES
    if mode != "korridor+gedaechtnis":
        bot.memory.ttl = 1   # nur der aktuelle Frame zaehlt

    env.new_episode()
    start_monsters, start_humans = env.alive_counts()
    start = env.stats()
    start_ammo = start["ammo"]
    friendly_fire = 0
    for _ in range(steps):
        if env.finished():
            break
        frame = env.frame()
        if frame is None:
            break
        # Ohne die breite Klasse: diese Messung vergleicht die Sicherungen
        # gegen Menschen, nicht die Erkennung von Unbekanntem.
        dets = bot.filter_visible(detector.detect(
            frame, HOSTILE_CLASSES + PROTECTED_CLASSES, protected_classes, broad=[]))
        decision = bot.decide(dets, env.stats()["ammo"])
        truth = env.ground_truth()
        half = bot.fire_tolerance_px + max(0.0, bot.safety_margin_px)
        left, right = env.width / 2 - half, env.width / 2 + half
        if decision.attack and any(b.kind == "human" and b.x2 >= left and b.x1 <= right
                                   for b in truth):
            friendly_fire += 1
        env.step(turn_deg=decision.turn_deg, attack=decision.attack, tics=tics,
                 burst=burst or None)

    monsters_alive, humans_alive = env.alive_counts()
    end = env.stats()
    killed = start_monsters - monsters_alive
    ammo_used = start_ammo - end["ammo"]
    hits = end["hits"] - start["hits"]
    env.close()
    return dict(shots=bot.shots, holds=bot.holds, friendly_fire=friendly_fire,
                ammo=int(ammo_used), hits=int(hits),
                # Zurechenbar: Anteil der Schuesse, die getroffen haben.
                # "Monster erlegt" enthaelt auch fremde Abschuesse.
                hit_rate=round(hits / ammo_used, 3) if ammo_used else None,
                damage=int(end["damage"] - start["damage"]),
                # Die eigentliche Guete: wie viel kostet ein Abschuss? Wildes
                # Ballern faellt hier auf, auch wenn am Ende alles tot ist.
                ammo_per_kill=round(ammo_used / killed, 1) if killed else None,
                monsters_killed=killed,
                humans_lost=start_humans - humans_alive,
                monsters=start_monsters, humans=start_humans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--monsters", type=int, default=7)
    ap.add_argument("--humans", type=int, default=5)
    ap.add_argument("--tics", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=14.0,
                    help="wie weit das Ziel vom Fadenkreuz abweichen darf (px)")
    ap.add_argument("--burst", type=int, default=0,
                    help="Tics, die der Abzug gehalten wird (0 = ganze Aktion)")
    args = ap.parse_args()

    detector = LocateAnythingDetector()
    print(f"\n{'Sicherung':<24}{'Munition':>9}{'Treffer':>9}{'Quote':>8}"
          f"{'Schaden':>9}{'gesperrt':>10}{'Eigenb.':>9}{'Monster':>9}")
    print("-" * 88)
    rows = []
    for mode in ("ohne", "korridor", "korridor+gedaechtnis"):
        r = run(detector, mode, args.steps, args.seed, args.monsters, args.humans,
                args.tics, args.tolerance, args.burst)
        rows.append(dict(guard=mode, **r))
        quote = "-" if r["hit_rate"] is None else f"{r['hit_rate']:.0%}"
        print(f"{mode:<24}{r['ammo']:>9}{r['hits']:>9}{quote:>8}{r['damage']:>9}"
              f"{r['holds']:>10}{r['friendly_fire']:>9}"
              f"{r['monsters_killed']:>6}/{r['monsters']:<3}", flush=True)
    print(f"\n{args.steps} Schritte je Zeile, gleiche Startaufstellung (Seed {args.seed}).")
    print("Eigenbeschuss = Schuss abgegeben, waehrend laut Engine ein Mensch "
          "im Schusskorridor stand.")
    print("Treffer/Quote/Schaden sind dem Agenten zurechenbar; 'Monster erlegt'"
          "\nenthaelt auch, was befreundete Marines und Rangeleien untereinander"
          "\nerledigt haben. Weniger Munition ist besser -"
          "\nSparsamkeit erzwingt Identifikation vor dem Schuss und ist damit selbst"
          "\neine Schutzmassnahme.")
    path = record("eval_policy", vars(args), rows,
                  notes="Wirkung der Sicherungen gegen Eigenbeschuss.")
    print(f"Gesichert: {path}")


if __name__ == "__main__":
    main()
