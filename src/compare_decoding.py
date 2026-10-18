"""Zwei Dekodiermodi nebeneinander - in echter Echtzeit.

In main.py steht das Spiel still, solange das Modell rechnet. Damit sieht jede
Aufzeichnung gleich fluessig aus, egal wie langsam der Detektor ist. Hier
nicht: waehrend der Inferenz laeuft die Welt weiter, mit 35 Tics pro
Sekunde, und der Agent kennt nur die Boxen des Bildes, auf dem er gerade
rechnet. Erst danach handelt er - auf veralteter Grundlage.

Das Video laeuft mit 35 Bildern pro Sekunde, also exakt in Spielzeit: eine
Sekunde im Video ist eine Sekunde, die das Modell zum Rechnen hatte.

  hybrid   Dekodierung aus dem Modellcode (bisheriger Stand)
  guided   vorgegebene Struktur mit Schiedsrichter, siehe detector.py
  *.pt     ein YOLO-Modell, z. B. das von src/distill.py trainierte

Beide Seiten teilen Seed und Modell und laufen nacheinander, damit sie sich
die GPU nicht teilen muessen und die Latenzen sauber bleiben.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                              # noqa: E402
from detector import LocateAnythingDetector                         # noqa: E402
from env import DoomEnv                                             # noqa: E402
from main import _human_in_corridor                                 # noqa: E402
from overlay import AMBER, FG, GREEN, RED                               # noqa: E402
from results import record                                          # noqa: E402
from tracking import pixel_shift                                    # noqa: E402
import overlay                                                      # noqa: E402

TICRATE = 35
HEADER_H = 92
FONT = cv2.FONT_HERSHEY_DUPLEX
DECODINGS = ("hybrid", "guided", "slow")


def make_detector(mode, cache):
    """LocateAnything wird nur einmal geladen und von allen Modi geteilt."""
    if mode in DECODINGS:
        if "la" not in cache:
            cache["la"] = LocateAnythingDetector()
        cache["la"].decoding = mode
        return cache["la"]
    if mode not in cache:
        from yolo_detector import YoloDetector
        cache[mode] = YoloDetector(mode)
    return cache[mode]


def run_side(detector, mode, args, path):
    """Spielt eine Seite fuer `args.seconds` Sekunden Spielzeit.

    Schreibt ein Bild je Tic nach `path` und liefert je Tic die Bilanz, die
    spaeter in die Kopfzeile kommt.
    """
    env = DoomEnv(scenario="arena_mixed", show_window=False, seed=args.seed,
                  monsters=args.monsters, humans=args.humans, invulnerable=True)
    env.new_episode()
    bot = AimBot(env.width, env.height, env.deg_per_pixel(), can_move=env.can_move)
    bot.learning = False
    # Ohne Klassifikator fuer Unbekanntes gehoert die weite Klasse nicht in den
    # Prompt - mit broad=[] wuerde der Parser sie sonst als feindlich werten.
    bot.broad_classes = []
    writer = None
    budget = args.seconds * TICRATE
    tally = dict(decisions=0, shots=0, friendly_fire=0, ms=[])
    # Der Weckschuss in env.wake() kann schon treffen, bevor der Agent
    # handelt - HITCOUNT startet dann nicht bei null. Die Munition fuellt
    # wake() wieder auf, die Treffer nicht. Beides zaehlt ab hier.
    start = env.stats()
    incidents = []   # Tic und Lage bei jedem Schuss mit Mensch im Korridor
    per_tic = []

    def emit(frame, detections, decision, stats, shift, thinking, flash=False):
        nonlocal writer
        canvas = overlay.render(frame, detections, decision, stats,
                                detector.last_latency, bot.classes, bot,
                                muzzle_flash=flash, shift=shift)
        if thinking:
            cv2.putText(canvas, "rechnet ...", (12, env.height - 16), FONT, 0.7,
                        (0, 200, 255), 2, cv2.LINE_AA)
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     TICRATE, (w, h))
        writer.write(canvas)
        per_tic.append(dict(tally, ms=list(tally["ms"]),
                            hits=int(stats["hits"] - start["hits"]),
                            bullets=int(start["ammo"] - stats["ammo"])))

    # Vor der ersten Erkennung gibt es noch nichts zu zeigen.
    detections, decision = [], None
    while len(per_tic) < budget and not env.finished():
        frame = env.frame().copy()
        detections = bot.filter_visible(detector.detect(
            frame, bot.classes, bot.protected_classes, []))
        tally["ms"].append(detector.last_latency * 1000)

        # Die Welt wartet nicht: so viele Tics, wie die Inferenz gedauert hat,
        # laeuft das Spiel ohne Eingabe weiter. Angezeigt wird, was der Agent
        # weiss - die Boxen des Bildes, auf dem er rechnet.
        stale = decision
        for _ in range(round(detector.last_latency * TICRATE)):
            if len(per_tic) >= budget or env.finished():
                break
            env.step(tics=1)
            emit(env.frame(), detections, stale, env.stats(), 0.0, thinking=True)

        # Nachschub: Die befreundeten Marines raeumen die Arena sonst in
        # wenigen Sekunden leer, und der Agent haette nichts mehr zu tun.
        if env.alive_counts()[0] <= 1:
            env.spawn_wave()
        stats = env.stats()
        decision = bot.decide(detections, stats["ammo"], frame=frame,
                              health=stats["health"])
        tally["decisions"] += 1
        if decision.attack:
            tally["shots"] += 1
            if _human_in_corridor(env.ground_truth(), bot, env.width):
                tally["friendly_fire"] += 1
                incidents.append(dict(
                    tic=len(per_tic),
                    seen_humans=sum(d.protected for d in detections),
                    target=getattr(decision.target, "label", None),
                    truth=[(g.kind, g.x1, g.x2, g.y2 - g.y1) for g in env.ground_truth()]))
        for sub, turned in env.step_capture(
                turn_deg=decision.turn_deg, attack=decision.attack,
                forward=decision.forward, backward=decision.backward,
                tics=args.tics):
            if len(per_tic) >= budget:
                break
            emit(sub, detections, decision, env.stats(),
                 pixel_shift(turned, env.deg_per_pixel()), thinking=False,
                 flash=decision.attack)

    writer.release()
    env.close()
    final = dict(per_tic[-1], mode=mode,
                 ms_median=float(np.median(tally["ms"])) if tally["ms"] else 0.0)
    final.pop("ms")
    final["incidents"] = incidents
    return per_tic, final


def header(width, title, tone, tic, info, seconds):
    img = np.full((HEADER_H, width, 3), (16, 16, 20), np.uint8)
    ms = np.median(info["ms"]) if info["ms"] else 0
    cv2.putText(img, title, (16, 34), FONT, 0.95, tone, 2, cv2.LINE_AA)
    cv2.putText(img, f"{ms:.0f} ms je Bild", (width - 230, 34), FONT, 0.8, FG, 1, cv2.LINE_AA)
    rate = f"{info['hits'] / info['bullets']:.0%}" if info["bullets"] else "-"
    line = (f"Entscheidungen {info['decisions']}  Kugeln {info['bullets']}  "
            f"Treffer {info['hits']} ({rate})  Mensch im Korridor {info['friendly_fire']}")
    cv2.putText(img, line, (16, 74), FONT, 0.62, FG, 1, cv2.LINE_AA)
    cv2.putText(img, f"{tic / TICRATE:4.1f} / {seconds} s", (width - 170, 74), FONT, 0.62,
                FG, 1, cv2.LINE_AA)
    return img


def compose(paths, logs, titles, out, seconds):
    caps = [cv2.VideoCapture(p) for p in paths]
    ok_frames = [None] * len(caps)
    writer = None
    total = max(len(log) for log in logs)
    for tic in range(total):
        tiles = []
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok:
                ok_frames[i] = frame
            frame = ok_frames[i]
            info = logs[i][min(tic, len(logs[i]) - 1)]
            # Links vorher, rechts nachher - dazwischen alles in Gelb.
            tone = RED if i == 0 else GREEN if i == len(caps) - 1 else AMBER
            tiles.append(np.vstack([header(frame.shape[1], titles[i], tone, tic, info, seconds),
                                    frame]))
        gap = np.full((tiles[0].shape[0], 8, 3), (16, 16, 20), np.uint8)
        canvas = np.hstack([part for tile in tiles for part in (tile, gap)][:-1])
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(out + ".raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                     TICRATE, (w, h))
        writer.write(canvas)
    writer.release()
    for cap in caps:
        cap.release()
    # H.264/yuv420p, damit es ueberall abspielt - mp4v spielen viele Player nicht.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".raw.mp4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    os.remove(out + ".raw.mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["hybrid", "guided"],
                    help="hybrid, guided, slow oder Pfad zu YOLO-Gewichten")
    ap.add_argument("--titles", nargs="+",
                    default=["vorher: hybrid (Modellcode)", "nachher: guided"])
    ap.add_argument("--seconds", type=int, default=30, help="Spielzeit je Seite")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--monsters", type=int, default=8)
    ap.add_argument("--humans", type=int, default=3)
    ap.add_argument("--tics", type=int, default=4, help="Tics je Aktion")
    ap.add_argument("--out", default="out/vergleich_dekodierung.mp4")
    args = ap.parse_args()
    if len(args.titles) != len(args.modes):
        ap.error("--titles braucht genauso viele Eintraege wie --modes")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cache = {}
    # Die ersten Aufrufe sind deutlich langsamer (CUDA-Kernel, Caches) und
    # wuerden die Seite benachteiligen, die zuerst laeuft.
    warm = DoomEnv(scenario="arena_mixed", show_window=False, seed=args.seed)
    warm.new_episode()
    for mode in args.modes:
        detector = make_detector(mode, cache)
        for _ in range(2):
            detector.detect(warm.frame(), broad=[])
    warm.close()
    tmp = tempfile.mkdtemp(prefix="compare_decoding_")
    paths, logs, rows = [], [], []
    for index, mode in enumerate(args.modes):
        path = os.path.join(tmp, f"seite{index}.mp4")
        log, final = run_side(make_detector(mode, cache), mode, args, path)
        print(f"{mode:>7}: {final['decisions']} Entscheidungen in {args.seconds} s, "
              f"{final['ms_median']:.0f} ms je Bild, {final['shots']} Feuerbefehle, "
              f"{final['bullets']} Kugeln, {final['hits']} Treffer, "
              f"Mensch im Korridor {final['friendly_fire']}",
              flush=True)
        paths.append(path)
        logs.append(log)
        rows.append(final)
    compose(paths, logs, args.titles, args.out, args.seconds)
    print(f"Video: {args.out}")
    path = record("compare_decoding", vars(args), rows,
                  notes="Echtzeit: waehrend der Inferenz laeuft das Spiel weiter.")
    print(f"Gesichert: {path}")


if __name__ == "__main__":
    main()
