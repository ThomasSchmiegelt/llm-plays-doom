"""LocateAnything als Lehrer: ein kleines YOLO lernt, was das grosse sieht.

YOLO-World mit denselben Textprompts ist zwar schnell (16 ms), erkennt
Doom-Sprites aber kaum - es kennt Fotos, keine Pixeldaemonen. LocateAnything
erkennt sie, braucht aber 0,5 s. Also beschriftet LocateAnything Bilder, und
ein normales YOLO lernt daraus.

  label   Bilder sammeln und von LocateAnything beschriften lassen
  train   YOLO auf diesen Beschriftungen trainieren

Wichtig fuer die Ehrlichkeit der Messung: Die Engine-Wahrheit wird hier nicht
angefasst. Sie bleibt allein src/eval_detector.py vorbehalten, und dessen
Seed (11) sowie der Seed der Vergleichsvideos (21) kommen im Training nicht
vor. Der Schueler kann also hoechstens so gut werden wie sein Lehrer - samt
dessen Fehlern.
"""
import argparse
import json
import os
import random
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import HOSTILE_CLASSES, PROTECTED_CLASSES             # noqa: E402
from env import DoomEnv                                             # noqa: E402

CLASSES = HOSTILE_CLASSES + PROTECTED_CLASSES
RESERVED_SEEDS = {11, 21}      # eval_detector.py und compare_decoding.py


def label(args):
    from detector import LocateAnythingDetector

    root = os.path.abspath(args.data)
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
    detector = LocateAnythingDetector()
    rng = random.Random(args.seed)
    seeds = [s for s in range(100, 100 + 4 * args.episodes) if s not in RESERVED_SEEDS]
    per_episode = -(-args.frames // args.episodes)
    count, boxes, start = 0, 0, time.perf_counter()

    for episode, seed in enumerate(seeds[:args.episodes]):
        # Wechselnde Besetzung, damit nicht jedes Bild gleich aussieht.
        env = DoomEnv(scenario="arena_mixed", show_window=False, seed=seed,
                      monsters=rng.randint(4, 10), humans=rng.randint(2, 6),
                      invulnerable=True)
        env.new_episode()
        # Validierung nach Episoden trennen, nicht nach Bildern: aufeinander-
        # folgende Bilder sind sich zu aehnlich und wuerden sie schoenfaerben.
        split = "val" if episode % 10 == 9 else "train"
        for _ in range(per_episode):
            if count >= args.frames or env.finished():
                break
            if env.alive_counts()[0] <= 1:
                env.spawn_wave()
            env.step(turn_deg=rng.uniform(-45, 45), attack=rng.random() < 0.2,
                     tics=rng.randint(2, 12))
            frame = env.frame()
            if frame is None:
                break
            dets = detector.detect(frame, CLASSES, PROTECTED_CLASSES, broad=[])
            name = f"s{seed}_{count:05d}"
            cv2.imwrite(os.path.join(root, "images", split, name + ".jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = frame.shape[:2]
            with open(os.path.join(root, "labels", split, name + ".txt"), "w") as fh:
                for d in dets:
                    fh.write(f"{CLASSES.index(d.label)} {d.cx / w:.6f} {d.cy / h:.6f} "
                             f"{(d.x2 - d.x1) / w:.6f} {(d.y2 - d.y1) / h:.6f}\n")
            count += 1
            boxes += len(dets)
            if count % 100 == 0:
                rate = count / (time.perf_counter() - start)
                print(f"  {count}/{args.frames} Bilder, {boxes} Boxen, "
                      f"{rate:.1f} Bilder/s", flush=True)
        env.close()

    with open(os.path.join(root, "data.yaml"), "w") as fh:
        fh.write(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(CLASSES):
            fh.write(f"  {i}: {c}\n")
    with open(os.path.join(root, "teacher.json"), "w") as fh:
        json.dump(dict(frames=count, boxes=boxes, seeds=seeds[:args.episodes],
                       teacher="LocateAnything-3B guided",
                       seconds=round(time.perf_counter() - start)), fh, indent=1)
    print(f"Fertig: {count} Bilder, {boxes} Boxen -> {root}")


def train(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=os.path.join(os.path.abspath(args.data), "data.yaml"),
                epochs=args.epochs, imgsz=640, batch=32, project=os.path.abspath(args.project),
                name=args.name, exist_ok=True, plots=True, verbose=False,
                # Doom spiegelt sich nicht schadlos: Sprites sind asymmetrisch,
                # aber das stoert die Klassen nicht. Farbverschiebung dagegen
                # schon - "pink" und "blau" sind Teil der Klassennamen.
                hsv_h=0.0, hsv_s=0.3, hsv_v=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--data", default="data/distill")
    lab.add_argument("--frames", type=int, default=2000)
    lab.add_argument("--episodes", type=int, default=20)
    lab.add_argument("--seed", type=int, default=0)
    tr = sub.add_parser("train")
    tr.add_argument("--data", default="data/distill")
    tr.add_argument("--model", default="yolo11s.pt")
    tr.add_argument("--epochs", type=int, default=80)
    tr.add_argument("--project", default="models/runs")
    tr.add_argument("--name", default="distill")
    args = ap.parse_args()
    {"label": label, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
