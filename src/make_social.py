"""Baut aus der Hochkant-Aufzeichnung eine Datei fuer Shorts und Reels.

Beide Plattformen wollen 1080x1920, H.264 in yuv420p und eine Tonspur - auch
eine stumme, weil manche Uploads sonst zicken. Die Laenge wird auf das Ziel
gestreckt oder gestaucht, indem die Eingangsbildrate umgerechnet wird; die
Ausgabe laeuft mit 30 fps.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vertical  # noqa: E402

INTRO_SECONDS = 2.0
OUTRO_SECONDS = 3.0
OUT_FPS = 30


def cap_holds(source: str, target: str, max_hold: int, threshold: float = 0.6) -> int:
    """Begrenzt, wie lange dasselbe Bild stehen bleibt.

    Die Aufzeichnung schreibt Schluesselmomente mehrfach, damit sie lesbar
    sind. Zu viele Wiederholungen kippen aber ins Ruckeln: die Haelfte der
    Bilder war identisch, und dazwischen sprang es. Hier bleibt ein Standbild
    hoechstens `max_hold` Bilder lang stehen, der Rest faellt weg.
    """
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    previous, run, kept = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (96, 160)), cv2.COLOR_BGR2GRAY)
        same = (previous is not None and
                float(np.abs(small.astype(np.int16) - previous).mean()) < threshold)
        run = run + 1 if same else 0
        previous = small.astype(np.int16)
        if run >= max_hold:
            continue
        writer.write(frame)
        kept += 1
    cap.release()
    writer.release()
    return kept


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise SystemExit(f"Keine Frames in {path}")
    return count


def build_cards(work_dir: str, results: dict) -> tuple[str, str]:
    intro = vertical.title_card(
        results.get("intro") or
        ["LLM SPIELT DOOM", "Monster: schiessen", "Menschen: verschonen"],
        results.get("intro_sub") or
        "Ein Textprompt unterscheidet beide - NVIDIA LocateAnything-3B")
    # Bewusst nicht "N Menschen verschont": Marines sterben auch durch
    # Monster, die Zahl waere dem Agenten faelschlich zugerechnet. Die
    # Feuersperren sind das, was der Agent tatsaechlich geleistet hat.
    outro = vertical.title_card(
        results.get("outro") or
        [f"{results['monsters']} Monster erlegt",
         f"{results['holds']}x Feuer gesperrt",
         "Mensch in der Schusslinie"],
        results.get("outro_sub") or
        "Erkennung: LocateAnything-3B  |  Spiel: ViZDoom")
    intro_path = os.path.join(work_dir, "_intro.png")
    outro_path = os.path.join(work_dir, "_outro.png")
    cv2.imwrite(intro_path, intro)
    cv2.imwrite(outro_path, outro)
    return intro_path, outro_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="out/raw_vertical.mp4")
    ap.add_argument("--out", default="out/doom_short_1080x1920.mp4")
    ap.add_argument("--duration", type=float, default=55.0,
                    help="Gesamtlaenge in Sekunden (Shorts: <= 60, Reels: <= 90)")
    ap.add_argument("--monsters", type=int, default=0, help="Ergebnis fuer die Abspannkarte")
    ap.add_argument("--holds", type=int, default=0)
    ap.add_argument("--intro", nargs="+", default=[], help="Zeilen der Titelkarte")
    ap.add_argument("--intro-sub", default="")
    ap.add_argument("--outro", nargs="+", default=[], help="Zeilen der Abspannkarte")
    ap.add_argument("--outro-sub", default="")
    ap.add_argument("--crf", type=int, default=20,
                    help="Qualitaetsstufe von libx264; hoeher = kleinere Datei")
    ap.add_argument("--max-hold", type=int, default=0,
                    help="wie viele Bilder ein Standbild hoechstens stehen bleibt "
                         "(0 = unveraendert lassen)")
    ap.add_argument("--nvenc", action="store_true", help="mit der GPU kodieren")
    args = ap.parse_args()

    source = args.source
    if args.max_hold:
        source = os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".",
                              "_capped.mp4")
        kept = cap_holds(args.source, source, args.max_hold)
        print(f"Standzeiten begrenzt: {frame_count(args.source)} -> {kept} Bilder")
    frames = frame_count(source)
    body_seconds = args.duration - INTRO_SECONDS - OUTRO_SECONDS
    if body_seconds <= 1:
        raise SystemExit("Zieldauer zu kurz fuer Vor- und Abspann")
    # Eingangsbildrate so setzen, dass der Hauptteil genau die Restzeit fuellt.
    in_fps = frames / body_seconds

    work = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(work, exist_ok=True)
    intro, outro = build_cards(work, dict(
        monsters=args.monsters, holds=args.holds,
        intro=args.intro, intro_sub=args.intro_sub,
        outro=args.outro, outro_sub=args.outro_sub))

    codec = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(args.crf)] if args.nvenc
             else ["-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf)])
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(INTRO_SECONDS), "-i", intro,
        "-r", f"{in_fps:.6f}", "-i", source,
        "-loop", "1", "-t", str(OUTRO_SECONDS), "-i", outro,
        # Stille Tonspur: manche Uploads verweigern Video ohne Audio.
        "-f", "lavfi", "-t", str(args.duration), "-i", "anullsrc=r=44100:cl=stereo",
        "-filter_complex",
        f"[0:v]fps={OUT_FPS},setsar=1[a];"
        f"[1:v]fps={OUT_FPS},setsar=1[b];"
        f"[2:v]fps={OUT_FPS},setsar=1[c];"
        f"[a][b][c]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a",
        *codec,
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        "-movflags", "+faststart",
        args.out,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.remove(intro)
    os.remove(outro)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt",
         "-show_entries", "format=duration,size", "-of", "default=nw=1", args.out],
        capture_output=True, text=True)
    print(f"\n{args.out}\n{probe.stdout.strip()}")


if __name__ == "__main__":
    main()
