"""Setzt aus den Einzelaufnahmen einen zusammenhaengenden Film.

Zwischen den Abschnitten steht jeweils eine Karte, die sagt, was als
Naechstes zu sehen ist - und was daran der Punkt war. Hochkant gedrehte
Aufnahmen stehen rechts, die Erklaerung nutzt den Platz daneben.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W, H, FPS = 1920, 1080, 30
BG = (16, 16, 20)
CARD_BG = (26, 26, 32)
FG = (232, 232, 238)
DIM = (160, 160, 176)
FAINT = (116, 116, 138)
GREEN = (90, 220, 120)
RED = (70, 70, 245)
BLUE = (255, 180, 60)
VIOLET = (230, 120, 190)
FONT = cv2.FONT_HERSHEY_DUPLEX

# Aus der Hochkant-Aufnahme wird nur das Spielbild samt Zustandszeile
# uebernommen. Die ganze Hochkantflaeche auf 608 Pixel Breite zu quetschen
# machte das eingebaute Panel unlesbar und doppelte die Erklaerung daneben.
CROP_TOP, CROP_BOTTOM = 224, 1288         # Spielbild, Zustand und Begruendung
VID_W, VID_H = 1046, 1022
VID_X, VID_Y = W - VID_W - 60, (H - VID_H) // 2


def text(canvas, s, x, y, colour, scale, thickness=1):
    cv2.putText(canvas, s, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def card(kapitel, titel, zeilen, kicker=None, tone=VIOLET):
    """Zwischenkarte: was kommt jetzt, und warum."""
    canvas = np.full((H, W, 3), BG, np.uint8)
    text(canvas, kapitel, 120, 250, tone, 0.85, 2)
    cv2.line(canvas, (120, 285), (120 + 260, 285), tone, 3)
    y = 400
    for i, part in enumerate(titel):
        text(canvas, part, 120, y, FG, 2.6 if i == 0 else 2.0, 4 if i == 0 else 3)
        y += 96 if i == 0 else 82
    y += 40
    for zeile in zeilen:
        text(canvas, zeile, 120, y, DIM, 0.95, 1)
        y += 52
    if kicker:
        text(canvas, kicker, 120, H - 150, tone, 1.5, 3)
    return canvas


def with_panel(frame, kapitel, titel, punkte, tone):
    """Ausschnitt der Hochkant-Aufnahme rechts, Erklaerung links."""
    canvas = np.full((H, W, 3), BG, np.uint8)
    clip = cv2.resize(frame[CROP_TOP:CROP_BOTTOM], (VID_W, VID_H),
                      interpolation=cv2.INTER_AREA)
    canvas[VID_Y:VID_Y + VID_H, VID_X:VID_X + VID_W] = clip
    cv2.rectangle(canvas, (VID_X - 2, VID_Y - 2),
                  (VID_X + VID_W + 1, VID_Y + VID_H + 1), (52, 52, 64), 2)

    text(canvas, kapitel, 70, 210, tone, 0.68, 2)
    cv2.line(canvas, (70, 238), (70 + 190, 238), tone, 3)
    y = 330
    for i, part in enumerate(titel):
        text(canvas, part, 70, y, FG, 1.35, 3)
        y += 62
    y += 46
    for colour, punkt in punkte:
        zeilen = wrap(punkt, 30)
        cv2.rectangle(canvas, (70, y - 19), (70 + 24, y + 1), colour, -1)
        for j, chunk in enumerate(zeilen):
            text(canvas, chunk, 108, y + j * 36, FG if j == 0 else DIM, 0.72, 1)
        y += 40 + 36 * (len(zeilen) - 1)
    return canvas


def banner(frame, kapitel, titel, tone):
    """Querformat: nur ein schmaler Streifen oben."""
    canvas = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    strip = canvas[:78].astype(np.float32) * 0.25
    canvas[:78] = strip.astype(np.uint8)
    text(canvas, kapitel, 30, 34, tone, 0.62, 2)
    text(canvas, titel, 30, 66, FG, 0.86, 2)
    return canvas


def wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def read_clip(path, start_s, length_s):
    """Frames aus einer Quelle holen und auf die Filmbildrate bringen.

    Nicht alle Aufnahmen laufen mit 30 Bildern je Sekunde - die aelteste
    wurde mit 6 aufgezeichnet. Ohne Umrechnung wuerden daraus statt zwanzig
    Sekunden nur vier.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    quelle = []
    while len(quelle) < int(length_s * fps):
        ok, frame = cap.read()
        if not ok:
            break
        quelle.append(frame)
    cap.release()
    if not quelle:
        return []
    # Jede Quellsekunde auf FPS Bilder dehnen, indem gleichmaessig
    # nachgeschlagen wird - kein Nachbearbeiten, nur zeitliches Strecken.
    want = int(length_s * FPS)
    return [quelle[min(len(quelle) - 1, int(i * len(quelle) / want))]
            for i in range(want)]


# Der Ablauf des Films. Chronologisch, wie das Projekt entstanden ist.
DREHBUCH = [
    dict(art="titel",
         titel=["THE STORY OF DOOM"],
         zeilen=["Ein Vision-Language-Modell lernt, worauf es schiessen darf.",
                 "Sechs Stufen, jede an einer Zahl entschieden.",
                 "",
                 "NVIDIA LocateAnything-3B  +  ViZDoom"],
         dauer=5.0, tone=FG),

    dict(art="karte", kapitel="ERSTE STUFE", tone=GREEN,
         titel=["Sehen und zielen"],
         zeilen=["Das Modell bekommt das Spielbild und eine Textbeschreibung.",
                 "Zurueck kommen Boxen. Das Zielen danach ist reine Geometrie.",
                 "",
                 "Das Spiel steht still, solange gerechnet wird - die Demo",
                 "haengt nicht an der Rechenzeit."],
         kicker="97 % Treffer auf Doom-Sprites", dauer=6.5),
    dict(art="hoch", quelle="out/raw_vertical.mp4", start=22, dauer=20,
         kapitel="ERSTE UND ZWEITE STUFE", tone=GREEN,
         titel=["Monster und", "Menschen"],
         punkte=[(GREEN, "Monster erkannt"),
                 (RED, "aktuelles Ziel"),
                 (BLUE, "Mensch - nicht schiessen")]),

    dict(art="karte", kapitel="ZWEITE STUFE", tone=BLUE,
         titel=["Die Wortwahl", "entscheidet"],
         zeilen=["Mit Kategorienamen hielt das Modell 56 % der Marines fuer",
                 "Monster. Mit Beschreibungen dessen, was auf dem Schirm steht,",
                 "waren es 6 %. Gleiche Bilder, gleiches Modell, andere Worte.",
                 "",
                 "Dazu zwei Sicherungen: ein gesperrter Schusskorridor und ein",
                 "Gedaechtnis, das Menschen ueber Aussetzer hinweg festhaelt."],
         kicker="19 Eigenbeschuesse  ->  0", dauer=8.0),

    dict(art="karte", kapitel="DRITTE STUFE", tone=VIOLET,
         titel=["Was tun mit etwas,", "das man nicht kennt?"],
         zeilen=["Eine absichtlich weite Prompt-Klasse faengt auch das ein,",
                 "wofuer es keine Beschreibung gibt. Der Agent haelt an,",
                 "beobachtet - und stellt so lange das Feuer ein."],
         kicker="Violett = unbekannt, in Beobachtung", dauer=7.0),
    dict(art="hoch", quelle="out/raw_unknown_v.mp4", start=8, dauer=22,
         kapitel="DRITTE STUFE", tone=VIOLET,
         titel=["Das Unbekannte"],
         punkte=[(VIOLET, "keine Beschreibung dafuer im Prompt"),
                 (FG, "Hypothesen ueber mehrere Bilder abgestimmt"),
                 (GREEN, "danach eingeordnet und gemerkt")]),

    dict(art="karte", kapitel="VIERTE STUFE", tone=RED,
         titel=["Sparsamkeit ist", "Schutz"],
         zeilen=["Wer jeden Schuss rechtfertigen muss, feuert nicht ins",
                 "Ungewisse. Munitionsdisziplin und Personenschutz sind",
                 "dieselbe Sache aus zwei Blickwinkeln.",
                 "",
                 "Links ohne Sicherung, rechts mit - gleicher Seed,",
                 "gleiche Arena, beide sehen dasselbe."],
         kicker="68 Schuss  gegen  18", dauer=8.0),
    dict(art="quer", quelle="out/raw_compare_yt.mp4", start=0, dauer=21,
         kapitel="VIERTE STUFE", titel="Munitionsdisziplin", tone=RED),

    dict(art="karte", kapitel="FUENFTE STUFE", tone=BLUE,
         titel=["Drei Regelwerke"],
         zeilen=["Wild schiesst auf alles. Streng haelt an, bis Unbekanntes",
                 "eingeordnet ist - und wird darueber handlungsunfaehig.",
                 "Gewichtet beobachtet nebenher.",
                 "",
                 "Der Mittelweg zahlt drauf: mehr Schuss, kein Gewinn."],
         kicker="Vorsicht hat einen Preis", dauer=8.0),
    dict(art="quer", quelle="out/raw_drei_yt.mp4", start=0, dauer=21,
         kapitel="FUENFTE STUFE", titel="Drei Regelwerke im Vergleich", tone=BLUE),

    dict(art="karte", kapitel="SECHSTE STUFE", tone=VIOLET,
         titel=["Als das Aussehen", "aufhoerte zu genuegen"],
         zeilen=["Drei Marines, pixelgleich: Zivilist, Verbuendeter, Feind.",
                 "Zwei Monsterarten, gleich bedrohlich - nur eine greift an.",
                 "",
                 "Der visuelle Klassifikator lag auf allen friedlich",
                 "gesetzten Wesen zu 0 % richtig. Das ist Zufallsniveau,",
                 "und es ist kein Fehler des Modells: die Frage ist",
                 "visuell grundsaetzlich nicht beantwortbar."],
         kicker="0 von 6 richtig", dauer=9.0),

    dict(art="karte", kapitel="DIE ANTWORT", tone=GREEN,
         titel=["Was jemand tut,", "statt wie er aussieht"],
         zeilen=["Faellt die eigene Gesundheit, waehrend ein Wesen zu sehen",
                 "ist, dann ist es gefaehrlich. Zehn von zehn richtig.",
                 "",
                 "Der Preis: der erste Schaden kommt im Mittel erst nach",
                 "sieben Schritten. Man weiss es, wenn es passiert ist."],
         kicker="Aussehen 0 %   Verhalten 100 %", dauer=8.0),
    dict(art="hoch", quelle="out/raw_verhalten_v.mp4", start=10, dauer=22,
         kapitel="SECHSTE STUFE", tone=GREEN,
         titel=["Verhalten schlaegt", "Aussehen"],
         punkte=[(VIOLET, "(Aussehen) ist eine Vermutung"),
                 (GREEN, "(Verhalten) ist endgueltig und darf widerlegen"),
                 (BLUE, "Mensch bleibt Mensch, egal wie er aussieht")]),

    dict(art="titel", tone=VIOLET,
         titel=["Was bleibt"],
         zeilen=["Wahrnehmung war das Billige. Die Technik um ihre Unschaerfe",
                 "herum war die Arbeit.",
                 "",
                 "Alle Messungen, auch die gescheiterten:",
                 "github.com/ThomasSchmiegelt/llm-plays-doom"],
         dauer=7.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/raw_story.mp4")
    ap.add_argument("--final", default="out/the_story_of_doom.mp4")
    ap.add_argument("--crf", type=int, default=30)
    args = ap.parse_args()

    os.makedirs("out", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gesamt = 0

    for teil in DREHBUCH:
        if teil["art"] in ("titel", "karte"):
            frame = card(teil.get("kapitel", ""), teil["titel"], teil["zeilen"],
                         teil.get("kicker"), teil.get("tone", VIOLET))
            anzahl = int(teil["dauer"] * FPS)
            for _ in range(anzahl):
                writer.write(frame)
            gesamt += anzahl
            print(f"  Karte    {teil['titel'][0][:34]:<36}{teil['dauer']:>5.1f}s", flush=True)
            continue

        clip = read_clip(teil["quelle"], teil["start"], teil["dauer"])
        for frame in clip:
            if teil["art"] == "hoch":
                writer.write(with_panel(frame, teil["kapitel"], teil["titel"],
                                        teil["punkte"], teil["tone"]))
            else:
                writer.write(banner(frame, teil["kapitel"], teil["titel"], teil["tone"]))
        gesamt += len(clip)
        print(f"  Aufnahme {os.path.basename(teil['quelle']):<36}"
              f"{len(clip)/FPS:>5.1f}s", flush=True)

    writer.release()
    print(f"\n{gesamt} Bilder = {gesamt/FPS:.0f} s")

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", args.out,
           "-f", "lavfi", "-t", str(gesamt / FPS), "-i", "anullsrc=r=44100:cl=stereo",
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-profile:v", "high",
           "-c:a", "aac", "-b:a", "128k", "-shortest",
           "-movflags", "+faststart", args.final]
    subprocess.run(cmd, check=True)
    size = os.path.getsize(args.final) / 1048576
    print(f"Fertig: {args.final}  ({size:.0f} MB)")


if __name__ == "__main__":
    main()
