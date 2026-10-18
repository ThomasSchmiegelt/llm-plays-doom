"""LLM spielt Doom - erkennt Monster, verschont Menschen.

Ablauf pro Schritt:
  1. Frame aus ViZDoom holen (das Spiel steht dabei still)
  2. LocateAnything-3B nach Monstern UND Menschen fragen (ein Prompt)
  3. Nur auf feindlich gelabelte Objekte zielen; Feuer sperren, sobald ein
     als Mensch erkanntes Objekt im Schusskorridor steht
  4. Overlay zeichnen, Spiel einige Tics weiterlaufen lassen
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import AimBot                                                  # noqa: E402
from classifier import (ABSTRACT_HYPOTHESES, DESCRIPTIVE_HYPOTHESES,    # noqa: E402
                        UnknownClassifier)
from detector import (BROAD_CLASSES, DEFAULT_CLASSES, HOSTILE_CLASSES,  # noqa: E402
                      PROTECTED_CLASSES, Detection)
from env import SCENARIOS, DoomEnv                                      # noqa: E402
from tracking import pixel_shift                                        # noqa: E402
import overlay                                                          # noqa: E402
import vertical                                                         # noqa: E402

WINDOW = "LLM plays Doom - Agent Vision"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="arena_mixed", choices=list(SCENARIOS))
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Textklassen fuer den Detektor")
    ap.add_argument("--protected", nargs="+", default=PROTECTED_CLASSES,
                    help="Teilmenge von --classes, auf die nicht geschossen wird")
    ap.add_argument("--monsters", type=int, default=7, help="Monster je Welle")
    ap.add_argument("--humans", type=int, default=5, help="Menschen je Welle")
    ap.add_argument("--waves", type=int, default=1,
                    help="Nachschub: neue Welle, sobald fast alle Monster tot sind")
    ap.add_argument("--unknowns", type=int, default=0,
                    help="wie viele unbekannte Wesen im Lauf auftauchen (zufaellig gewaehlt)")
    ap.add_argument("--unknown-every", type=int, default=60,
                    help="Schritte zwischen zwei Auftritten")
    ap.add_argument("--hypotheses", default="descriptive",
                    choices=["descriptive", "abstract"],
                    help="Hypothesensatz fuer die Einordnung")
    ap.add_argument("--probe-mode", default="crop", choices=["crop", "frame"],
                    help="Abfrage auf dem Bildausschnitt oder dem ganzen Bild")
    ap.add_argument("--window", type=int, default=6,
                    help="wie viele Proben eine Einordnung hoechstens braucht")
    ap.add_argument("--no-learn", action="store_true",
                    help="eingeordnete Wesen nicht in den Prompt aufnehmen - "
                         "so wird jeder Auftritt einzeln beobachtet")
    ap.add_argument("--tics", type=int, default=4,
                    help="Spieltics pro Agentenschritt (mehr = fluessiger, aber traeger)")
    ap.add_argument("--steps", type=int, default=0, help="Schritte je Episode (0 = unbegrenzt)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--god", action="store_true",
                    help="Agent unverwundbar - fuer lange Aufnahmen ueber mehrere Wellen")
    ap.add_argument("--show-gt", action="store_true", help="Engine-Ground-Truth einblenden")
    ap.add_argument("--game-window", action="store_true",
                    help="zusaetzlich das rohe ViZDoom-Fenster anzeigen")
    ap.add_argument("--no-display", action="store_true",
                    help="kein Fenster oeffnen (nur aufzeichnen)")
    ap.add_argument("--record", default="", help="Pfad fuer eine mp4-Aufzeichnung")
    ap.add_argument("--record-vertical", default="",
                    help="zusaetzliche Aufzeichnung in 1080x1920 fuer Shorts/Reels")
    ap.add_argument("--fps", type=float, default=6.0, help="Bildrate der Aufzeichnung")
    ap.add_argument("--smooth", action="store_true",
                    help="Bild nach jedem Tic aufzeichnen und die Drehung darauf "
                         "verteilen - vervierfacht die Bilder und macht den Schwenk "
                         "fluessig statt sprunghaft")
    ap.add_argument("--emphasis", type=int, default=0,
                    help="Schluesselmomente (Beobachtung, Einordnung) so oft zusaetzlich "
                         "in die Aufzeichnung schreiben - sonst huschen sie vorbei")
    ap.add_argument("--decoding", default="guided", choices=["guided", "hybrid", "slow"],
                    help="Dekodierung von LocateAnything: guided (Default, schnell), "
                         "hybrid (Modellcode), slow (Einzeltoken)")
    ap.add_argument("--detector", default="locateanything",
                    help="locateanything, oder Pfad zu YOLO-Gewichten - etwa das "
                         "von src/distill.py trainierte Modell (Echtzeit)")
    ap.add_argument("--fake-detector", action="store_true",
                    help="ohne Modell laufen, Boxen aus der Engine (nur zum Testen)")
    return ap.parse_args()


class _EngineDetector:
    """Ersatzdetektor aus den Engine-Labels - zum Testen der Pipeline ohne GPU."""

    def __init__(self, env):
        self.env = env
        self.last_latency = 0.0
        # Wird von main() gesetzt, sobald der Klassifikator steht: Beschreibung
        # -> Wahrheit. Damit kann der Trockenlauf auch die Hypothesenabfrage
        # bedienen und die Zustandsfolge wird vollstaendig durchlaufen.
        self.hypotheses: dict[str, str] = {}

    # Die Engine-Wahrheit auf dieselben drei Sorten abbilden, die der echte
    # Detektor liefert - Unbekanntes kommt hier ohne Urteil heraus.
    _KINDS = {"monster": ("monster", "hostile"),
              "human": ("human", "protected"),
              "unknown_dangerous": ("any creature or figure", "unknown"),
              "unknown_harmless": ("any creature or figure", "unknown")}

    def detect(self, frame, classes, protected=None, broad=None):
        start = time.perf_counter()
        if self.hypotheses and set(classes) <= set(self.hypotheses):
            out = self._probe(classes)
            self.last_latency = time.perf_counter() - start
            return out
        out = []
        for g in self.env.ground_truth():
            label, kind = self._KINDS.get(g.kind, ("monster", "hostile"))
            if kind == "unknown" and not broad:
                continue  # breite Klasse abgeschaltet
            out.append(Detection(g.x1, g.y1, g.x2, g.y2, label, kind))
        self.last_latency = time.perf_counter() - start
        return out

    def _probe(self, classes):
        """Antwortet auf eine Hypothesenabfrage mit der Wahrheit der Engine."""
        out = []
        for g in self.env.ground_truth():
            if not g.kind.startswith("unknown_"):
                continue
            truth = g.kind.split("_", 1)[1]
            fitting = [c for c in classes if self.hypotheses.get(c) == truth]
            if fitting:
                out.append(Detection(g.x1, g.y1, g.x2, g.y2, fitting[0], "hostile"))
        return out


def main():
    args = parse_args()

    env = DoomEnv(scenario=args.scenario, show_window=args.game_window, seed=args.seed,
                  monsters=args.monsters, humans=args.humans, invulnerable=args.god)
    print(f"Szenario: {args.scenario}  {env.width}x{env.height}  FOV {env.fov:.0f}°")

    if args.fake_detector:
        detector = _EngineDetector(env)
        print("Detektor: Engine-Labels (Testmodus)")
    elif args.detector != "locateanything":
        from yolo_detector import YoloDetector
        detector = YoloDetector(args.detector)
        print(f"Detektor: YOLO ({args.detector})")
    else:
        from detector import LocateAnythingDetector
        print("Lade LocateAnything-3B ...", flush=True)
        t0 = time.perf_counter()
        detector = LocateAnythingDetector(decoding=args.decoding)
        print(f"Modell bereit ({time.perf_counter() - t0:.1f}s)")
    print(f"Feindlich:  {', '.join(c for c in args.classes if c not in args.protected)}")
    print(f"Geschuetzt: {', '.join(args.protected)}", flush=True)

    bot = AimBot(env.width, env.height, env.deg_per_pixel(), can_move=env.can_move)
    bot.hostile_classes = [c for c in args.classes if c not in args.protected]
    bot.protected_classes = list(args.protected)
    bot.learning = not args.no_learn
    hypotheses = (ABSTRACT_HYPOTHESES if args.hypotheses == "abstract"
                  else DESCRIPTIVE_HYPOTHESES)
    classifier = UnknownClassifier(detector, hypotheses, mode=args.probe_mode,
                                   window=args.window) if args.unknowns else None
    if classifier and args.fake_detector:
        detector.hypotheses = dict(hypotheses)
    if classifier:
        print(f"Unbekanntes: {args.unknowns} Auftritte, alle {args.unknown_every} Schritte, "
              f"{args.hypotheses}e Hypothesen ({len(hypotheses)}), Fenster {args.window}")
    writer = _make_writer(args.record, env.width + overlay.PANEL_W, env.height, args.fps)
    writer_v = _make_writer(args.record_vertical, vertical.W, vertical.H, args.fps)

    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    show_gt, paused, quit_all = args.show_gt, False, False

    try:
        for episode in range(args.episodes):
            env.new_episode()
            start_monsters, start_humans = env.alive_counts()
            monsters_spawned, humans_spawned = start_monsters, start_humans
            waves_left = args.waves - 1
            friendly_fire = 0
            steps, flash = 0, False
            unknowns_left = args.unknowns
            next_unknown = args.unknown_every

            while not env.finished():
                frame = env.frame()
                if frame is None:
                    break

                # Der Prompt kommt vom Agenten, nicht aus den Argumenten -
                # er waechst, sobald etwas eingeordnet wurde.
                detections = bot.filter_visible(detector.detect(
                    frame, bot.classes, bot.protected_classes,
                    bot.broad_classes if classifier else []))
                truth = env.ground_truth()
                monsters_alive, humans_alive = env.alive_counts()
                stats = env.stats()
                stats["monsters_killed"] = monsters_spawned - monsters_alive
                stats["humans_killed"] = humans_spawned - humans_alive
                stats["friendly_fire"] = friendly_fire
                learned_before = len(bot.learned)
                decision = bot.decide(detections, stats["ammo"], frame=frame,
                                      classifier=classifier, health=stats["health"])
                if len(bot.learned) > learned_before:
                    phrase, verdict = bot.learned[-1]
                    print(f"  Schritt {steps}: gelernt -> {verdict.upper()}: {phrase!r} "
                          f"(Prompt jetzt {len(bot.classes)} Klassen)", flush=True)

                # Unbekannte tauchen in festem Abstand auf, ausgewuerfelt aus
                # UNKNOWN_ACTORS ueber den Zufallsgenerator der Episode.
                if unknowns_left > 0 and steps >= next_unknown:
                    # wake=True: feindliche Akteure schlafen, bis Laerm sie
                    # weckt - ohne das greift keiner an und die
                    # Verhaltensbeobachtung liefe ins Leere.
                    spawned = env.spawn_unknown(wake=True)
                    next_unknown = steps + args.unknown_every
                    if spawned:
                        unknowns_left -= 1
                        print(f"  Schritt {steps}: unbekanntes Wesen erscheint "
                              f"({spawned[0]})", flush=True)

                # Nachschub, damit ein langer Lauf nicht ins Leere laeuft.
                if waves_left > 0 and monsters_alive <= 1:
                    env.spawn_wave()
                    extra_m, extra_h = env.alive_counts()
                    monsters_spawned += max(0, extra_m - monsters_alive)
                    humans_spawned += max(0, extra_h - humans_alive)
                    waves_left -= 1

                canvas = overlay.render(frame, detections, decision, stats,
                                        detector.last_latency, bot.classes, bot,
                                        truth, show_gt, muzzle_flash=flash)
                # Eine Beobachtung dauert im Spiel nur Sekundenbruchteile.
                # Fuer eine Aufzeichnung, die jemand ansehen soll, muessen die
                # Schluesselmomente laenger stehen.
                repeat = 1
                if args.emphasis:
                    if decision.verdict:
                        repeat = args.emphasis * 3
                    elif decision.state == "OBSERVING":
                        repeat = args.emphasis
                canvas_v = (vertical.render(
                    frame, detections, decision, stats, detector.last_latency,
                    bot, truth, show_gt, muzzle_flash=flash)
                    if writer_v is not None else None)
                for _ in range(repeat):
                    if writer is not None:
                        writer.write(canvas)
                    if writer_v is not None:
                        writer_v.write(canvas_v)
                if not args.no_display:
                    cv2.imshow(WINDOW, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    while paused and not quit_all:
                        key = cv2.waitKey(50) & 0xFF
                        paused, show_gt, quit_all = _handle_key(key, paused, show_gt, quit_all)
                    paused, show_gt, quit_all = _handle_key(key, paused, show_gt, quit_all)
                    if quit_all:
                        break
                    if key == ord("r"):
                        break

                # Eigenbeschuss zaehlen: Schuss abgegeben, waehrend laut Engine
                # ein Mensch im Schusskorridor stand.
                if decision.attack and _human_in_corridor(truth, bot, env.width):
                    friendly_fire += 1

                flash = decision.attack
                recording = writer is not None or writer_v is not None
                if args.smooth and recording:
                    # Zwischenbilder: das Spiel Tic fuer Tic weiterlaufen
                    # lassen und die Boxen um die inzwischen gedrehte Strecke
                    # mitfuehren, damit sie am Objekt kleben bleiben.
                    for sub, turned in env.step_capture(
                            turn_deg=decision.turn_deg, attack=decision.attack,
                            forward=decision.forward, backward=decision.backward,
                            tics=args.tics):
                        offset = pixel_shift(turned, env.deg_per_pixel())
                        if writer is not None:
                            writer.write(overlay.render(
                                sub, detections, decision, stats,
                                detector.last_latency, bot.classes, bot,
                                truth, show_gt, muzzle_flash=flash, shift=offset))
                        if writer_v is not None:
                            writer_v.write(vertical.render(
                                sub, detections, decision, stats,
                                detector.last_latency, bot, truth, show_gt,
                                muzzle_flash=flash, shift=offset))
                else:
                    env.step(turn_deg=decision.turn_deg, attack=decision.attack,
                             forward=decision.forward, backward=decision.backward,
                             tics=args.tics)

                steps += 1
                if args.steps and steps >= args.steps:
                    break
                if steps % 50 == 0:
                    print(f"  Schritt {steps}: {bot.shots} Schuesse, {bot.holds} Sperren, "
                          f"{stats['monsters_killed']} Monster erlegt, "
                          f"{humans_alive}/{humans_spawned} Menschen am Leben", flush=True)

            if quit_all:
                break
            _report(episode, args.episodes, steps, bot, env,
                    monsters_spawned, humans_spawned, friendly_fire)
    finally:
        if writer is not None:
            writer.release()
            print(f"Aufzeichnung: {args.record}")
        if writer_v is not None:
            writer_v.release()
            print(f"Aufzeichnung hochkant: {args.record_vertical}")
        cv2.destroyAllWindows()
        env.close()


def _human_in_corridor(truth, bot, width) -> bool:
    half = bot.fire_tolerance_px + bot.safety_margin_px
    left, right = width / 2 - half, width / 2 + half
    return any(b.kind == "human" and b.x2 >= left and b.x1 <= right for b in truth)


def _report(episode, episodes, steps, bot, env, monsters_spawned, humans_spawned, friendly_fire):
    monsters_alive, humans_alive = env.alive_counts()
    print(f"\nEpisode {episode + 1}/{episodes} - {steps} Schritte")
    print(f"  Monster:  {monsters_spawned - monsters_alive}/{monsters_spawned} erlegt")
    print(f"  Menschen: {humans_alive}/{humans_spawned} am Leben")
    print(f"  Schuesse: {bot.shots}, Feuer gesperrt: {bot.holds}")
    print(f"  Schuesse mit Mensch im Korridor (Engine-Wahrheit): {friendly_fire}", flush=True)


def _handle_key(key, paused, show_gt, quit_all):
    if key in (ord("q"), 27):
        quit_all = True
    elif key == ord("g"):
        show_gt = not show_gt
    elif key == ord(" "):
        paused = not paused
    return paused, show_gt, quit_all


def _make_writer(path, width, height, fps):
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


if __name__ == "__main__":
    main()
