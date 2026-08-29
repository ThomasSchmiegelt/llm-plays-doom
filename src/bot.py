"""Ziel- und Schusslogik auf Basis der erkannten Boxen.

Die Regeln sind bewusst einfach und deterministisch: das Sehen uebernimmt das
Vision-Language-Modell, das Handeln ist nachvollziehbare Geometrie.

Der Agent schiesst nur auf Objekte, die das Modell als feindlich gelabelt hat.
Weil die Klassifikation nicht perfekt ist, kommt eine zweite Sicherung dazu:
steht ein als Mensch erkanntes Objekt im Schusskorridor, wird nicht gefeuert -
auch dann nicht, wenn dahinter ein gueltiges Ziel steht.
"""
from dataclasses import dataclass, field, replace

from detector import (BROAD_CLASSES, HOSTILE_CLASSES, PROTECTED_CLASSES,
                      Detection, _iou)
from tracking import Tracker, Track


@dataclass
class Decision:
    turn_deg: float
    attack: bool
    forward: bool
    backward: bool
    target: Detection | None
    state: str          # SCANNING | AIMING | FIRING | HOLD_FIRE | OBSERVING
    reason: str
    blockers: list[Detection] = field(default_factory=list)
    # Beim Beobachten: die verfolgte Spur und der Stand der Abstimmung.
    observed: Track | None = None
    progress: tuple[int, int] | None = None
    # Gesetzt genau in dem Schritt, in dem die Einordnung faellt.
    verdict: str | None = None
    verdict_source: str | None = None   # "Aussehen" oder "Verhalten"


class ProtectionMemory:
    """Haelt erkannte Menschen ueber mehrere Frames fest.

    Der Detektor uebersieht einen Marine gelegentlich fuer einen einzelnen
    Frame. Ein Sicherheitsmechanismus darf davon nicht abhaengen, deshalb
    bleibt jede geschuetzte Box einige Schritte lang gueltig.

    Weil der Agent zwischen den Frames dreht, wandern die Objekte im Bild.
    Der Drehwinkel ist bekannt, also laesst sich die Verschiebung exakt
    herausrechnen: eine Drehung nach rechts schiebt alles nach links.
    """

    MAX_ENTRIES = 8

    def __init__(self, deg_per_pixel: float, width: int, ttl: int = 4):
        self.deg_per_pixel = deg_per_pixel
        self.width = width
        self.ttl = ttl
        self._entries: list[tuple[Detection, int]] = []

    def advance(self, turn_deg: float):
        shift = -turn_deg / self.deg_per_pixel
        moved = []
        for det, age in self._entries:
            if age + 1 >= self.ttl:
                continue
            box = replace(det, x1=int(det.x1 + shift), x2=int(det.x2 + shift))
            if box.x2 < 0 or box.x1 > self.width:
                continue  # aus dem Bild gedreht
            moved.append((box, age + 1))
        self._entries = moved

    def update(self, fresh: list[Detection]):
        """Frische Erkennungen ersetzen die erinnerten an derselben Stelle.

        Ohne Zusammenlegen wuerden sich ueber mehrere Frames Dutzende leicht
        versetzter Boxen desselben Marines ansammeln, und der Agent kaeme nie
        mehr zum Schuss.
        """
        merged: list[tuple[Detection, int]] = []
        # Frische zuerst, dann die aelteren - so gewinnt immer die juengste
        # Box an einer Stelle.
        for det, age in [(d, 0) for d in fresh] + self._entries:
            if any(_iou(det, other) >= 0.3 for other, _ in merged):
                continue
            merged.append((det, age))
        self._entries = merged[:self.MAX_ENTRIES]

    def boxes(self) -> list[Detection]:
        return [det for det, _ in self._entries]

    def remembered(self) -> list[Detection]:
        """Nur die Boxen, die in diesem Frame nicht frisch erkannt wurden."""
        return [det for det, age in self._entries if age > 0]


class AimBot:
    def __init__(self, width: int, height: int, deg_per_pixel: float,
                 fire_tolerance_px: float = 6.0, scan_step_deg: float = 22.0,
                 safety_margin_px: float = 26.0, can_move: bool = False):
        self.width = width
        self.height = height
        self.deg_per_pixel = deg_per_pixel
        # 6 statt 14 px, gemessen mit src/eval_policy.py: enger zielen senkt den
        # Munitionsverbrauch von 20 auf 6 Schuss, hebt die Trefferquote von 20 %
        # auf 50 % und beseitigt den Eigenbeschuss - bei gleichem Endstand.
        # Wichtig: das gilt nur ZUSAMMEN mit den Sicherungen. Ohne sie wird es
        # schlechter, weil der Agent dann auch auf Menschen praeziser zielt.
        self.fire_tolerance_px = fire_tolerance_px
        self.scan_step_deg = scan_step_deg
        # Halber Schusskorridor: so weit neben dem Fadenkreuz darf kein
        # geschuetztes Objekt stehen, sonst wird das Feuer eingestellt.
        self.safety_margin_px = safety_margin_px
        self.can_move = can_move
        self._scan_dir = 1.0
        self._pending_turn = 0.0
        self.memory = ProtectionMemory(deg_per_pixel, width)
        # Unbekanntes wird ueber mehrere Frames verfolgt, damit die
        # Abstimmung des Klassifikators demselben Wesen zugeordnet bleibt.
        self.unknowns = Tracker(deg_per_pixel, width, ttl=8)
        # Der Wortschatz waechst waehrend des Laufs - deshalb gehoert er dem
        # Agenten und nicht dem Modul.
        self.hostile_classes = list(HOSTILE_CLASSES)
        self.protected_classes = list(PROTECTED_CLASSES)
        self.broad_classes = list(BROAD_CLASSES)
        self.learned: list[tuple[str, str]] = []
        # Aus: jedes Wesen wird einzeln beobachtet und eingeordnet, der Prompt
        # bleibt unveraendert. Sinnvoll fuer Aufnahmen, in denen der Vorgang
        # mehrfach zu sehen sein soll - eine gelernte Beschreibung wie
        # "man in dark uniform holding a gun" deckt sonst schon nach dem
        # ersten Mal jede weitere gepanzerte Figur ab.
        self.learning = True
        # Aus: geschuetzte Objekte werden weiterhin erkannt und blau markiert,
        # gelten aber trotzdem als Ziel. Damit laesst sich Wahrnehmung von
        # Regelwerk trennen - beide Agenten sehen dasselbe, nur einer haelt
        # sich daran.
        self.respect_protection = True
        # Aus: Beobachten haelt den Agenten nicht an. Er zielt und feuert
        # weiter auf bekannte Ziele, waehrend die Hypothesenabfrage nebenher
        # laeuft. Das Unbekannte sperrt weiterhin den Schusskorridor - es wird
        # also nicht beschossen, nur nicht mehr angestarrt.
        #
        # Mit Unterbrechung wird der Agent bei mehreren Auftritten praktisch
        # handlungsunfaehig: gemessen 4 Schuss und 0 Treffer ueber 160 Schritte.
        self.observe_interrupts = True
        # Wie lange ein Wesen unauffaellig bleiben muss, bis es als harmlos
        # gilt. Gemessen: der erste Schaden kam im Mittel nach 7,3 Schritten,
        # spaetestens nach 11 - darueber liegt die Geduld mit Reserve.
        self.patience = 14
        self.shots = 0
        self.holds = 0
        self.observations = 0
        self.step_index = 0

    def filter_visible(self, detections: list[Detection]) -> list[Detection]:
        """Entfernt die eigene Waffe. Anzeige und Entscheidung sollen
        dieselbe Liste sehen."""
        return [d for d in detections if not self._is_own_weapon(d)]

    @property
    def classes(self) -> list[str]:
        """Der aktuelle Prompt - waechst, wenn der Agent etwas gelernt hat."""
        return self.hostile_classes + self.protected_classes + self.broad_classes

    def learn(self, phrase: str, verdict: str) -> bool:
        """Nimmt eine Beschreibung dauerhaft in den Prompt auf.

        Ab jetzt erkennt der Detektor diese Art unmittelbar, ohne dass sie
        noch einmal beobachtet werden muss.
        """
        if not self.learning:
            return False
        if phrase in self.hostile_classes or phrase in self.protected_classes:
            return False
        target = self.hostile_classes if verdict == "dangerous" else self.protected_classes
        target.append(phrase)
        self.learned.append((phrase, verdict))
        return True

    def decide(self, visible: list[Detection], ammo: float,
               frame=None, classifier=None, health: float = 100.0) -> Decision:
        """`visible` muss bereits durch filter_visible() gelaufen sein.

        Sind `frame` und `classifier` gesetzt, wird ein noch nicht
        eingeordnetes unbekanntes Wesen in diesem Schritt beobachtet.
        """
        self.step_index += 1
        # Die im letzten Schritt ausgefuehrte Drehung aus dem Gedaechtnis
        # herausrechnen, dann die frischen Erkennungen einpflegen.
        self.memory.advance(self._pending_turn)
        self.memory.update([d for d in visible if d.protected])
        self.unknowns.advance(self._pending_turn)
        self.unknowns.update([d for d in visible if d.unknown],
                             step=self.step_index, health=health)

        decision = self._observe_unknown(visible, frame, classifier)
        if decision is not None:
            return decision

        # Eingeordnetes Unbekanntes wird behandelt wie seinesgleichen:
        # gefaehrlich ist Ziel, harmlos ist geschuetzt. Noch Unentschiedenes
        # sperrt vorsorglich den Korridor.
        judged_dangerous, judged_harmless, undecided = [], [], []
        for track in self.unknowns.tracks():
            (judged_dangerous if track.verdict == "dangerous" else
             judged_harmless if track.verdict == "harmless" else
             undecided).append(track.box)

        # Ziele nur aus frischen Erkennungen, Schutz auch aus dem Gedaechtnis.
        if self.respect_protection:
            protected = self.memory.boxes() + judged_harmless + undecided
            targets = ([d for d in visible if not d.protected and not d.unknown]
                       + judged_dangerous)
        else:
            # Kein Regelwerk: alles ist Ziel, auch was noch gar nicht
            # eingeordnet ist. Das Beobachten entfaellt damit ebenfalls.
            protected = []
            targets = list(visible) + judged_dangerous

        if not targets:
            blockers = self._in_corridor(protected)
            if blockers:
                return self._remember(Decision(
                    self.scan_step_deg * self._scan_dir, False, False, False, None,
                    "SCANNING", f"nur Geschuetztes im Bild ({len(protected)}) - weitersuchen",
                    blockers))
            return self._remember(Decision(
                self.scan_step_deg * self._scan_dir, False, False, False, None,
                "SCANNING", "kein Ziel erkannt - Umgebung absuchen"))

        target = max(targets, key=self._threat)
        dx = target.cx - self.width / 2
        turn_deg = dx * self.deg_per_pixel

        if abs(dx) > self.fire_tolerance_px:
            # Beim Drehen die Richtung fuer die naechste Suche merken.
            self._scan_dir = 1.0 if turn_deg > 0 else -1.0
            return self._remember(Decision(turn_deg, False, False, False, target, "AIMING",
                                            f"drehe {turn_deg:+.1f} Grad auf '{target.label}'"))

        # Ziel steht im Fadenkreuz - vor dem Schuss die Schusslinie pruefen.
        corridor = self._in_corridor(protected)
        blockers = corridor + [d for d in protected
                               if d not in corridor and self._overlaps(d, target)]
        if blockers:
            self.holds += 1
            unknown_blocker = any(b.unknown for b in blockers)
            remembered = any(b in self.memory.remembered() for b in blockers)
            who = "Unbekanntes" if unknown_blocker else "Mensch"
            note = " (aus dem Gedaechtnis)" if remembered else ""
            return self._remember(Decision(0.0, False, False, False, target, "HOLD_FIRE",
                                            f"{who} in der Schusslinie{note} - Feuer eingestellt",
                                            blockers))
        if ammo <= 0:
            return self._remember(Decision(0.0, False, False, self.can_move, target, "AIMING",
                                            "Ziel erfasst, aber keine Munition"))

        self.shots += 1
        return self._remember(Decision(0.0, True, False, False, target, "FIRING",
                                        f"'{target.label}' im Fadenkreuz (dx={dx:+.0f} px) - Feuer"))

    def _observe_unknown(self, visible, frame, classifier) -> Decision | None:
        """Hinschauen, solange noch nicht klar ist, was da steht.

        Der Agent dreht sich auf das Wesen, damit der Ausschnitt fuer die
        Abfrage etwas taugt, und stellt das Feuer ein. Erst wenn die
        Abstimmung steht, laeuft der normale Ablauf weiter.
        """
        if frame is None or classifier is None or not self.respect_protection:
            return None
        # Ein Urteil aus dem Aussehen ist vorlaeufig: es bleibt beobachtet,
        # bis das Verhalten es bestaetigt oder widerlegt. Nur ein
        # Verhaltensurteil ist endgueltig - es liegt zu 100 % richtig, das
        # Aussehen auf den entscheidenden Faellen zu 0 %.
        pending = [t for t in self.unknowns.tracks()
                   if t.verdict_source != "Verhalten" and t.fresh]
        if not pending:
            return None

        # Immer nur eines auf einmal - das groesste, also naechste.
        focus = max(pending, key=lambda t: t.box.area)

        # Das Verhalten schlaegt das Aussehen. Gemessen auf denselben zehn
        # Rollen: die Regel "hat Schaden verursacht" liegt zu 100 % richtig,
        # das Aussehen auf den entscheidenden Faellen zu 0 %. Drei Marines
        # sehen gleich aus und sind Zivilist, Verbuendeter und Feind - da
        # kann kein Bild helfen.
        behaviour = self._behaviour_verdict(focus)
        if behaviour is not None:
            revised = focus.verdict is not None and focus.verdict != behaviour
            focus.verdict = behaviour
            focus.verdict_source = "Verhalten"
            grund = ("hat Schaden verursacht" if behaviour == "dangerous"
                     else f"greift seit {self.patience} Schritten nicht an")
            if revised:
                grund = "Aussehen widerlegt - " + grund
            return self._remember(Decision(
                0.0, False, False, False, None, "OBSERVING",
                f"{grund}", [focus.box], observed=focus,
                progress=classifier.progress(focus), verdict=behaviour,
                verdict_source="Verhalten"))

        if focus.verdict is not None:
            # Aussehen hat schon geraten - jetzt nur noch zusehen.
            return None if not self.observe_interrupts else self._remember(Decision(
                0.0, False, False, False, None, "OBSERVING",
                f"vorlaeufig {focus.verdict} - warte auf das Verhalten",
                [focus.box], observed=focus,
                progress=(len(focus.history), self.patience)))

        verdict = classifier.observe(frame, focus)
        self.observations += 1

        if verdict is None and not self.observe_interrupts:
            # Nebenher beobachten: die Probe ist gezaehlt, der normale Ablauf
            # geht weiter.
            return None

        if verdict is not None:
            focus.verdict_source = "Aussehen"
            learned = self.learn(verdict.phrase, verdict.kind)
            word = "GEFAEHRLICH" if verdict.kind == "dangerous" else "HARMLOS"
            note = " - in den Prompt aufgenommen" if learned else ""
            return self._remember(Decision(
                0.0, False, False, False, None, "OBSERVING",
                f"'{verdict.phrase}' - {verdict.confidence:.0%} aus "
                f"{verdict.probes} Proben{note}",
                [focus.box], observed=focus,
                progress=classifier.progress(focus), verdict=verdict.kind,
                verdict_source="Aussehen"))

        dx = focus.box.cx - self.width / 2
        turn_deg = dx * self.deg_per_pixel
        seen, window = classifier.progress(focus)
        return self._remember(Decision(
            turn_deg, False, False, False, None, "OBSERVING",
            f"unbekanntes Wesen - beobachte ({seen}/{window})",
            [focus.box], observed=focus, progress=(seen, window)))

    def _behaviour_verdict(self, track: Track) -> str | None:
        """Urteil aus dem Verhalten, oder None solange nichts feststeht.

        Schaden ist ein spaetes, aber eindeutiges Zeichen - man weiss es erst,
        wenn es passiert ist. Umgekehrt gilt: wer lange genug zusieht, ohne
        angegriffen zu werden, darf das Wesen als harmlos abhaken.
        """
        if track.harmed():
            return "dangerous"
        if len(track.history) >= self.patience:
            return "harmless"
        return None

    def _remember(self, decision: Decision) -> Decision:
        """Merkt die Drehung, damit das Gedaechtnis sie naechste Runde
        herausrechnen kann."""
        self._pending_turn = decision.turn_deg
        return decision

    # -- Hilfsfunktionen -------------------------------------------------
    def _in_corridor(self, protected: list[Detection]) -> list[Detection]:
        """Geschuetzte Objekte, die den senkrechten Streifen um das
        Fadenkreuz beruehren - dort schlagen die Schuesse ein."""
        left = self.width / 2 - self.fire_tolerance_px - self.safety_margin_px
        right = self.width / 2 + self.fire_tolerance_px + self.safety_margin_px
        return [d for d in protected if d.x2 >= left and d.x1 <= right]

    @staticmethod
    def _overlaps(a: Detection, b: Detection) -> bool:
        """Ueberdecken sich die Boxen? Dann ist unklar, was vorne steht."""
        return a.x2 > b.x1 and a.x1 < b.x2 and a.y2 > b.y1 and a.y1 < b.y2

    def _threat(self, d: Detection) -> float:
        """Groesse zaehlt am meisten (naeher = gefaehrlicher), Naehe zur
        Bildmitte gibt einen Bonus, damit der Agent nicht zwischen zwei
        gleich grossen Zielen hin- und herspringt."""
        centrality = 1.0 - min(1.0, abs(d.cx - self.width / 2) / (self.width / 2))
        return d.area * (1.0 + 0.35 * centrality)

    def _is_own_weapon(self, d: Detection) -> bool:
        """Die eigene Waffe klebt mittig am unteren Bildrand und wird von
        Detektoren gern als Gegner gemeldet."""
        touches_bottom = d.y2 >= self.height * 0.86
        horizontally_centered = abs(d.cx - self.width / 2) < self.width * 0.26
        big_enough = d.area > 0.015 * self.width * self.height
        return touches_bottom and horizontally_centered and big_enough
